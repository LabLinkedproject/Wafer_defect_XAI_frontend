"""
training/fetch_wm811k.py
WM-811K 전체를 내려받아 맵마다 불량 좌표만 남긴다. 라벨 없는 63만 장도 추가 라벨 후보로 보관한다.

출처는 Hugging Face 의 khaubeil/wm811k-demo-data 다. 원본 LSWMD.pkl 을 제3자가 샤드로 변환한 것이어서,
라벨 수를 보고서의 원본 통계와 대조해 내용이 맞는지 확인한 뒤 저장한다.

실행: ./venv-train/bin/python -m training.fetch_wm811k
"""
from __future__ import annotations
import concurrent.futures as cf
import gzip
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import durable                            # noqa: E402
import training_status                                  # noqa: E402

URL = "https://huggingface.co/datasets/khaubeil/wm811k-demo-data/resolve/main/analysis/shard-{}.json.gz"
SHARDS = 2048
OUTPUT = "datasets/wm811k_labeled.npz"
PARTS = "datasets/wm811k_parts"
WORKERS = 3                  # 압축 해제와 JSON 해석이 병목이라 프로세스를 나눈다
OFFLINE_LIMIT = 600          # 네트워크가 이 시간 동안 없으면 받은 샤드로 진행한다. 샤드는 웨이퍼를 고르게 나눈 것이라 일부만으로도 편향이 없다
NORMAL_STRIDE = 3            # 패턴 없는 맵은 14만 장이라 3장에 1장만 남긴다 (라벨 수 대조는 전체로 한다)
NAMES = {"Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge Ring": 3, "Edge-Ring": 3, "Loc": 4, "Random": 5,
         "Scratch": 6, "Near-full": 7, "Near Full": 7, "None": 8, "none": 8}
# 노트북 wm-811k-wafermap.ipynb 출력의 원본 통계
EXPECTED = {"labeled": 172950, "pattern": 25519, "per_class": [4294, 555, 5189, 9680, 3593, 866, 1193, 149]}


def table(rows: list[dict], prefix: str = "") -> dict:
    xy = [np.array(r["mask"]["fail_xy"], np.uint16).reshape(-1, 2) for r in rows]
    out = {"ids": np.array([r["sample_id"] for r in rows], np.int32),
           "shapes": np.array([(r["shape"]["h"], r["shape"]["w"]) for r in rows], np.uint16).reshape(-1, 2),
           "ellipse": np.array([[r["mask"][c] for c in "cx cy a b".split()] for r in rows], np.float32).reshape(-1, 4),
           "lengths": np.array([len(a) for a in xy], np.int32),
           "xy": np.concatenate(xy) if xy else np.zeros((0, 2), np.uint16)}
    return {prefix + k: v for k, v in out.items()}


def fetch(k: int):
    """샤드 하나를 받아 필요한 값만 디스크에 남긴다. 받지 못하면 None"""
    part = f"{PARTS}/{k:04d}.npz"
    if os.path.exists(part):
        try:
            return np.load(part)["counts"]
        except Exception:                                # 전원이 끊긴 순간의 파일
            os.remove(part)
    try:
        with urllib.request.urlopen(URL.format(k), timeout=30) as r:
            data = json.loads(gzip.decompress(r.read()))
    except Exception:
        return None
    rows, unlabeled, counts = [], [], np.zeros(9, np.int64)
    for s in data.values():
        name = s["ground_truth_name"]
        if name is None:
            unlabeled.append(s)
            continue
        counts[NAMES[name]] += 1
        if NAMES[name] != 8 or s["sample_id"] % NORMAL_STRIDE == 0:
            rows.append(s)
    del data
    tmp = part + ".tmp.npz"
    np.savez(tmp, counts=counts, labels=np.array([NAMES[r["ground_truth_name"]] for r in rows], np.uint8), **table(rows), **table(unlabeled, "u_"))
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, part)
    return counts


def merge(shards):
    tables = {k: [] for k in ("ids", "labels", "shapes", "ellipse", "lengths", "xy")}
    for k in shards:
        d = np.load(f"{PARTS}/{k:04d}.npz")
        for name, items in tables.items():
            items.append(d[name])
    out = {k: np.concatenate(v) for k, v in tables.items()}
    order = np.argsort(out["ids"])
    starts = np.concatenate([[0], np.cumsum(out["lengths"])])
    xy = np.concatenate([out["xy"][starts[i]:starts[i + 1]] for i in order])
    np.savez(OUTPUT + ".tmp.npz", ids=out["ids"][order], labels=out["labels"][order], shapes=out["shapes"][order], ellipse=out["ellipse"][order],
             offsets=np.concatenate([[0], np.cumsum(out["lengths"][order])]), xy=xy)
    os.replace(OUTPUT + ".tmp.npz", OUTPUT)


def main():
    os.makedirs(PARTS, exist_ok=True)
    durable.catch_stop_signal()
    results, last_ok = {}, time.time()
    pending = list(range(SHARDS))
    durable.event(f"내려받기 시작: 남은 샤드 확인 중")
    with cf.ProcessPoolExecutor(WORKERS) as ex:
        while pending and not durable.stop_requested():
            batch, pending = pending[:WORKERS * 8], pending[WORKERS * 8:]
            for k, counts in zip(batch, ex.map(fetch, batch)):
                if counts is None:
                    pending.append(k)
                else:
                    results[k] = counts
                    last_ok = time.time()
            durable.wait_if_hot("fetch_wm811k", len(results), SHARDS, "샤드")
            offline = time.time() - last_ok
            if offline > 5:
                training_status.report("fetch_wm811k", len(results), SHARDS, f"네트워크 대기 {offline:.0f}초, 받은 샤드는 보존됨", unit="샤드")
                time.sleep(10)
                if offline > OFFLINE_LIMIT:
                    break
            else:
                total = sum(results.values())
                training_status.report("fetch_wm811k", len(results), SHARDS, f"라벨 웨이퍼 {int(total.sum()):,}장 확인", unit="샤드")
    if durable.stop_requested():
        durable.event(f"내려받기 멈춤: {len(results)}/{SHARDS}")
        sys.exit(durable.EXIT_PAUSED)
    counts = sum(results.values())
    complete = len(results) == SHARDS
    print("샤드", len(results), "라벨", counts.sum(), "패턴", counts[:8].sum(), "클래스별", counts.tolist())
    if complete:
        assert counts.sum() == EXPECTED["labeled"] and counts[:8].tolist() == EXPECTED["per_class"], "원본 통계와 다릅니다"
    else:
        durable.event(f"네트워크가 {OFFLINE_LIMIT // 60}분 없어 받은 샤드 {len(results)}/{SHARDS} 로 진행")
    merge(sorted(results))
    durable.write_json("models/data_status.json", {"shards": len(results), "total": SHARDS, "complete": complete})
    durable.event(f"내려받기 완료: {len(results)}/{SHARDS} 샤드, 라벨 {int(counts.sum()):,}장")
    print("저장", OUTPUT, os.path.getsize(OUTPUT) // 1024 // 1024, "MB")


if __name__ == "__main__":
    main()
