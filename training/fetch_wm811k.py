"""
training/fetch_wm811k.py
WM-811K 라벨 데이터를 내려받아 불량 좌표만 남긴다. 파일이 아니라 메모리에서 바로 풀어 쓴다.

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
import training_status                                  # noqa: E402

URL = "https://huggingface.co/datasets/khaubeil/wm811k-demo-data/resolve/main/analysis/shard-{}.json.gz"
SHARDS = 2048
OUTPUT = "datasets/wm811k_labeled.npz"
PARTS = "datasets/wm811k_parts"
NORMAL_STRIDE = 3            # 패턴 없는 맵은 14만 장이라 3장에 1장만 남긴다 (라벨 수 대조는 전체로 한다)
NAMES = {"Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge Ring": 3, "Edge-Ring": 3, "Loc": 4, "Random": 5,
         "Scratch": 6, "Near-full": 7, "Near Full": 7, "None": 8, "none": 8}
# 노트북 wm-811k-wafermap.ipynb 출력의 원본 통계
EXPECTED = {"labeled": 172950, "pattern": 25519, "per_class": [4294, 555, 5189, 9680, 3593, 866, 1193, 149]}


def fetch(k: int) -> dict:
    """샤드 하나를 받아 필요한 값만 디스크에 남긴다. 메모리에 쌓지 않아 보드가 버티고, 끊겨도 이어 받는다."""
    part = f"{PARTS}/{k:04d}.npz"
    if not os.path.exists(part):
        for attempt in range(6):
            try:
                with urllib.request.urlopen(URL.format(k), timeout=60) as r:
                    data = json.loads(gzip.decompress(r.read()))
                break
            except Exception:                            # 일시적 끊김·요청 제한
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        rows, counts = [], np.zeros(9, np.int64)
        for s in data.values():
            name = s["ground_truth_name"]
            if name is None:
                continue
            counts[NAMES[name]] += 1
            if NAMES[name] == 8 and s["sample_id"] % NORMAL_STRIDE:
                continue
            rows.append(s)
        del data
        ids = np.array([r["sample_id"] for r in rows], np.int32)
        xy = [np.array(r["mask"]["fail_xy"], np.uint16).reshape(-1, 2) for r in rows]
        tmp = part + ".tmp.npz"
        np.savez(tmp, ids=ids, counts=counts, labels=np.array([NAMES[r["ground_truth_name"]] for r in rows], np.uint8),
                 shapes=np.array([(r["shape"]["h"], r["shape"]["w"]) for r in rows], np.uint16).reshape(-1, 2),
                 ellipse=np.array([[r["mask"][c] for c in "cx cy a b".split()] for r in rows], np.float32).reshape(-1, 4),
                 lengths=np.array([len(a) for a in xy], np.int32),
                 xy=np.concatenate(xy) if xy else np.zeros((0, 2), np.uint16))
        os.replace(tmp, part)
    return np.load(part)["counts"]


def merge():
    tables = {k: [] for k in ("ids", "labels", "shapes", "ellipse", "lengths", "xy")}
    for k in range(SHARDS):
        d = np.load(f"{PARTS}/{k:04d}.npz")
        for name, items in tables.items():
            items.append(d[name])
    out = {k: np.concatenate(v) for k, v in tables.items()}
    order = np.argsort(out["ids"])
    starts = np.concatenate([[0], np.cumsum(out["lengths"])])
    xy = np.concatenate([out["xy"][starts[i]:starts[i + 1]] for i in order])
    np.savez(OUTPUT, ids=out["ids"][order], labels=out["labels"][order], shapes=out["shapes"][order], ellipse=out["ellipse"][order],
             offsets=np.concatenate([[0], np.cumsum(out["lengths"][order])]), xy=xy)


def main():
    os.makedirs(PARTS, exist_ok=True)
    counts, done = np.zeros(9, np.int64), 0
    training_status.report("fetch_wm811k", 0, SHARDS, "샤드 내려받는 중", unit="샤드")
    with cf.ThreadPoolExecutor(3) as ex:
        for part in ex.map(fetch, range(SHARDS)):
            counts += part
            done += 1
            if done % 4 == 0:
                training_status.report("fetch_wm811k", done, SHARDS, f"라벨 웨이퍼 {counts.sum():,}장 확인", unit="샤드")
    print("라벨", counts.sum(), "패턴", counts[:8].sum(), "클래스별", counts.tolist())
    assert counts.sum() == EXPECTED["labeled"] and counts[:8].tolist() == EXPECTED["per_class"], "원본 통계와 다릅니다"
    merge()
    print("저장", OUTPUT, os.path.getsize(OUTPUT) // 1024 // 1024, "MB")


if __name__ == "__main__":
    main()
