"""
training/pseudo_label.py
라벨 없는 WM-811K 맵을 우리 44다이 스케일로 내려 학습 모델로 판정하고, 확실한 것만 추가 라벨로 쓴다.

스케일다운은 맵 수를 늘리지 않는다. 라벨 없는 맵 63만 장이 실제로 늘어나는 원천이다.
- 학습 구간의 맵만 쓴다. 검증·시험 구간과 그 경계의 맵은 쓰지 않는다.
- 구역 크기가 다른 두 입력(1×1, 3×3)이 같은 단일 패턴을 높은 확률로 가리킬 때만 받는다.
- 맵 순서를 섞고 패턴별 상한을 두어 특정 로트에 쏠리지 않게 한다.

추가 라벨은 모델의 판단이지 정답이 아니다. 시험에는 쓰지 않으며, 이 모델의 채택은 검증 점수로만 정한다.

실행: ./venv-train/bin/python -m training.pseudo_label
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import durable                           # noqa: E402
from training.evaluate import load_net, sigmoid        # noqa: E402
from training.fetch_wm811k import PARTS, SHARDS        # noqa: E402
from training.model import Geometry, quadrat_map       # noqa: E402
from training.prepare_data import (BLOCK, MIN_DIAMETER_DIES, PURGE, block_side, structure, to_grid)  # noqa: E402
from training.train import CKPT, DATA_DIR, TRAIN, to_float            # noqa: E402
import training_status                                 # noqa: E402
import wafer_raster as R                               # noqa: E402

CONFIDENT = 0.90              # 두 입력 모두 이 확률 이상
OTHERS_BELOW = 0.20           # 나머지 패턴은 이 확률 미만
NORMAL_BELOW = 0.05           # 정상 맵은 모든 패턴이 이 확률 미만
PER_CLASS = 6000
NORMAL_CAP = 6000
BATCH = 256


def usable(ids: np.ndarray, ellipse: np.ndarray) -> np.ndarray:
    blocks, pos = ids // BLOCK, ids % BLOCK
    sides = {int(b): block_side(b) for b in set(blocks.tolist()) | set((blocks - 1).tolist()) | set((blocks + 1).tolist())}
    ok = np.array([sides[b] == TRAIN for b in blocks.tolist()])
    for near, off in ((pos < PURGE, -1), (pos >= BLOCK - PURGE, 1)):
        ok &= ~(near & np.array([sides[b + off] != TRAIN for b in blocks.tolist()]))
    return ok & (2 * ellipse[:, 2:4].min(1) + 1 >= MIN_DIAMETER_DIES)


def rasters(d, picks):
    raw, core = np.zeros((2, len(picks), R.SIZE, R.SIZE), np.uint8)
    starts = np.concatenate([[0], np.cumsum(d["u_lengths"])])
    for n, i in enumerate(picks):
        h, w = d["u_shapes"][i]
        fail = np.zeros((h, w), np.uint8)
        pts = d["u_xy"][starts[i]:starts[i + 1]]
        fail[pts[:, 1], pts[:, 0]] = 1
        cx, cy, a, b = d["u_ellipse"][i]
        box = (cx - a, cx + a + 1, cy - b, cy + b + 1)
        raw[n], core[n] = to_grid(fail * 255, box), to_grid(structure(fail, np.pi * (a + 0.5) * (b + 0.5)) * 255, box)
    return raw, core


def main():
    with open("models/calibration_a.json") as f:
        t = json.load(f)["temperature"]
    with open(f"{DATA_DIR}/info.json") as f:
        patterns = json.load(f)["patterns"]
    net, _ = load_net(CKPT, len(patterns))
    net.eval()
    g = Geometry()
    caps = np.array([PER_CLASS] * len(patterns) + [NORMAL_CAP])
    got = np.zeros(len(patterns) + 1, int)
    kept_raw, kept_core, kept_y = [], [], []
    seen = 0
    order = np.random.default_rng(7).permutation(SHARDS)
    durable.catch_stop_signal()
    for k, shard in enumerate(order):
        if not os.path.exists(f"{PARTS}/{shard:04d}.npz"):
            continue                                     # 네트워크 없이 진행한 경우 일부 샤드가 없다
        if durable.stop_requested():
            sys.exit(durable.EXIT_PAUSED)
        durable.wait_if_hot("pseudo_label", k, SHARDS, "샤드")
        d = np.load(f"{PARTS}/{shard:04d}.npz")
        d = {n: d[n] for n in ("u_ids", "u_shapes", "u_ellipse", "u_lengths", "u_xy")}
        picks = np.where(usable(d["u_ids"], d["u_ellipse"]))[0]
        if len(picks):
            raw, core = rasters(d, picks)
            c = to_float(torch.from_numpy(core))
            with torch.no_grad():
                p1 = sigmoid(net(quadrat_map(c, g, 1)).numpy() / t)
                p3 = sigmoid(net(quadrat_map(c, g, 3)).numpy() / t)
            seen += len(picks)
            low, high = np.minimum(p1, p3), np.maximum(p1, p3)
            top = low.argmax(1)
            rows = np.arange(len(top))
            others = high.copy()
            others[rows, top] = 0
            sure = (low[rows, top] >= CONFIDENT) & (others.max(1) < OTHERS_BELOW)
            normal = high.max(1) < NORMAL_BELOW
            for i in np.where(sure | normal)[0]:
                cls = len(patterns) if normal[i] else top[i]
                if got[cls] >= caps[cls]:
                    continue
                got[cls] += 1
                y = np.zeros(len(patterns), np.uint8)
                if not normal[i]:
                    y[cls] = 1
                kept_raw.append(raw[i])
                kept_core.append(core[i])
                kept_y.append(y)
        if k % 16 == 0:
            training_status.report("pseudo_label", k, SHARDS, f"{seen:,}장 판정, {sum(got):,}장 채택", unit="샤드")
    np.save(f"{DATA_DIR}/pseudo_raw.npy", np.stack(kept_raw))
    np.save(f"{DATA_DIR}/pseudo_core.npy", np.stack(kept_core))
    np.save(f"{DATA_DIR}/pseudo_labels.npy", np.stack(kept_y))
    res = {"judged": seen, "kept": int(sum(got)), "per_class": dict(zip(patterns + ["Normal"], got.tolist()))}
    print(res, flush=True)
    durable.write_json("models/pseudo_label.json", res)
    durable.event(f"추가 라벨 {res['kept']:,}장 채택 ({res['judged']:,}장 판정)")


if __name__ == "__main__":
    main()
