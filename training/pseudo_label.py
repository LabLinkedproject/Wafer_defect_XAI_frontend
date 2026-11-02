"""
training/pseudo_label.py
라벨 없는 WM-811K 맵을 우리 44다이 스케일로 내려 학습 모델로 판정하고, 확실한 것만 추가 라벨로 쓴다.

스케일다운은 맵 수를 늘리지 않는다. 라벨 없는 맵 63만 장이 실제로 늘어나는 원천이다.
- 학습 구간의 맵만 쓴다. 검증·시험 구간과 그 경계의 맵은 쓰지 않는다.
- 3×3, 6×6 입력이 함께 한 패턴을 가리킬 때만 받는다. 기준값은 패턴마다 검증 분할(WM-811K)에서
  정밀도가 TARGET_PRECISION 이상이 되는 값으로 정한다. 기준을 만들 수 없는 패턴은 받지 않는다.
- 맵 순서를 섞고 패턴별 상한을 두어 특정 로트에 쏠리지 않게 한다.

추가 라벨은 모델의 판단이지 정답이 아니다. 시험에는 쓰지 않으며, 이 모델의 채택은 검증 점수로만 정한다.

실행: ./venv-train/bin/python -m training.pseudo_label
"""
from __future__ import annotations
import json
import os
import resource
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import durable                           # noqa: E402
from training.evaluate import load_net, sigmoid        # noqa: E402
from training.fetch_wm811k import PARTS, SHARDS        # noqa: E402
from training.model import Geometry, quadrat_map       # noqa: E402
from training.prepare_data import (BLOCK, MIN_DIAMETER_DIES, PURGE, block_side, structure, to_grid)  # noqa: E402
from training.train import CKPT, DATA_DIR, TRAIN, VAL, WM811K, load_images, load_meta, to_float  # noqa: E402
import training_status                                 # noqa: E402
import wafer_raster as R                               # noqa: E402

TARGET_PRECISION = 0.95
NORMAL_PRECISION = 0.98
MIN_HITS = 15                 # 정밀도를 믿을 수 있는 최소 표본
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


def scores(net, core, g, t):
    """3×3 과 6×6 입력의 확률"""
    c = to_float(torch.from_numpy(core))
    with torch.no_grad():
        return (sigmoid(net(quadrat_map(c, g, 3)).numpy() / t), sigmoid(net(quadrat_map(c, g, 6)).numpy() / t))


def fit_rules(net, g, t, patterns):
    """검증 분할에서 패턴별 채택 기준을 정하고, 예상 정밀도를 함께 돌려준다."""
    labels, split, source, _, _ = load_meta()
    va = np.where((split == VAL) & (source == WM811K))[0]
    core = load_images(va)[1]
    y = labels[va] > 0
    p3, p6 = np.zeros((2, len(va), len(patterns)))
    for i in range(0, len(va), BATCH):
        p3[i:i + BATCH], p6[i:i + BATCH] = scores(net, core[i:i + BATCH].numpy(), g, t)
    low, high = np.minimum(p3, p6), np.maximum(p3, p6)
    rules, precision = np.full(len(patterns), np.inf), {}
    for k, name in enumerate(patterns):
        for thr in np.arange(0.2, 0.99, 0.02):
            hit = low[:, k] >= thr
            if hit.sum() >= MIN_HITS and y[hit, k].mean() >= TARGET_PRECISION:
                rules[k], precision[name] = thr, round(float(y[hit, k].mean()), 3)
                break
    finite = np.isfinite(rules)
    normal_rule = 0.0
    if finite.any():
        margin = (high[:, finite] / rules[finite]).max(1)
        for thr in np.arange(0.9, 0.05, -0.05):
            hit = margin < thr
            if hit.sum() >= MIN_HITS and (y[hit].sum(1) == 0).mean() >= NORMAL_PRECISION:
                normal_rule = float(thr)
                precision["Normal"] = round(float((y[hit].sum(1) == 0).mean()), 3)
                break
    return rules, normal_rule, precision


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
    rules, normal_rule, expected = fit_rules(net, g, t, patterns)
    durable.event(f"추가 라벨 기준: 예상 정밀도 {expected}")
    finite = np.isfinite(rules)
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
            p3, p6 = scores(net, core, g, t)
            seen += len(picks)
            low, high = np.minimum(p3, p6), np.maximum(p3, p6)
            margin = low / rules                         # 기준이 없는 패턴은 0 이 되어 뽑히지 않는다
            top = margin.argmax(1)
            rows = np.arange(len(top))
            others = high / rules
            others[rows, top] = 0
            sure = (margin[rows, top] >= 1) & (others.max(1) < 1)
            normal = (high[:, finite] / rules[finite]).max(1) < normal_rule if finite.any() else np.zeros(len(top), bool)
            for i in np.where(sure | normal)[0]:
                cls = len(patterns) if normal[i] else top[i]
                if got[cls] >= caps[cls]:
                    continue
                got[cls] += 1
                y = np.zeros(len(patterns), np.uint8)
                if not normal[i]:
                    y[cls] = 1
                kept_raw.append(raw[i].copy())   # 뷰로 두면 샤드 전체가 메모리에 남는다
                kept_core.append(core[i].copy())
                kept_y.append(y)
        if k % 16 == 0:
            training_status.report("pseudo_label", k, SHARDS, f"{seen:,}장 판정, {sum(got):,}장 채택", unit="샤드")
    np.save(f"{DATA_DIR}/pseudo_raw.npy", np.stack(kept_raw))
    np.save(f"{DATA_DIR}/pseudo_core.npy", np.stack(kept_core))
    np.save(f"{DATA_DIR}/pseudo_labels.npy", np.stack(kept_y))
    res = {"judged": seen, "kept": int(sum(got)), "per_class": dict(zip(patterns + ["Normal"], got.tolist())),
           "expected_precision": expected, "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}
    print(res, flush=True)
    durable.write_json("models/pseudo_label.json", res)
    durable.event(f"추가 라벨 {res['kept']:,}장 채택 ({res['judged']:,}장 판정)")


if __name__ == "__main__":
    main()
