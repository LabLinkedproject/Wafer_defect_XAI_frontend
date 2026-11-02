"""
training/prepare_data.py
공개 웨이퍼 맵을 실물 웨이퍼(지름 270 mm, 다이 32 mm) 기준 격자로 가공한다.

실행: ./venv-train/bin/python -m training.prepare_data

자료별로 반영한 내용
- WM-811K 노트북: 맵 크기가 632종이라 크기 정규화가 필수이고, 라벨 없는 맵은 제외하며,
  가장 두드러진 영역 추출이 곧 잡음 제거다.
- MixedWM38: 복합 결함 8종 복수 라벨.
- 유영지 외(2015): 요약값은 패턴 정보를 잃으므로 다이를 세부 구역(quadrat)으로 나눠 밀도를 본다.
  공개 데이터의 다이 하나가 우리 다이의 1/36 구역 하나에 해당한다.

처리 순서
1. 맵마다 웨이퍼 타원을 270 mm 원에 맞춘다. 다이 크기가 달라도 물리 위치가 같은 셀에 놓인다.
2. 연결 크기로 패턴 구조와 잡음을 나눈다. 기준은 다이 수가 아니라 면적이라 맵 크기와 무관하다.
3. 분할한다. WM-811K 는 파일 순서가 로트 순서이므로 연속 구간 단위로 나누고 경계의 웨이퍼를 버려
   같은 로트가 학습과 시험에 함께 들어가지 않게 한다. MixedWM38 은 동일 맵을 묶고 라벨 조합별로 층화한다.
4. MixedWM38 의 Random 866장과 Near-full 149장은 수가 WM-811K 와 똑같아 같은 원본으로 보고,
   로트 분할을 지키기 위해 사용하지 않는다.
"""
from __future__ import annotations
import collections
import hashlib
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adc.knowledge import PATTERNS                     # noqa: E402
import training_status                                 # noqa: E402
import wafer_raster as R                               # noqa: E402

MIXED_SOURCE = "datasets/Wafer_Map_Datasets.npz"
WM811K_SOURCE = "datasets/wm811k_labeled.npz"
OUT_DIR = "datasets/wafer270"
MIXED_LABELS = ("Center", "Donut", "Edge-Loc", "Edge-Ring", "Local", "Near-full", "Scratch", "Random")
WM811K_LABELS = ("Center", "Donut", "Edge-Loc", "Edge-Ring", "Local", "Random", "Scratch", "Near-full")
DUPLICATED_IN_WM811K = ("Random", "Near-full")
MIN_STRUCTURE_FRACTION = 16 / (np.pi * 26 ** 2)   # 지름 52다이 맵의 19 % 잡음을 95 % 거르는 16다이의 면적비
MIN_STRUCTURE_DIES = 3
MIN_DIAMETER_DIES = 20        # 이보다 성긴 맵은 다이 한 칸이 13 mm 를 넘어 패턴 형상을 담지 못한다
SUPER = 4                     # 재표본화 초과 표본
SPLIT = (0.75, 0.10, 0.15)    # 학습 / 검증 / 시험
BLOCK = 2000                  # WM-811K 분할 단위 (파일 순서 기준 웨이퍼 수)
PURGE = 25                    # 구간 경계에서 버리는 폭. 로트 하나가 최대 25장이다
NORMAL_TRAIN = 12000          # 정상 맵은 패턴 맵보다 6배 많아 학습용은 줄인다
NORMAL_EVAL = 6000            # 검증·시험에 각각 쓰는 정상 맵 수
SEED = 7
TRAIN, VAL, TEST = 0, 1, 2
MIXED, WM811K = 0, 1


def structure(fail: np.ndarray, area: float) -> np.ndarray:
    """연결 크기가 작은 불량(배경 잡음)을 지운다. area: 웨이퍼 면적(다이 수)"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fail, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= max(MIN_STRUCTURE_DIES, MIN_STRUCTURE_FRACTION * area)
    keep[0] = False
    return keep[lab].astype(np.uint8)


def to_grid(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """웨이퍼 외곽 box(x0, x1, y0, y1)를 270 mm 원에 맞춘 면적 평균 격자. 반환은 0~255."""
    x0, x1, y0, y1 = box
    k = R.RADIUS_CELLS * SUPER * 2
    sx, sy = k / (x1 - x0), k / (y1 - y0)
    c = R.SIZE * SUPER / 2
    M = np.array([[sx, 0, c - sx * (x0 + x1) / 2], [0, sy, c - sy * (y0 + y1) / 2]], np.float32)
    big = cv2.warpAffine(img, M, (R.SIZE * SUPER,) * 2, flags=cv2.INTER_NEAREST)
    return cv2.resize(big, (R.SIZE, R.SIZE), interpolation=cv2.INTER_AREA)


def split_mixed(hashes, combos) -> np.ndarray:
    """동일 맵은 같은 쪽에, 라벨 조합별 비율은 같게."""
    rng = np.random.default_rng(SEED)
    by_combo = collections.defaultdict(set)
    for h, c in zip(hashes, combos):
        by_combo[c].add(h)
    assign = {}
    for c in sorted(by_combo, key=str):
        hs = sorted(h for h in by_combo[c] if h not in assign)
        hs = [hs[i] for i in rng.permutation(len(hs))]
        a, b = int(len(hs) * SPLIT[0]), int(len(hs) * (SPLIT[0] + SPLIT[1]))
        assign.update({h: TRAIN if i < a else VAL if i < b else TEST for i, h in enumerate(hs)})
    return np.array([assign[h] for h in hashes], np.uint8)


def split_wm811k(ids: np.ndarray) -> np.ndarray:
    """연속 구간 단위 분할. 경계 근처 웨이퍼는 255(미사용)로 표시한다."""
    rng = np.random.default_rng(SEED)
    blocks = ids // BLOCK
    uniq = np.unique(blocks)
    u = rng.random(len(uniq))
    which = np.where(u < SPLIT[0], TRAIN, np.where(u < SPLIT[0] + SPLIT[1], VAL, TEST))
    table = dict(zip(uniq.tolist(), which.tolist()))
    split = np.array([table[b] for b in blocks.tolist()], np.uint8)
    pos = ids % BLOCK
    near = (pos < PURGE) | (pos >= BLOCK - PURGE)
    for side, off in ((pos < PURGE, -1), (pos >= BLOCK - PURGE, 1)):
        other = np.array([table.get(b + off, -1) for b in blocks.tolist()])
        split[side & (other != split) & (other >= 0)] = 255
    return split


def load_mixed():
    src = np.load(MIXED_SOURCE)
    X, Y = src["arr_0"].astype(np.uint8), src["arr_1"].astype(np.uint8)
    del src
    labels = Y[:, [MIXED_LABELS.index(p) for p in PATTERNS]]
    drop = sum(labels[:, list(PATTERNS).index(p)] for p in DUPLICATED_IN_WM811K) > 0
    hashes = [hashlib.md5(m.tobytes()).hexdigest() for m in X]
    split = split_mixed(hashes, [tuple(y) for y in labels])
    use = np.where(~drop)[0]
    return X, labels[use], split[use], use


def write_mixed(X, use, raw, core):
    for n, i in enumerate(use):
        m = X[i]
        v = np.where((m > 0).any(1))[0], np.where((m > 0).any(0))[0]
        box = (v[1][0], v[1][-1] + 1, v[0][0], v[0][-1] + 1)
        fail = (m >= 2).astype(np.uint8)                # 값 3(105장, 평균 2칸)은 불량으로 본다
        raw[n], core[n] = to_grid(fail * 255, box), to_grid(structure(fail, (m > 0).sum()) * 255, box)
        if n % 1000 == 0:
            training_status.report("prepare_data", n, len(use), "MixedWM38 을 270 mm 격자로 변환", unit="장")


def load_wm811k():
    src = np.load(WM811K_SOURCE)
    tables = {k: src[k] for k in ("ids", "labels", "shapes", "ellipse", "offsets", "xy")}
    ids, y, ell = tables["ids"], tables["labels"], tables["ellipse"]
    split = split_wm811k(ids)
    split[2 * ell[:, 2:4].min(1) + 1 < MIN_DIAMETER_DIES] = 255
    rng = np.random.default_rng(SEED)
    for side, cap in ((TRAIN, NORMAL_TRAIN), (VAL, NORMAL_EVAL), (TEST, NORMAL_EVAL)):
        idx = np.where((y == 8) & (split == side))[0]
        if len(idx) > cap:
            split[rng.permutation(idx)[cap:]] = 255
    use = np.where(split != 255)[0]
    order = [list(PATTERNS).index(p) for p in WM811K_LABELS]
    labels = np.zeros((len(use), len(PATTERNS)), np.uint8)
    for n, i in enumerate(use):
        if y[i] < 8:
            labels[n, order[y[i]]] = 1
    return tables, labels, split[use], use


def write_wm811k(t, use, raw, core):
    for n, i in enumerate(use):
        h, w = t["shapes"][i]
        fail = np.zeros((h, w), np.uint8)
        pts = t["xy"][t["offsets"][i]:t["offsets"][i + 1]]
        fail[pts[:, 1], pts[:, 0]] = 1
        cx, cy, a, b = t["ellipse"][i]
        box = (cx - a, cx + a + 1, cy - b, cy + b + 1)
        area = np.pi * (a + 0.5) * (b + 0.5)
        raw[n], core[n] = to_grid(fail * 255, box), to_grid(structure(fail, area) * 255, box)
        if n % 1000 == 0:
            training_status.report("prepare_data", n, len(use), "WM-811K 를 270 mm 격자로 변환", unit="장")


def main():
    X, y0, s0, u0 = load_mixed()
    t, y1, s1, u1 = load_wm811k()
    os.makedirs(OUT_DIR, exist_ok=True)
    n0, n = len(u0), len(u0) + len(u1)
    shape = (n, R.SIZE, R.SIZE)
    raw = np.lib.format.open_memmap(f"{OUT_DIR}/raw.npy", "w+", np.uint8, shape)
    core = np.lib.format.open_memmap(f"{OUT_DIR}/core.npy", "w+", np.uint8, shape)
    write_mixed(X, u0, raw[:n0], core[:n0])
    del X
    write_wm811k(t, u1, raw[n0:], core[n0:])
    raw.flush()
    core.flush()
    labels, split = np.concatenate([y0, y1]), np.concatenate([s0, s1])
    source = np.concatenate([np.full(n0, MIXED, np.uint8), np.full(len(u1), WM811K, np.uint8)])
    wafer = np.concatenate([np.zeros(n0, np.int32) - 1, t["ids"][u1]])
    np.savez(f"{OUT_DIR}/meta.npz", labels=labels, split=split, source=source, wafer=wafer)
    info = {"patterns": list(PATTERNS), "cell_mm": R.CELL_MM, "size": R.SIZE,
            "counts": {f"{'MixedWM38' if s == MIXED else 'WM-811K'}/{['학습', '검증', '시험'][k]}":
                       int(((source == s) & (split == k)).sum()) for s in (MIXED, WM811K) for k in range(3)},
            "labels": dict(zip(PATTERNS, labels[split == TRAIN].sum(0).tolist()))}
    with open(f"{OUT_DIR}/info.json", "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    w = wafer[source == WM811K]
    sp = split[source == WM811K]
    print("학습·시험 중복", len(set(w[sp == TRAIN].tolist()) & set(w[sp == TEST].tolist())))


if __name__ == "__main__":
    main()
