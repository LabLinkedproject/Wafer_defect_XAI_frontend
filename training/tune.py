"""
training/tune.py
실사용 입력에 맞춰 운용 판정선과 모델을 정한다. 학습은 하지 않는다.

wafer_detect 는 다이를 2×2 구역으로 나눈 입력만 쓰고, 현장에서는 정상 웨이퍼를 불량으로 잘못 알리는 비용이 크다.
그래서 WM-811K 검증 분할의 2×2 입력에서 정상 정답률 목표를 지키는 판정선을 찾고, 그 조건의 Macro-F1 로 모델을 고른다.
시험 분할은 고른 모델의 값을 기록할 때만 쓴다.

실행: taskset -c 4-7 ./venv-train/bin/python -m training.tune
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import durable                           # noqa: E402
from training.evaluate import BOOT_BLOCK, fit_thresholds, load_net, macro, report, sigmoid  # noqa: E402
from training.model import Geometry, quadrat_map       # noqa: E402
from training.train import TEST, VAL, WM811K, load_images, load_meta, logits_of  # noqa: E402
import training_status                                 # noqa: E402

OUT = "models/operating.json"
CANDIDATES = (("a", "models/wafernet.pt"), ("b", "models/wafernet_b.pt"))
PER_DIE = 2
NORMAL_TARGET = 0.95
STEP = 0.02
TOP = 0.98


def normal_rate(prob, y, thr) -> float:
    normal = y.sum(1) == 0
    return float(((prob[normal] >= thr).sum(1) == 0).mean())


def hold_normal(prob, y, thr, target=NORMAL_TARGET) -> np.ndarray:
    """정상 정답률이 목표에 닿을 때까지, F1 손실 대비 정상 개선이 가장 큰 패턴의 판정선부터 올린다."""
    thr = thr.copy()
    while normal_rate(prob, y, thr) < target:
        base_f1, base_normal = macro(prob >= thr, y), normal_rate(prob, y, thr)
        best, best_gain = None, -np.inf
        for k in np.where(thr + STEP <= TOP + 1e-9)[0]:
            t = thr.copy()
            t[k] += STEP
            gained = normal_rate(prob, y, t) - base_normal
            if gained > 0 and (macro(prob >= t, y) - base_f1) / gained > best_gain:
                best, best_gain = k, (macro(prob >= t, y) - base_f1) / gained
        if best is None:                                 # 한 칸으로 변화가 없으면 정상 맵에서 가장 많이 울리는 패턴을 올린다
            alarms = ((prob >= thr) & (y.sum(1, keepdims=True) == 0)).sum(0)
            left = [k for k in np.argsort(-alarms) if thr[k] + STEP <= TOP + 1e-9]
            if not left:
                break
            best = left[0]
        thr[best] += STEP
    return thr


def main():
    labels, split, source, wafer, patterns = load_meta()
    g = Geometry()
    make = lambda r, c: quadrat_map(c, g, PER_DIE)
    va = np.where((split == VAL) & (source == WM811K))[0]
    raw, core = load_images(va)
    y = labels[va]
    rows = {}
    for step, (tag, path) in enumerate(CANDIDATES):
        training_status.report("tune", step, 3, f"모델 {tag} 운용 판정선 찾기", unit="단계")
        net, _ = load_net(path, len(patterns))
        with open(f"models/calibration_{tag}.json") as f:
            temperature = json.load(f)["temperature"]
        prob = sigmoid(logits_of(net, make, raw, core) / temperature)
        thr = hold_normal(prob, y, fit_thresholds(prob, y))
        rows[tag] = {"net": net, "temperature": temperature, "thresholds": thr,
                     "val_macro_f1": macro(prob >= thr, y), "val_normal": normal_rate(prob, y, thr)}
        print(tag, rows[tag]["val_macro_f1"], rows[tag]["val_normal"], dict(zip(patterns, thr.round(2).tolist())), flush=True)
    pick = max(rows, key=lambda k: rows[k]["val_macro_f1"])
    chosen = rows[pick]

    training_status.report("tune", 2, 3, f"모델 {pick} 시험 분할 기록", unit="단계")
    te = np.where((split == TEST) & (source == WM811K))[0]
    raw, core = load_images(te)
    prob = sigmoid(logits_of(chosen["net"], make, raw, core) / chosen["temperature"])
    test = report(prob, labels[te], chosen["thresholds"], patterns, wafer[te] // BOOT_BLOCK)
    print("선택", pick, {k: v for k, v in test.items() if k != "per_class"}, flush=True)
    durable.write_json(OUT, {"variant": pick, "per_die": PER_DIE, "normal_target": NORMAL_TARGET, "temperature": chosen["temperature"],
                             "thresholds": dict(zip(patterns, chosen["thresholds"].round(2).tolist())),
                             "val": {k: {"macro_f1": round(v["val_macro_f1"], 4), "normal": round(v["val_normal"], 4)} for k, v in rows.items()},
                             "test": test})


if __name__ == "__main__":
    main()
