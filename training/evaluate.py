"""
training/evaluate.py
학습에 쓰지 않은 시험 분할로 성능을 잰다. 판정 기준과 확률 보정은 검증 분할에서만 정한다.

실행: ./venv-train/bin/python -m training.evaluate

신뢰성 항목
- 로트 분리 시험: WM-811K 의 학습에 없던 로트 구간 (실제 공장 데이터)
- 교차 데이터셋: MixedWM38 만으로 학습한 v1 모델을 WM-811K 로 시험
- 구간 부트스트랩 95 % 신뢰구간: 로트 구간을 단위로 다시 뽑는다
- 클래스별 G-mean: 유영지 외(2015)가 불균형 데이터에 쓴 지표 √(재현율 × 특이도)
- 온도 보정과 ECE: 화면의 확률이 실제 적중률과 맞는지 잰다
- 기존 규칙 분류기와 같은 입력으로 비교
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adc import scenarios, scoring                     # noqa: E402
from training.model import Geometry, WaferNet, full_map, quadrat_map   # noqa: E402
from training.train import (CKPT, MIXED, TEST, VAL, WM811K, load_images, load_meta, logits_of, to_float)  # noqa: E402
import training_status                                 # noqa: E402
import wafer_model                                     # noqa: E402
import wafer_raster as R                               # noqa: E402

V1_CKPT = "models/v1/wafernet.pt"
OUT = "models/wafernet_metrics.json"
META = "models/calibration.json"
BOOT = 400
BOOT_BLOCK = 2000
DIES = scoring.annotate(wafer_model.die_layout())


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def conditions(g):
    return {"전면 맵, 실제 잡음": lambda r, c: full_map(r, g), "전면 맵, 잡음 제거": lambda r, c: full_map(c, g),
            "44다이 6×6 구역": lambda r, c: quadrat_map(c, g, 6), "44다이 3×3 구역": lambda r, c: quadrat_map(c, g, 3),
            "44다이 2×2 구역": lambda r, c: quadrat_map(c, g, 2), "44다이 판정": lambda r, c: quadrat_map(c, g, 1)}


def bce(logit, y, t):
    p = np.clip(sigmoid(logit / t), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_temperature(logit, y):
    grid = np.arange(0.5, 4.01, 0.05)
    return float(grid[int(np.argmin([bce(logit, y, t) for t in grid]))])


def fit_thresholds(prob, y):
    """클래스별로 검증 F1 이 가장 높은 판정선"""
    grid = np.arange(0.1, 0.91, 0.02)
    out = []
    for k in range(y.shape[1]):
        f1 = []
        for t in grid:
            p = prob[:, k] >= t
            tp = (p & (y[:, k] > 0)).sum()
            f1.append(2 * tp / max(1, p.sum() + (y[:, k] > 0).sum()))
        out.append(round(float(grid[int(np.argmax(f1))]), 2))
    return np.array(out)


def expected_calibration_error(prob, y, bins=10):
    p, t = prob.ravel(), (y > 0).ravel().astype(float)
    # 대부분 칸이 확실한 음성이라 그대로 평균하면 값이 작게 나온다. 판정에 영향 있는 칸만 본다
    use = p >= 0.05
    p, t = p[use], t[use]
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.any():
            total += m.mean() * abs(p[m].mean() - t[m].mean())
    return float(total)


def score(pred, y):
    truth = y > 0
    tp, fp = (pred & truth).sum(0), (pred & ~truth).sum(0)
    fn, tn = (~pred & truth).sum(0), (~pred & ~truth).sum(0)
    has = truth.sum(0) > 0
    return tp, fp, fn, tn, has


def macro(pred, y):
    tp, fp, fn, _, has = score(pred, y)
    return float((2 * tp / np.maximum(1, 2 * tp + fp + fn))[has].mean())


def report(prob, y, thresholds, patterns, groups):
    pred = prob >= thresholds
    tp, fp, fn, tn, has = score(pred, y)
    per = {}
    for k, name in enumerate(patterns):
        if has[k]:
            rec, spec = tp[k] / max(1, tp[k] + fn[k]), tn[k] / max(1, tn[k] + fp[k])
            per[name] = {"precision": round(float(tp[k] / max(1, tp[k] + fp[k])), 4), "recall": round(float(rec), 4),
                         "f1": round(float(2 * tp[k] / max(1, 2 * tp[k] + fp[k] + fn[k])), 4),
                         "gmean": round(float(np.sqrt(rec * spec)), 4), "support": int(tp[k] + fn[k])}
    normal = y.sum(1) == 0
    rng = np.random.default_rng(0)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    boot_f1, boot_ex = [], []
    for _ in range(BOOT):
        ids = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        boot_f1.append(macro(pred[ids], y[ids]))
        boot_ex.append(float((pred[ids] == (y[ids] > 0)).all(1).mean()))
    return {"exact_match": round(float((pred == (y > 0)).all(1).mean()), 4),
            "exact_ci": [round(float(v), 4) for v in np.percentile(boot_ex, [2.5, 97.5])],
            "macro_f1": round(macro(pred, y), 4),
            "macro_ci": [round(float(v), 4) for v in np.percentile(boot_f1, [2.5, 97.5])],
            "normal_correct": round(float((pred[normal].sum(1) == 0).mean()), 4) if normal.any() else None,
            "ece": round(expected_calibration_error(prob, y), 4), "per_class": per}


def to_defects(state: np.ndarray) -> dict:
    """다이 판정 → 규칙 분류기 입력 (light 1개, heavy 3개)"""
    out = {}
    for d in DIES:
        r, c = R.die_slot(d)
        v = state[r * R.CELLS_PER_DIE, c * R.CELLS_PER_DIE]
        if v > 0:
            out[d["id"]] = {"stain": 3 if v > 0.9 else 1}
    return out


def load_net(path, n):
    net = WaferNet(n)
    ck = torch.load(path)
    net.load_state_dict(ck["state"])
    return net, ck


def main():
    labels, split, source, wafer, patterns = load_meta()
    g = Geometry()
    net, ck = load_net(CKPT, len(patterns))
    training_status.report("evaluate", 0, 5, "검증 분할로 확률 보정", unit="단계")

    va = np.where(split == VAL)[0]
    raw, core = load_images(va)
    val_logit = {k: logits_of(net, make, raw, core) for k, make in conditions(g).items()}
    joined = np.concatenate(list(val_logit.values()))
    y_val = np.concatenate([labels[va]] * len(val_logit))
    temperature = fit_temperature(joined, y_val)
    thresholds = fit_thresholds(sigmoid(joined / temperature), y_val)
    print("온도", temperature, "판정선", dict(zip(patterns, thresholds.tolist())), flush=True)
    with open(META, "w") as f:
        json.dump({"temperature": temperature, "thresholds": dict(zip(patterns, thresholds.tolist()))}, f, ensure_ascii=False, indent=2)

    result = {"epoch": ck["epoch"], "temperature": temperature, "tests": {}}
    names = {WM811K: "WM-811K 로트 분리 시험 (실제 공장 데이터)", MIXED: "MixedWM38 시험"}
    keep = {}
    for step, src in enumerate((WM811K, MIXED), 1):
        te = np.where((split == TEST) & (source == src))[0]
        raw, core = load_images(te)
        y = labels[te]
        groups = wafer[te] // BOOT_BLOCK if src == WM811K else np.arange(len(te)) // 50
        rows = {}
        for name, make in conditions(g).items():
            training_status.report("evaluate", step, 5, f"{names[src]}: {name}", unit="단계")
            logit = logits_of(net, make, raw, core)
            rows[name] = report(sigmoid(logit / temperature), y, thresholds, patterns, groups)
            rows[name]["ece_before"] = round(expected_calibration_error(sigmoid(logit), y), 4)
            print(names[src], name, {k: v for k, v in rows[name].items() if k != "per_class"}, flush=True)
        result["tests"][names[src]] = {"size": len(te), "conditions": rows}
        keep[src] = (raw, core, y, groups)

    raw, core, y, groups = keep[WM811K]
    training_status.report("evaluate", 3, 5, "기존 규칙 분류기 비교", unit="단계")
    rule = np.zeros_like(y)
    for i in range(0, len(core), 256):
        states = quadrat_map(to_float(core[i:i + 256]), g, 1)[:, 0].numpy()
        for j, st in enumerate(states):
            s = scoring.predict(DIES, to_defects(st))
            rule[i + j] = [s[c]["score"] for c in patterns]
    result["rule_baseline_44die"] = report(rule, y, 0.5, patterns, groups)
    print("규칙 분류기", {k: v for k, v in result["rule_baseline_44die"].items() if k != "per_class"}, flush=True)

    if os.path.exists(V1_CKPT):
        training_status.report("evaluate", 4, 5, "교차 데이터셋 시험", unit="단계")
        v1, _ = load_net(V1_CKPT, len(patterns))
        common = [k for k, c in enumerate(patterns) if c not in ("Random", "Near-full")]
        use = y[:, [patterns.index("Random"), patterns.index("Near-full")]].sum(1) == 0
        prob = sigmoid(logits_of(v1, lambda r, c: full_map(r, g), raw, core))
        result["cross_dataset"] = {"train": "MixedWM38 만 학습 (v1)", "test": names[WM811K], "note": "두 데이터의 원본이 겹치는 Random, Near-full 제외",
                                   **report(prob[use][:, common], y[use][:, common], 0.5, [patterns[k] for k in common], groups[use])}
        print("교차", {k: v for k, v in result["cross_dataset"].items() if k != "per_class"}, flush=True)

    rows = {}
    for name in scenarios.SCENARIOS:
        defects = scenarios.generate(name, DIES)
        vals = {i: scoring.die_weight(c) for i, c in defects.items()}
        with torch.no_grad():
            p = sigmoid(net.eval()(torch.from_numpy(R.from_dies(DIES, vals))[None]).numpy()[0] / temperature)
        got = [patterns[k] for k in np.argsort(-p) if p[k] >= thresholds[k]]
        rows[name] = {"학습 모델": got, "규칙": scoring.labels_from(scoring.predict(DIES, defects))}
        print(f"  {name:14s} {rows[name]}", flush=True)
    result["scenarios"] = rows
    with open(OUT, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)


if __name__ == "__main__":
    main()
