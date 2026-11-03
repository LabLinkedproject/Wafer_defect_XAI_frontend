"""
training/report_metrics.py
발표·보고서용 성능 지표를 계산한다. 학습은 하지 않는다.

운용 모델(models/operating.json)과 운용 판정선으로 WM-811K 로트 분리 시험 세트를 판정하고
- 웨이퍼 단위 혼동 행렬 (불량 패턴 있음 / 정상)
- 패턴별 TP · FP · FN · TN, 정밀도, 재현율, F1, 특이도, G-mean
- 로트 구간 부트스트랩 95 % 신뢰구간
- 확률 신뢰도 곡선과 ECE
- '모두 정상'이라고 하는 기준 모델과의 비교
를 models/report_metrics.json 에 쓴다. 입력 방식 두 가지를 함께 잰다.
  quadrat2 : 다이를 2×2 구역으로 나눈 입력 (판정선을 정한 조건)
  die_any  : 결함이 하나라도 있으면 다이 전체 불량 (실측 경로 adc/cnn.DIE_LEVEL)
모두 공개 데이터 값이며 실물 웨이퍼 정확도가 아니다.

실행: taskset -c 4-7 ./venv-train/bin/python -m training.report_metrics
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import durable                           # noqa: E402
from training.evaluate import BOOT_BLOCK, expected_calibration_error, load_net, sigmoid  # noqa: E402
from training.model import Geometry, quadrat_map       # noqa: E402
from training.train import TEST, WM811K, load_images, load_meta, logits_of  # noqa: E402

OUT = "models/report_metrics.json"
BOOT = 400


def counts(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    return tp, fp, fn, tn


def rates(tp, fp, fn, tn):
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn); s = tn / max(1, tn + fp)
    return {"precision": round(p, 4), "recall": round(r, 4), "specificity": round(s, 4),
            "f1": round(2 * tp / max(1, 2 * tp + fp + fn), 4), "gmean": round(float(np.sqrt(r * s)), 4),
            "accuracy": round((tp + tn) / max(1, tp + fp + fn + tn), 4)}


def reliability(prob, y, bins=10):
    p, t = prob.ravel(), (y > 0).ravel().astype(float)
    use = p >= 0.05                                    # evaluate.expected_calibration_error 와 같은 기준
    p, t = p[use], t[use]
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
        if m.any():
            rows.append({"lo": round(float(a), 2), "hi": round(float(b), 2), "n": int(m.sum()),
                         "predicted": round(float(p[m].mean()), 4), "observed": round(float(t[m].mean()), 4)})
    return rows


def evaluate_mode(prob, y, thr, patterns, groups):
    pred = prob >= thr
    truth = y > 0
    wafer_pred, wafer_true = pred.any(1), truth.any(1)
    out = {"wafer": {"tp_fp_fn_tn": counts(wafer_pred, wafer_true)}}
    out["wafer"].update(rates(*out["wafer"]["tp_fp_fn_tn"]))
    per = {}
    for k, name in enumerate(patterns):
        c = counts(pred[:, k], truth[:, k])
        per[name] = {"tp_fp_fn_tn": c, "support": int(truth[:, k].sum()), **rates(*c)}
    out["per_class"] = per
    out["macro_f1"] = round(float(np.mean([v["f1"] for v in per.values() if v["support"]])), 4)
    out["exact_match"] = round(float((pred == truth).all(1).mean()), 4)
    # 로트 구간 부트스트랩
    rng = np.random.default_rng(0)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    boots = {"macro_f1": [], "wafer_recall": [], "wafer_specificity": [], "exact_match": []}
    for _ in range(BOOT):
        ids = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        pp, tt = pred[ids], truth[ids]
        f1s = [2 * (pp[:, k] & tt[:, k]).sum() / max(1, 2 * (pp[:, k] & tt[:, k]).sum() + (pp[:, k] & ~tt[:, k]).sum()
                                                       + (~pp[:, k] & tt[:, k]).sum()) for k in range(len(patterns)) if tt[:, k].any()]
        boots["macro_f1"].append(float(np.mean(f1s)))
        wtp, wfp, wfn, wtn = counts(pp.any(1), tt.any(1))
        boots["wafer_recall"].append(wtp / max(1, wtp + wfn))
        boots["wafer_specificity"].append(wtn / max(1, wtn + wfp))
        boots["exact_match"].append(float((pp == tt).all(1).mean()))
    out["ci95"] = {k: [round(float(v), 4) for v in np.percentile(b, [2.5, 97.5])] for k, b in boots.items()}
    out["calibration"] = {"ece": round(expected_calibration_error(prob, y), 4), "bins": reliability(prob, y)}
    return out


def main():
    labels, split, source, wafer, patterns = load_meta()
    with open("models/operating.json") as f:
        op = json.load(f)
    thr = np.array([op["thresholds"][p] for p in patterns])
    net, _ = load_net("models/wafernet_b.pt" if op["variant"] == "b" else "models/wafernet.pt", len(patterns))
    g = Geometry()
    te = np.where((split == TEST) & (source == WM811K))[0]
    raw, core = load_images(te)
    y = labels[te]
    groups = wafer[te] // BOOT_BLOCK
    modes = {
        "quadrat2": lambda r, c: quadrat_map(c, g, 2),
        # 다이 안에 불량 셀이 하나라도 있으면 다이 전체 heavy (실측 경로와 같은 의미)
        "die_any": lambda r, c: quadrat_map(c, g, 1, light=1e-6, heavy=1e-6),
    }
    res = {"dataset": "WM-811K 로트 분리 시험 (공개 공장 데이터, 실물 웨이퍼 아님)", "size": int(len(te)),
           "model": f"WaferNet {op['variant'].upper()}", "thresholds": op["thresholds"], "modes": {}}
    for name, make in modes.items():
        prob = sigmoid(logits_of(net, make, raw, core) / op["temperature"])
        res["modes"][name] = evaluate_mode(prob, y, thr, patterns, groups)
        w = res["modes"][name]["wafer"]
        print(name, "웨이퍼", w["tp_fp_fn_tn"], "재현율", w["recall"], "특이도", w["specificity"],
              "Macro-F1", res["modes"][name]["macro_f1"], flush=True)
    n_def = int((y > 0).any(1).sum())
    res["all_normal_baseline"] = {"tp_fp_fn_tn": [0, 0, n_def, int(len(te) - n_def)],
                                  "accuracy": round((len(te) - n_def) / len(te), 4), "recall": 0.0}
    durable.write_json(OUT, res)


if __name__ == "__main__":
    main()
