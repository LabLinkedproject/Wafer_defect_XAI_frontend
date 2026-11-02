"""
training/compare.py
노트북 SVM 과 학습 모델을 같은 시험 조건으로 비교하고, 정상 맵을 뺀 성능을 잰다.
baseline.py 의 87 % 는 웨이퍼 전체를 쓴 값이라 44다이 입력과 바로 비교할 수 없다.

실행: taskset -c 4-7 ./venv-train/bin/python -m training.compare
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import torch
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import durable                           # noqa: E402
from training.baseline import density, geometry, radon_stats  # noqa: E402
from training.evaluate import load_net                 # noqa: E402
from training.model import Geometry, quadrat_map       # noqa: E402
from training.train import TEST, TRAIN, WM811K, load_images, load_meta, logits_of, to_float  # noqa: E402

OUT = "models/compare.json"
QUADRATS = {"44다이 판정": 1, "44다이 2×2 구역": 2, "44다이 3×3 구역": 3}


def inputs(core, g, q):
    return torch.cat([quadrat_map(to_float(core[i:i + 256]), g, q)[:, 0] for i in range(0, len(core), 256)]).numpy()


def features(states):
    return np.array([density(x) + radon_stats(x.astype(float)) + geometry(x) for x in (states > 0).astype(np.uint8)], np.float32)


def per_class(pred, y, patterns):
    return {patterns[k]: {"recall": round(float((pred[y == k] == k).mean()), 4), "n": int((y == k).sum())} for k in np.unique(y)}


def main():
    labels, split, source, _, patterns = load_meta()
    g = Geometry()
    with open("models/wafernet.json") as f:
        variant = json.load(f)["variant"]
    net, _ = load_net("models/wafernet_b.pt" if variant == "b" else "models/wafernet.pt", len(patterns))
    single = labels.sum(1) == 1
    tr = np.where((split == TRAIN) & (source == WM811K) & single)[0]
    te = np.where((split == TEST) & (source == WM811K) & single)[0]
    y_tr, y_te = labels[tr].argmax(1), labels[te].argmax(1)
    raw_te, core_te = load_images(te)
    _, core_tr = load_images(tr)
    res = {"test_maps": len(te), "conditions": {}}
    for name, q in QUADRATS.items():
        pred = logits_of(net, lambda r, c: quadrat_map(c, g, q), raw_te, core_te).argmax(1)
        cnn = float((pred == y_te).mean())
        x_tr, x_te = features(inputs(core_tr, g, q)), features(inputs(core_te, g, q))
        svm = OneVsOneClassifier(LinearSVC(random_state=42, max_iter=3000)).fit(x_tr, y_tr).predict(x_te)
        res["conditions"][name] = {"cnn_accuracy": round(cnn, 4), "svm_accuracy": round(float((svm == y_te).mean()), 4),
                                   "cnn": per_class(pred, y_te, patterns), "svm": per_class(svm, y_te, patterns)}
        print(name, cnn, res["conditions"][name]["svm_accuracy"], flush=True)
    k = patterns.index("Scratch")
    s = y_te == k
    res["scratch"] = {"n": int(s.sum()), "raw_fail_cells": float((raw_te[s] > 128).float().sum((1, 2)).mean()),
                      "core_fail_cells": float((core_te[s] > 128).float().sum((1, 2)).mean()),
                      "core_empty_share": float(((core_te[s] > 128).sum((1, 2)) == 0).float().mean())}
    print(res["scratch"], flush=True)
    durable.write_json(OUT, res)


if __name__ == "__main__":
    main()
