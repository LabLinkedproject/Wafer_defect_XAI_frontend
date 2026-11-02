"""
training/baseline.py
wm-811k-wafermap.ipynb 의 방식을 우리 기준 격자와 같은 로트 분리 시험으로 재현한다.
밀도 13 + 라돈 40 + 기하 6 = 59개 특징, One-vs-One 선형 SVM. 패턴 웨이퍼 8종만 쓴다.

노트북의 시험 정확도 79.04 % 는 무작위 분할 값이고, 여기서는 같은 로트가 시험에 섞이지 않는다.

실행: ./venv-train/bin/python -m training.baseline
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
from scipy import interpolate
from skimage.measure import label, regionprops
from skimage.transform import radon
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train import TEST, TRAIN, WM811K, load_images, load_meta  # noqa: E402
import training_status                                 # noqa: E402

OUT = "models/baseline_svm.json"


def density(x):
    r, c = x.shape[0] // 5, x.shape[1] // 5
    regs = [x[:r], x[:, 4 * c:], x[4 * r:], x[:, :c]]
    regs += [x[(i + 1) * r:(i + 2) * r, (j + 1) * c:(j + 2) * c] for i in range(3) for j in range(3)]
    return [100 * a.mean() for a in regs]


def radon_stats(x):
    sino = radon(x, theta=np.linspace(0, 180, 60, endpoint=False), circle=False)
    out = []
    for row in (sino.mean(1), sino.std(1)):
        f = interpolate.interp1d(np.arange(row.size), row, kind="cubic")
        out.extend(f(np.linspace(0, row.size - 1, 20)) / 100)
    return out


def geometry(x):
    lab = label(x, connectivity=1)
    if lab.max() == 0:
        return [0.0] * 6
    p = max(regionprops(lab), key=lambda r: r.area)
    na, np_ = x.size, np.hypot(*x.shape)
    return [p.area / na, p.perimeter / np_, p.axis_major_length / np_, p.axis_minor_length / np_, p.eccentricity, p.solidity]


def features(raw):
    out = []
    for i, a in enumerate(raw.numpy()):
        x = (a >= 128).astype(np.uint8)
        out.append(density(x) + radon_stats(x.astype(float)) + geometry(x))
        if i % 500 == 0:
            training_status.report("baseline", i, len(raw), "59개 특징 추출", unit="장")
    return np.array(out, np.float32)


def main():
    labels, split, source, _, patterns = load_meta()
    pattern = labels.sum(1) == 1
    tr = np.where((split == TRAIN) & (source == WM811K) & pattern)[0]
    te = np.where((split == TEST) & (source == WM811K) & pattern)[0]
    x_tr, x_te = features(load_images(tr)[0]), features(load_images(te)[0])
    y_tr, y_te = labels[tr].argmax(1), labels[te].argmax(1)
    clf = OneVsOneClassifier(LinearSVC(random_state=42, max_iter=3000)).fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    res = {"method": "노트북 재현: 59개 특징 + One-vs-One 선형 SVM", "train": len(tr), "test": len(te),
           "test_accuracy": round(float((pred == y_te).mean()), 4),
           "per_class_recall": {patterns[k]: round(float((pred[y_te == k] == k).mean()), 4) for k in np.unique(y_te)}}
    print(res, flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
