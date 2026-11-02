"""
training/export.py
학습된 모델을 ONNX 로 내보내고, 웹 서버가 쓰는 OpenCV 추론과 결과가 같은지 확인한다.
웹 서버와 이후 EXE 는 torch 없이 이 파일 하나로 돌아간다.

실행: ./venv-train/bin/python -m training.export
"""
from __future__ import annotations
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.model import WaferNet                    # noqa: E402
import wafer_raster as R                               # noqa: E402

CKPT = "models/wafernet.pt"
CKPT_B = "models/wafernet_b.pt"
ONNX = "models/wafernet.onnx"
META = "models/wafernet.json"


def choose():
    """추가 라벨 모델은 검증 점수가 더 높을 때만 쓴다. 시험 분할로 고르지 않는다."""
    a = torch.load(CKPT)
    b = torch.load(CKPT_B) if os.path.exists(CKPT_B) else None
    return (b, "b") if b and b["val_score"] > a["val_score"] else (a, "a")


def main():
    ck, tag = choose()
    print("선택 모델", tag, "검증 점수", ck["val_score"])
    shutil.copy(f"models/metrics_{tag}.json", "models/wafernet_metrics.json")
    net = WaferNet(len(ck["patterns"]))
    net.load_state_dict(ck["state"])
    net.eval()
    x = torch.rand(1, 3, R.SIZE, R.SIZE)
    torch.onnx.export(net, (x,), ONNX, opset_version=13, input_names=["input"], output_names=["logit"], dynamo=False)
    with open(f"models/calibration_{tag}.json") as f:
        cal = json.load(f)
    with open(META, "w") as f:
        json.dump({"patterns": ck["patterns"], "epoch": ck["epoch"], "variant": tag, "size": R.SIZE, "cell_mm": R.CELL_MM,
                   "cells_per_die": R.CELLS_PER_DIE, **cal}, f, ensure_ascii=False, indent=2)
    dnn = cv2.dnn.readNetFromONNX(ONNX)
    worst = 0.0
    for _ in range(20):
        a = torch.rand(1, 3, R.SIZE, R.SIZE)
        dnn.setInput(a.numpy())
        worst = max(worst, float(np.abs(dnn.forward() - net(a).detach().numpy()).max()))
    t0 = time.time()
    for _ in range(200):
        dnn.setInput(a.numpy())
        dnn.forward()
    print(f"ONNX {os.path.getsize(ONNX) / 1024:.0f} kB, torch 대비 최대 차이 {worst:.2e}, OpenCV 추론 {(time.time() - t0) * 5:.2f} ms/장")
    assert worst < 1e-3, "OpenCV 추론 결과가 학습 모델과 다릅니다"


if __name__ == "__main__":
    main()
