"""
adc/cnn.py
학습 모델 추론. torch 없이 OpenCV 로 ONNX 를 돌린다. 모델 파일이 없으면 available() 이 False 다.

확률은 패턴 점수이지 실물 적중률이 아니다. 공장 데이터에서 온도 보정의 개선이 없었으므로 신뢰도처럼 표시하지 않는다.
"""
from __future__ import annotations
import json
import math
import os
import threading

import cv2
import numpy as np

import wafer_raster as R

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.join(BASE, "models", "wafernet.onnx")
META = os.path.join(BASE, "models", "wafernet.json")
_lock = threading.Lock()
_net = None
_meta = None


def available() -> bool:
    return os.path.exists(ONNX) and os.path.exists(META)


def _load():
    global _net, _meta
    if _net is None:
        _net = cv2.dnn.readNetFromONNX(ONNX)
        with open(META, encoding="utf8") as f:
            _meta = json.load(f)
    return _net, _meta


def scores(x: np.ndarray) -> dict[str, float]:
    """입력 3채널 → {패턴: 점수}"""
    with _lock:
        net, meta = _load()
        net.setInput(x[None].astype(np.float32))
        logit = net.forward()[0]
    p = 1 / (1 + np.exp(-logit / meta["temperature"]))
    return {name: round(float(v), 4) for name, v in zip(meta["patterns"], p)}


def labels(score: dict[str, float]) -> list[str]:
    """패턴별 판정선을 넘는 패턴, 점수 순"""
    thr = _load()[1]["thresholds"]
    return sorted((k for k, v in score.items() if v >= thr[k]), key=score.get, reverse=True)


def from_detection(dies: list[dict], quadrats: dict, per_die: int = 2) -> dict:
    """wafer_detect 의 구역 판정을 패턴 판정으로 바꾼다."""
    s = scores(R.from_quadrats(dies, quadrats, per_die))
    return {"scores": s, "labels": labels(s)}


class Model:
    """학습 모델 분류기. 입력은 검출이 만든 다이 구역 판정이다.
    다이를 가리면 그 다이의 구역을 모두 지우고 다시 추론한다. 점수 옆에는 같은 입력의 규칙 기반 해석을 붙인다."""
    kind = "cnn"

    def __init__(self, dies: list[dict], quadrats: dict, rule, per_die: int = 2):
        self.dies, self.quadrats, self.rule, self.per_die = dies, quadrats, rule, per_die
        self.ids = sorted({k[0] for k in quadrats})
        self.thr = _load()[1]["thresholds"]

    def threshold(self, cls: str) -> float:
        return float(self.thr[cls])

    def scores(self, drop=()) -> dict:
        q = {k: v for k, v in self.quadrats.items() if k[0] not in drop}
        s = scores(R.from_quadrats(self.dies, q, self.per_die))
        rule = self.rule.scores() if not drop else {}
        out = {}
        for c, v in s.items():
            p = min(max(v, 1e-6), 1 - 1e-6)
            out[c] = {"score": v, "logit": round(math.log(p / (1 - p)), 3), "channel": "cnn", "bias": None,
                      "contributions": rule.get(c, {}).get("contributions", []), "rule": rule.get(c)}
        return out
