"""
adc/scenarios.py
44다이 모델 위에 결함을 배치하는 시연 시나리오. 난수 시드가 고정되어 있어 매번 같은 결과가 나온다.
실제 검출 결과가 아니라 시뮬레이션 입력이다.
"""
from __future__ import annotations
import math
import random

SCENARIOS = {
    "normal":      {"name": "정상",               "note": "결함이 거의 없는 기준 웨이퍼"},
    "center":      {"name": "Center",             "note": "샤워헤드 중앙 막힘 가정"},
    "donut":       {"name": "Donut",              "note": "CVD/PVD 균일도 불량 가정"},
    "edge_ring":   {"name": "Edge-Ring",          "note": "EBR 불량 가정"},
    "edge_loc":    {"name": "Edge-Loc",           "note": "에지 그립 접촉 가정"},
    "local":       {"name": "Local",              "note": "척 접촉·세정 불량 가정"},
    "random":      {"name": "Random",             "note": "챔버 파티클 오염 가정"},
    "scratch":     {"name": "Scratch",            "note": "CMP·이송 접촉 가정"},
    "near_full":   {"name": "Near-full",          "note": "공정 전반 이탈 가정"},
    "donut_scratch": {"name": "Donut + Scratch",  "note": "복합 결함: 증착 불균일 + 이송 긁힘"},
    "ring_random":   {"name": "Edge-Ring + Random", "note": "복합 결함: EBR 불량 + 파티클 오염"},
}


def _put(out, die, dtype, count):
    out.setdefault(die["id"], {"particle": 0, "scratch": 0, "stain": 0})
    out[die["id"]][dtype] += count


def _line(dies, out, count=2, slope=0.55, offset=-10.0):
    for d in dies:
        dist = abs(d["y_mm"] - (slope * d["x_mm"] + offset)) / math.hypot(1, slope)
        if dist <= 10:
            _put(out, d, "scratch", count)


def generate(name: str, dies: list[dict]) -> dict[int, dict]:
    rng = random.Random(42)
    out: dict[int, dict] = {}
    if name == "normal":
        pass
    elif name == "center":
        for d in dies:
            if d["zone"] == "center":
                _put(out, d, "stain", 3)
            elif d["rho"] < 0.48:
                _put(out, d, "stain", 1)
    elif name in ("donut", "donut_scratch"):
        for d in dies:
            if d["zone"] == "middle":
                _put(out, d, "stain", 3 if 0.5 <= d["rho"] <= 0.73 else 1)
        if name == "donut_scratch":
            _line(dies, out)
    elif name in ("edge_ring", "ring_random"):
        for d in dies:
            if d["zone"] == "edge":
                _put(out, d, "stain", 3 if d["rho"] > 0.9 else 1)
        if name == "ring_random":
            inner = [d for d in dies if d["zone"] != "edge"]
            rng.shuffle(inner)
            picked = []
            for d in inner:          # 서로 붙지 않게 골라 산발 분포를 만든다
                if all(max(abs(d["row"] - q["row"]), abs(d["col"] - q["col"])) >= 2 for q in picked):
                    picked.append(d)
                if len(picked) == 6:
                    break
            for d in picked:
                _put(out, d, "particle", rng.choice((1, 2)))
    elif name == "edge_loc":
        for d in dies:
            if d["zone"] == "edge" and d["x_mm"] > 20 and d["y_mm"] > -40:
                _put(out, d, "particle", 3 if d["rho"] > 0.9 else 2)
    elif name == "local":
        for d in dies:
            if math.hypot(d["x_mm"] + 36, d["y_mm"] - 12) < 34:
                _put(out, d, "stain", 3 if math.hypot(d["x_mm"] + 36, d["y_mm"] - 12) < 20 else 1)
    elif name == "random":
        for d in rng.sample(dies, 11):
            _put(out, d, "particle", rng.choice((1, 1, 2, 3)))
    elif name == "scratch":
        _line(dies, out)
    elif name == "near_full":
        for d in dies:
            if rng.random() < 0.86:
                _put(out, d, rng.choice(("stain", "particle")), rng.choice((1, 3)))
    else:
        raise KeyError(name)
    return out
