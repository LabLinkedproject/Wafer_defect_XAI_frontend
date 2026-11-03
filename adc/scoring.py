"""
adc/scoring.py
다이 맵 → 공간 특징 → 8종 복수 라벨 점수 + XAI(기여도 분해, 다이 가림 시험, 반사실 설명).

분류기는 scores(drop) 와 threshold(패턴) 를 가진 객체다. RuleModel 이 규칙 기반이고,
adc/cnn.Model 이 학습 모델이다. 다이 가림 시험과 반사실은 분류기 종류와 상관없이 같은 방법으로 계산한다.
"""
from __future__ import annotations
import math

import numpy as np

from .knowledge import CLASS_WEIGHTS, FEATURE_LABELS, LABEL_THRESHOLD, PATTERNS

HEAVY_COUNT = 3                       # 다이당 결함 3개 이상이면 heavy
ZONE_EDGES = (0.33, 0.75)             # 최대 다이 중심 반경 대비 비율
ZONES = ("center", "middle", "edge")
CHANNELS = ("all", "stain", "scratch", "particle")
MIN_CHANNEL_DIES = 3                  # 유형별 채널은 이 수 이상일 때만 평가
CORE_MIN_DIES = 3                     # 이 수 이상 이어져야 패턴 구조로 본다
MAIN_GROUP_FULL = 12.0                # 최대 군집 크기 특징이 1이 되는 다이 수
ISOLATED_FULL = 6.0                   # 고립 결함 수 특징이 1이 되는 개수


def zone_of(rho: float) -> str:
    if rho < ZONE_EDGES[0]:
        return "center"
    return "middle" if rho < ZONE_EDGES[1] else "edge"


def annotate(dies: list[dict]) -> list[dict]:
    """다이에 반경 비율·존·섹터를 붙인다."""
    rmax = max(math.hypot(d["x_mm"], d["y_mm"]) for d in dies)
    out = []
    for d in dies:
        r = math.hypot(d["x_mm"], d["y_mm"])
        rho = r / rmax
        angle = math.atan2(d["y_mm"], d["x_mm"])
        out.append({**d, "rho": round(rho, 3), "zone": zone_of(rho),
                    "sector": int(((angle + math.pi) / (2 * math.pi)) * 8) % 8})
    return out


def die_weight(counts: dict) -> float:
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return 1.0 if n >= HEAVY_COUNT else 0.6


def _components(ids: list[int], by_id: dict) -> list[list[int]]:
    """8방향 인접으로 묶은 연결 요소 (다이 id 묶음)."""
    pos = {(by_id[i]["row"], by_id[i]["col"]): i for i in ids}
    seen, groups = set(), []
    for p in pos:
        if p in seen:
            continue
        stack, members = [p], []
        seen.add(p)
        while stack:
            r, c = stack.pop()
            members.append(pos[(r, c)])
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in pos and q not in seen:
                        seen.add(q)
                        stack.append(q)
        groups.append(members)
    return groups


def extract_features(dies: list[dict], weights: dict[int, float], total_defect_dies: int | None = None,
                     core_min: int = CORE_MIN_DIES) -> dict:
    """결함 다이의 공간 분포를 0~1 특징으로 요약한다.
    3개 이상 이어진 다이는 패턴 구조(core), 나머지는 고립 결함으로 나눈다.
    고립 파티클 몇 개가 군집·직선·각도 특징을 흔들지 않게 하기 위해서다.
    total_defect_dies: 유형별 채널에서도 웨이퍼 전체 불량 수 기준의 density·포화도를 쓰기 위한 값.
    core_min: 구조로 인정할 최소 연결 크기. 스크래치 채널은 선이 끊겨도 한 결함이므로 1을 쓴다."""
    by_id = {d["id"]: d for d in dies}
    ids = [i for i, w in weights.items() if w > 0]
    f = {k: 0.0 for k in FEATURE_LABELS}
    if not ids:
        return f

    n_total = len(dies)
    n_defect = total_defect_dies if total_defect_dies is not None else len(ids)
    f["density"] = n_defect / n_total
    f["saturation"] = min(1.0, max(0.0, (f["density"] - 0.6) / 0.3))

    groups = _components(ids, by_id)
    core = [i for g in groups if len(g) >= core_min for i in g]
    # 산발 정도는 최대 군집 밖의 결함 수로 잰다. 무작위 다이도 우연히 3개쯤은 이어질 수 있기 때문이다.
    outside = len(ids) - max(len(g) for g in groups)
    f["scatter"] = outside / len(ids)
    f["isolated"] = min(1.0, outside / ISOLATED_FULL)
    f["main_group"] = min(1.0, max(len(g) for g in groups) / MAIN_GROUP_FULL)
    if not core:
        return {k: round(float(v), 4) for k, v in f.items()}

    w = np.array([weights[i] for i in core], dtype=float)
    xy = np.array([[by_id[i]["x_mm"], by_id[i]["y_mm"]] for i in core], dtype=float)
    wsum = w.sum()

    for z in ZONES:
        size = sum(1 for d in dies if d["zone"] == z) or 1
        zw = sum(weights[i] for i in core if by_id[i]["zone"] == z)
        f[f"{z}_fill"] = zw / size
        f[f"{z}_share"] = zw / wsum
    f["hole"] = max(0.0, f["middle_fill"] - f["center_fill"])

    sectors = np.zeros(8)
    for i in core:
        sectors[by_id[i]["sector"]] += weights[i]
    p = sectors[sectors > 0] / wsum
    f["angular"] = float(-(p * np.log(p)).sum() / math.log(8))

    if len(core) >= 3:
        mean = (xy * w[:, None]).sum(0) / wsum
        dxy = xy - mean
        cov = (dxy * w[:, None]).T @ dxy / wsum
        vals, vecs = np.linalg.eigh(cov)
        lam2, lam1 = float(vals[0]), float(vals[1])
        f["linearity"] = 1 - lam2 / lam1 if lam1 > 1e-9 else 0.0
        axis = vecs[:, 1]
        along = dxy @ axis
        across = np.abs(dxy @ np.array([-axis[1], axis[0]]))
        pitch = abs(dies[0]["x_mm"] - dies[1]["x_mm"])
        near = w[across <= pitch * 0.75].sum() / wsum
        span = float(along.max() - along.min())
        f["line_fit"] = float(near * min(1.0, span / (pitch * 5)))

    f["cluster"] = max(len(g) for g in groups) / len(core)
    return {k: round(float(v), 4) for k, v in f.items()}


def _channel_weights(defects: dict[int, dict], channel: str) -> dict[int, float]:
    if channel == "all":
        return {i: die_weight(c) for i, c in defects.items() if sum(c.values())}
    return {i: die_weight({channel: c.get(channel, 0)}) for i, c in defects.items() if c.get(channel, 0)}


def _score_class(cls: str, feats: dict) -> tuple[float, float, list[dict]]:
    spec = CLASS_WEIGHTS[cls]
    logit, contrib = spec["bias"], []
    for name, weight in spec["weights"].items():
        impact = weight * feats[name]
        logit += impact
        contrib.append({"feature": name, "label": FEATURE_LABELS[name], "value": feats[name],
                        "weight": weight, "impact": round(impact, 3)})
    contrib.sort(key=lambda c: abs(c["impact"]), reverse=True)
    return 1 / (1 + math.exp(-logit)), logit, contrib


def predict(dies: list[dict], defects: dict[int, dict]) -> dict:
    """8종 패턴 점수. 혼합 결함을 위해 전체와 유형별 채널을 각각 평가하고 클래스별 최댓값을 취한다.
    반환: {클래스: {score, logit, channel, bias, contributions}}"""
    results = {}
    feats_by_channel = {}
    n_defect = len(_channel_weights(defects, "all"))
    for ch in CHANNELS:
        weights = _channel_weights(defects, ch)
        if ch != "all" and len(weights) < MIN_CHANNEL_DIES:
            continue
        if weights:
            feats_by_channel[ch] = extract_features(dies, weights, n_defect, 1 if ch == "scratch" else CORE_MIN_DIES)

    for cls in PATTERNS:
        best = None
        for ch, feats in feats_by_channel.items():
            if cls == "Near-full" and ch != "all":
                continue
            if cls == "Scratch" and ch not in ("all", "scratch"):
                continue
            score, logit, contrib = _score_class(cls, feats)
            if best is None or score > best["score"]:
                best = {"score": round(score, 4), "logit": round(logit, 3), "channel": ch,
                        "bias": CLASS_WEIGHTS[cls]["bias"], "contributions": contrib}
        results[cls] = best or {"score": 0.0, "logit": CLASS_WEIGHTS[cls]["bias"], "channel": None,
                                "bias": CLASS_WEIGHTS[cls]["bias"], "contributions": []}
    return results


class RuleModel:
    """규칙 기반 공간 특징 분류기"""
    kind = "rule"

    def __init__(self, dies: list[dict], defects: dict[int, dict]):
        self.dies, self.defects = dies, defects
        self.ids = sorted(defects)

    def scores(self, drop=()) -> dict:
        return predict(self.dies, {k: v for k, v in self.defects.items() if k not in drop})

    def threshold(self, cls: str) -> float:
        return LABEL_THRESHOLD


def labels_from(scores: dict, model=None) -> list[str]:
    thr = (lambda c: model.threshold(c)) if model else (lambda c: LABEL_THRESHOLD)
    return sorted((c for c, s in scores.items() if s["score"] >= thr(c)), key=lambda c: scores[c]["score"], reverse=True)


def occlusion_map(model, target: str, base_score: float) -> dict[int, float]:
    """다이 가림 시험: 결함 다이를 하나씩 지우고 target 점수가 얼마나 변하는지 잰다.
    양수면 그 다이가 판정을 지지하고, 음수면 판정을 방해한다. 모델 내부를 몰라도 되는 방법이다."""
    return {i: round(base_score - model.scores({i})[target]["score"], 4) for i in model.ids}


def counterfactual(model, target: str, relevance: dict[int, float], max_steps: int = 14) -> dict:
    """target 판정이 사라지는 최소 근사 다이 집합을 탐욕적으로 찾는다."""
    removed = []
    order = sorted(relevance, key=relevance.get, reverse=True)
    scores = model.scores()
    for i in order[:max_steps]:
        removed.append(i)
        scores = model.scores(set(removed))
        if scores[target]["score"] < model.threshold(target):
            labels = labels_from(scores, model)
            return {"found": True, "dies": removed, "target": target,
                    "after": labels[0] if labels else "Normal",
                    "after_score": round(scores[labels[0]]["score"], 3) if labels else None,
                    "target_after": scores[target]["score"]}
    return {"found": False, "dies": removed, "target": target, "after": None,
            "after_score": None, "target_after": scores[target]["score"]}
