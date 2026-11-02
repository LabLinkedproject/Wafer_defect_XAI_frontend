"""
adc/belief.py
능동 추론: 패턴 판정을 증거로 공정 원인 가설의 확률을 계산하고,
다음에 무엇을 확인해야 불확실성이 가장 많이 줄어드는지 스스로 고른 뒤,
엔지니어의 답을 받아 판단을 갱신한다.

    사후확률 ∝ 사전확률 × Π 패턴 우도^점수 × Π 유형 우도^비중 × Π 확인 결과 우도

⚠ 우도는 공정 지식 기반 추정값(knowledge.py)이다. 현장 이력이 쌓이면 데이터로 다시 추정해야 한다.
"""
from __future__ import annotations
import math

from .knowledge import CHECKS, HYPOTHESES

FLOOR = 0.03              # 지식표에 없는 패턴·유형의 우도 하한
CONVERGED = 0.70          # 1순위 가설이 이 값을 넘으면 수렴으로 본다
MIN_SCORE = 0.15          # 이보다 낮은 패턴 점수는 증거에서 제외 (잡음 누적 방지)

_CHECK_BY_ID = {c["id"]: c for c in CHECKS}


def _p_yes(check: dict, hyp_id: str) -> float:
    return check["yes"].get(hyp_id, check["default"])


def _normalize(log_w: dict[str, float]) -> dict[str, float]:
    m = max(log_w.values())
    w = {k: math.exp(v - m) for k, v in log_w.items()}
    t = sum(w.values())
    return {k: v / t for k, v in w.items()}


def entropy(p: dict[str, float]) -> float:
    return -sum(v * math.log2(v) for v in p.values() if v > 0)


def prior_from_evidence(pattern_scores: dict[str, float], type_shares: dict[str, float]) -> dict[str, float]:
    """영상 근거만 반영한 가설 확률 (현장 확인 전)."""
    log_w = {}
    for h in HYPOTHESES:
        v = math.log(h["prior"])
        for cls, score in pattern_scores.items():
            if score >= MIN_SCORE:
                v += score * math.log(h["patterns"].get(cls, FLOOR))
        for t, share in type_shares.items():
            if share > 0:
                v += share * math.log(h["types"].get(t, FLOOR))
        log_w[h["id"]] = v
    return _normalize(log_w)


def update(belief: dict[str, float], check_id: str, answer: str) -> dict[str, float]:
    """베이즈 갱신. answer 는 yes / no. '모름'은 정보가 없으므로 호출하지 않는다."""
    check = _CHECK_BY_ID[check_id]
    log_w = {}
    for hid, p in belief.items():
        like = _p_yes(check, hid) if answer == "yes" else 1 - _p_yes(check, hid)
        log_w[hid] = math.log(max(p, 1e-12)) + math.log(max(like, 1e-6))
    return _normalize(log_w)


def rank_checks(belief: dict[str, float], asked: set[str], top: int = 3) -> list[dict]:
    """기대 정보 이득이 큰 순서로 다음 확인 항목을 고른다.
    이득 = 현재 엔트로피 − 답을 얻은 뒤 기대 엔트로피 (비트)"""
    h0 = entropy(belief)
    ranked = []
    for c in CHECKS:
        if c["id"] in asked:
            continue
        p_yes = sum(p * _p_yes(c, hid) for hid, p in belief.items())
        h_yes = entropy(update(belief, c["id"], "yes"))
        h_no = entropy(update(belief, c["id"], "no"))
        gain = h0 - (p_yes * h_yes + (1 - p_yes) * h_no)
        target = max(belief, key=lambda hid: belief[hid] * (_p_yes(c, hid) - c["default"]))
        ranked.append({"id": c["id"], "ask": c["ask"], "where": c["where"],
                       "gain": round(gain, 3), "p_yes": round(p_yes, 3), "target": target})
    ranked.sort(key=lambda r: r["gain"], reverse=True)
    return [r for r in ranked[:top] if r["gain"] > 0.005]
