"""
adc/engine.py
다이별 결함 → ADC 판정 → XAI 근거 → 능동 추론 → 공정 피드백까지 한 번에 계산한다.
화면은 이 결과 JSON 만 보고 그린다.
"""
from __future__ import annotations

from wafer_model import active_radius
from . import belief as B
from . import render, scoring as S
from .knowledge import (CHECKS, DEFECT_TYPES, HYPOTHESES, LABEL_THRESHOLD, PATTERNS, RECIPE_PARAMS,
                        SEVERITY_ADVICE)

SYSTEMATIC_BONUS = 15          # 구조적 패턴이면 더하는 위험도
ZONE_KO = {"center": "중앙", "middle": "중간", "edge": "엣지"}
ENGINES = {
    "detector":   {"stage": "검출", "now": "시나리오 시뮬레이션", "next": "YOLOv8", "live": False},
    "classifier": {"stage": "분류", "now": "규칙 기반 공간 특징 분류기", "next": "CNN (WM-811K 8종)", "live": False},
    "explainer":  {"stage": "설명", "now": "기여도 분해 · 다이 가림 시험 · 반사실", "next": "Grad-CAM 병행", "live": True},
    "inference":  {"stage": "추론", "now": "베이즈 갱신 + 정보 이득 질의", "next": "로그 연동·우도 학습", "live": True},
}
_HYP = {h["id"]: h for h in HYPOTHESES}
_CHECK = {c["id"]: c for c in CHECKS}


def _status(total: int) -> str:
    if total == 0:
        return "normal"
    return "heavy" if total >= S.HEAVY_COUNT else "light"


def _risk(die_stats: dict, top: str | None) -> dict:
    active = die_stats["active"] or 1
    heavy, light = die_stats["heavy"], die_stats["light"]
    ratio_term = (heavy + light) / active * 60
    heavy_term = min(heavy * 1.5, 25)
    pattern_term = SYSTEMATIC_BONUS if top not in (None, "Random") else 0
    score = max(0.0, min(100.0, ratio_term + heavy_term + pattern_term))
    level = "Low" if score < 30 else "Medium" if score < 60 else "High" if score < 80 else "Critical"
    return {"score": round(score, 1), "level": level, "advice": SEVERITY_ADVICE[level],
            "terms": [{"name": "불량 다이 비율", "value": round(ratio_term, 1), "max": 60},
                      {"name": "심한 다이 가산", "value": round(heavy_term, 1), "max": 25},
                      {"name": "구조적 패턴 가산", "value": pattern_term, "max": SYSTEMATIC_BONUS}]}


def _yield(die_stats: dict) -> dict:
    active = die_stats["active"] or 1
    pct = round(die_stats["normal"] / active * 100, 1)
    level = "Excellent" if pct >= 90 else "Good" if pct >= 75 else "Marginal" if pct >= 50 else "Poor"
    return {"yield_pct": pct, "level": level, "good_die": die_stats["normal"], "light_die": die_stats["light"],
            "heavy_die": die_stats["heavy"], "active_die": die_stats["active"]}


def _conclusion(labels, top_hyp, zones_top, yield_pct) -> str:
    if not labels:
        return "뚜렷한 결함 패턴이 없습니다. 공정은 정상 범위로 판단합니다."
    names = " + ".join(labels)
    if not top_hyp:
        return f"{names} 패턴이 검출됐습니다. 예상 수율은 {yield_pct}%입니다."
    return (f"{names} 패턴이 {ZONE_KO.get(zones_top, zones_top)} 존 중심으로 나타났고, 현재 가장 유력한 원인은 "
            f"{top_hyp['process']} 공정의 {top_hyp['name']}입니다.")


def _build_trace(ctx: dict) -> list[dict]:
    """판단이 어떻게 이어졌는지 사람이 읽을 수 있게 기록한다. 값은 모두 실제 계산 결과다."""
    t = [{"kind": "observe", "title": "관찰",
          "text": f"다이 {ctx['active']}개 중 {ctx['defect_dies']}개에서 결함 {ctx['total']}건을 확인했습니다."}]
    if not ctx["labels"]:
        t.append({"kind": "conclude", "title": "결론", "text": "판정선을 넘는 패턴이 없어 정상으로 판단합니다."})
        return t
    t.append({"kind": "classify", "title": "분류",
              "text": f"{' + '.join(ctx['labels'])} 판정. 1순위 점수 {ctx['top_score']:.2f}, 2순위와 차이 {ctx['margin']:.2f}."})
    c = ctx["contrib"]
    if c:
        t.append({"kind": "explain", "title": "근거",
                  "text": f"가장 큰 근거는 '{c['label']}'입니다 (값 {c['value']:.2f}, 로그오즈 {c['impact']:+.2f})."})
    if ctx["cf"].get("found"):
        after = ctx["cf"]["after"]
        t.append({"kind": "explain", "title": "반사실",
                  "text": f"결함 다이 {len(ctx['cf']['dies'])}개가 없었다면 판정은 {after}입니다."})
    if ctx["hyp"]:
        h = ctx["hyp"][0]
        t.append({"kind": "hypothesize", "title": "가설",
                  "text": f"영상 근거만으로 본 1순위 원인은 {ctx['image_top_name']}입니다 ({ctx['image_top_p']:.0%})."})
    for e in ctx["history"]:
        ans = {"yes": "예", "no": "아니오"}[e["answer"]]
        change = (f"{e['after_name']} 확률이 {e['before_p']:.0%}에서 {e['after_p']:.0%}가 됐습니다."
                  if e["before_name"] == e["after_name"] else
                  f"1순위가 {e['before_name']}({e['before_p']:.0%})에서 {e['after_name']}({e['after_p']:.0%})가 됐습니다.")
        t.append({"kind": "update", "title": "갱신", "text": f"'{e['ask']}' → {ans}. {change}"})
    if ctx["converged"]:
        h = ctx["hyp"][0]
        t.append({"kind": "conclude", "title": "수렴",
                  "text": f"{h['name']}({h['p']:.0%})로 수렴했습니다. 조치: {h['action']}."})
    elif ctx["checks"]:
        q = ctx["checks"][0]
        t.append({"kind": "ask", "title": "다음 확인",
                  "text": f"아직 원인을 좁히지 못했습니다. '{q['ask']}' 확인이 가장 유용합니다 "
                          f"(기대 정보 이득 {q['gain']:.2f} bit)."})
    return t


def analyze(dies, spec, defects, answers=None, target=None) -> dict:
    """dies: scoring.annotate 결과, defects: {다이 id: {유형: 개수}}, answers: [{check, answer}]"""
    answers = answers or []
    defects = {i: c for i, c in defects.items() if sum(c.values()) > 0}

    scores = S.predict(dies, defects)
    labels = S.labels_from(scores)
    ordered = sorted(PATTERNS, key=lambda c: scores[c]["score"], reverse=True)
    top = labels[0] if labels else None
    second = scores[ordered[1]]["score"]
    top_score = scores[ordered[0]]["score"]
    margin = round(top_score - second, 3)

    # XAI 대상: 지정이 없으면 1순위 라벨
    tgt = target if target in PATTERNS else (top or ordered[0])
    base = scores[tgt]["score"]
    relevance = S.occlusion_map(dies, defects, tgt, base) if defects else {}
    cf = S.counterfactual(dies, defects, tgt, relevance) if (defects and base >= LABEL_THRESHOLD) else {"found": False}

    # 다이·존·유형 집계
    die_rows, stats = [], {"active": len(dies), "normal": 0, "light": 0, "heavy": 0}
    zones = {z: {"dies": 0, "defect_dies": 0, "count": 0} for z in S.ZONES}
    sectors = [0] * 8
    type_tot = {t: {"count": 0, "dies": 0} for t in DEFECT_TYPES}
    for d in dies:
        c = defects.get(d["id"], {})
        total = sum(c.values())
        st = _status(total)
        stats[st] += 1
        zones[d["zone"]]["dies"] += 1
        zones[d["zone"]]["count"] += total
        zones[d["zone"]]["defect_dies"] += 1 if total else 0
        sectors[d["sector"]] += total
        for t in DEFECT_TYPES:
            if c.get(t, 0):
                type_tot[t]["count"] += c[t]
                type_tot[t]["dies"] += 1
        die_rows.append({"id": d["id"], "row": d["row"], "col": d["col"], "x_mm": d["x_mm"], "y_mm": d["y_mm"],
                         "zone": d["zone"], "counts": {t: c.get(t, 0) for t in DEFECT_TYPES}, "total": total,
                         "type": max(c, key=c.get) if total else None, "status": st,
                         "relevance": relevance.get(d["id"], 0.0)})
    for z in zones.values():
        z["fill_pct"] = round(z["defect_dies"] / (z["dies"] or 1) * 100, 1)
    total_defects = sum(v["count"] for v in type_tot.values())
    shares = {t: (v["count"] / total_defects if total_defects else 0.0) for t, v in type_tot.items()}
    dominant_zone = max(zones, key=lambda z: zones[z]["count"]) if total_defects else None
    dominant_type = max(shares, key=shares.get) if total_defects else None

    risk = _risk(stats, top)
    yld = _yield(stats)

    # 능동 추론
    hyp_list, checks, history, converged, h_now = [], [], [], False, None
    image_belief = None
    if labels:
        evidence = {c: scores[c]["score"] for c in PATTERNS}
        image_belief = B.prior_from_evidence(evidence, shares)
        cur = dict(image_belief)
        asked = set()
        for a in answers:
            asked.add(a["check"])
            if a["answer"] not in ("yes", "no"):
                continue
            prev_id = max(cur, key=cur.get)
            prev_p = cur[prev_id]
            cur = B.update(cur, a["check"], a["answer"])
            new_id = max(cur, key=cur.get)
            history.append({"check": a["check"], "ask": _CHECK[a["check"]]["ask"], "answer": a["answer"],
                            "before_name": _HYP[prev_id]["name"], "before_p": round(prev_p, 3),
                            "after_name": _HYP[new_id]["name"], "after_p": round(cur[new_id], 3)})
        ranked = sorted(cur.items(), key=lambda kv: kv[1], reverse=True)
        hyp_list = [{"id": i, "name": _HYP[i]["name"], "process": _HYP[i]["process"], "p": round(p, 4),
                     "image_p": round(image_belief[i], 4), "action": _HYP[i]["action"], "params": _HYP[i]["params"]}
                    for i, p in ranked[:6]]
        h_now = round(B.entropy(cur), 3)
        converged = ranked[0][1] >= B.CONVERGED
        checks = B.rank_checks(cur, asked)
        for c in checks:
            c["target_name"] = _HYP[c["target"]]["name"]

    # 회복 상한: 판정을 지지하는 다이의 결함이 모두 해소된다고 가정한 최댓값 (실제 회복은 이보다 낮을 수 있다)
    support = [d for d in die_rows if d["relevance"] > 0]
    projected = None
    if labels and yld["active_die"]:
        recovered = sum(1 for d in support if d["total"] > 0)
        projected = {"yield_pct": round(min(100.0, (stats["normal"] + recovered) / yld["active_die"] * 100), 1),
                     "recovered_die": recovered}

    recipe = []
    related = set(hyp_list[0]["params"]) if hyp_list else set()
    for name, r in RECIPE_PARAMS.items():
        recipe.append({"param": name, **r, "related": name in related})

    image_top = max(image_belief, key=image_belief.get) if image_belief else None
    trace = _build_trace({
        "active": len(dies), "defect_dies": len(defects), "total": total_defects, "labels": labels,
        "top_score": top_score, "margin": margin, "cf": cf,
        "contrib": next((c for c in scores[tgt]["contributions"] if c["impact"] > 0), None),
        "hyp": hyp_list, "history": history, "converged": converged, "checks": checks,
        "image_top_name": _HYP[image_top]["name"] if image_top else None,
        "image_top_p": image_belief[image_top] if image_top else 0,
    })

    detections, stain = render.detect_regions(dies, defects, spec["pitch_mm"], spec["diameter_mm"])
    return {
        "engines": ENGINES,
        "spec": {"diameter_mm": round(spec["diameter_mm"], 1), "pitch_mm": spec["pitch_mm"],
                 "min_die_area": spec["min_die_area"], "die_count": len(dies), "grid": spec["grid"],
                 "active_radius_mm": round(active_radius(spec), 2)},
        "dies": die_rows,
        "zones": zones, "sectors": sectors,
        "types": {t: {**v, "ko": DEFECT_TYPES[t]["ko"], "share": round(shares[t], 3)} for t, v in type_tot.items()},
        "total_defects": total_defects, "defect_dies": len(defects),
        "dominant_zone": dominant_zone, "dominant_type": dominant_type,
        "threshold": LABEL_THRESHOLD,
        "patterns": [{"name": c, **PATTERNS[c], "score": scores[c]["score"], "channel": scores[c]["channel"],
                      "is_label": c in labels} for c in PATTERNS],
        "labels": labels, "top": top, "top_score": top_score, "margin": margin, "ambiguous": bool(labels) and margin < 0.2,
        "xai": {"target": tgt, "score": scores[tgt]["score"], "logit": scores[tgt]["logit"], "bias": scores[tgt]["bias"],
                "channel": scores[tgt]["channel"], "contributions": scores[tgt]["contributions"],
                "counterfactual": cf},
        "belief": {"hypotheses": hyp_list, "checks": checks, "history": history, "converged": converged,
                   "entropy": h_now, "answered": [a["check"] for a in answers]},
        "risk": risk, "yield": yld, "projected": projected, "recipe": recipe,
        "stain": stain, "detections": detections, "image_size": render.IMAGE_SIZE,
        "conclusion": _conclusion(labels, hyp_list[0] if hyp_list else None, dominant_zone, yld["yield_pct"]),
        "trace": trace,
    }
