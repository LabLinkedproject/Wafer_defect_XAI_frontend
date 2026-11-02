"""
adc/render.py
검출 결과에 맞는 웨이퍼 입력 이미지를 합성하고, 결함을 유형별 표현으로 변환한다.
    stain            : 연속 면적 결함 → 다이 합집합 폴리곤 (도넛의 구멍 포함)
    particle/scratch : 다이별 박스

합성 이미지는 시연용이다. 실제 촬영본은 /scan 의 스캔 결과를 쓴다.
"""
from __future__ import annotations
import math
import random

import cv2
import numpy as np

IMAGE_SIZE = 620
PADDING = 22


def px_per_mm(diameter_mm: float) -> float:
    return (IMAGE_SIZE - 2 * PADDING) / diameter_mm


def to_px(x_mm: float, y_mm: float, s: float) -> tuple[int, int]:
    return int(IMAGE_SIZE / 2 + x_mm * s), int(IMAGE_SIZE / 2 - y_mm * s)


def _die_rect(d, pitch, s):
    x, y = to_px(d["x_mm"] - pitch / 2, d["y_mm"] + pitch / 2, s)
    side = int(pitch * s)
    return x, y, side


def _stain_mask(stain_dies, pitch, s):
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    for d in stain_dies:
        x, y, side = _die_rect(d, pitch, s)
        cv2.rectangle(mask, (x, y), (x + side, y + side), 255, -1)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    return cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)[1]


def _distribution(regions: int, has_hole: bool, location: str) -> str:
    if regions == 1:
        if has_hole:
            return "Ring (Donut)"
        return {"center": "Center Cluster", "edge": "Edge Cluster"}.get(location, "Local Cluster")
    if regions == 2:
        return "Dual Cluster"
    return "Multi-cluster" if regions <= 4 else "Scattered"


def detect_regions(dies, defects, pitch, diameter):
    """결함 다이 → 화면 오버레이용 검출 표현 + Stain 영역 지표."""
    s = px_per_mm(diameter)
    by_id = {d["id"]: d for d in dies}
    stain_dies = [by_id[i] for i, c in defects.items() if c.get("stain", 0) > 0]
    out, metrics = [], None

    if stain_dies:
        mask = _stain_mask(stain_dies, pitch, s)
        contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 80:
                continue
            approx = cv2.approxPolyDP(cnt, 0.003 * cv2.arcLength(cnt, True), True)
            polygons.append({"is_hole": bool(hier[0][i][3] != -1),
                             "points": [[int(p[0][0]), int(p[0][1])] for p in approx]})
        regions = sum(1 for p in polygons if not p["is_hole"])
        has_hole = any(p["is_hole"] for p in polygons)
        zones = {}
        for d in stain_dies:
            zones[d["zone"]] = zones.get(d["zone"], 0) + 1
        location = max(zones, key=zones.get)
        m = cv2.moments(mask)
        cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (IMAGE_SIZE / 2, IMAGE_SIZE / 2)
        cx_mm, cy_mm = (cx - IMAGE_SIZE / 2) / s, (IMAGE_SIZE / 2 - cy) / s
        metrics = {
            "region_count": regions, "has_hole": has_hole,
            "area_mm2": round(float((mask > 0).sum()) / s ** 2, 1),
            "location": location,
            "centroid_mm": {"x": round(cx_mm, 1), "y": round(cy_mm, 1)},
            "centroid_distance_mm": round(math.hypot(cx_mm, cy_mm), 1),
            "distribution": _distribution(regions, has_hole, location),
            "die_count": len(stain_dies),
            "coverage_pct": round(len(stain_dies) / len(dies) * 100, 1),
        }
        out.append({"shape": "polygon", "type": "stain", "label": "stain", "polygons": polygons,
                    "die_count": len(stain_dies)})

    for t in ("particle", "scratch"):
        for i, c in defects.items():
            if c.get(t, 0) > 0:
                x, y, side = _die_rect(by_id[i], pitch, s)
                pad = int(side * 0.16)
                out.append({"shape": "bbox", "type": t, "label": t, "die": i, "count": c[t],
                            "bbox": [x + pad, y + pad, side - 2 * pad, side - 2 * pad]})
    return out, metrics


def render_input(dies, defects, pitch, diameter) -> bytes:
    """결함 분포를 반영한 합성 웨이퍼 이미지 (JPEG)."""
    s = px_per_mm(diameter)
    rng = random.Random(7)
    np_rng = np.random.default_rng(7)
    c = IMAGE_SIZE // 2
    R = int(diameter / 2 * s)
    by_id = {d["id"]: d for d in dies}

    img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 22, np.uint8)
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    rr = np.hypot(xx - c, yy - c) / R
    shade = 172 - 34 * rr + 16 * np.sin((xx + yy) / 150)
    body = np.dstack([shade + 8, shade + 2, shade - 6]).clip(0, 255).astype(np.uint8)
    disk = rr <= 1
    img[disk] = body[disk]

    for d in dies:
        x, y, side = _die_rect(d, pitch, s)
        cv2.rectangle(img, (x + 1, y + 1), (x + side - 1, y + side - 1), (205, 200, 192), -1)
        cv2.rectangle(img, (x, y), (x + side, y + side), (92, 92, 96), 1)

    stain_dies = [by_id[i] for i, k in defects.items() if k.get("stain", 0) > 0]
    if stain_dies:
        soft = cv2.GaussianBlur(_stain_mask(stain_dies, pitch, s), (31, 31), 0).astype(float) / 255
        tint = np.array([42, 72, 104], float)
        img = (img * (1 - 0.72 * soft[..., None]) + tint * 0.72 * soft[..., None]).astype(np.uint8)

    for i, k in defects.items():
        d = by_id[i]
        x, y, side = _die_rect(d, pitch, s)
        for _ in range(k.get("particle", 0) * 5):
            p = (x + rng.randint(5, side - 5), y + rng.randint(5, side - 5))
            cv2.circle(img, p, rng.choice((1, 2, 2)), (30, 34, 40), -1, cv2.LINE_AA)

    # 스크래치는 다이마다 끊지 않고 결함 다이를 잇는 한 줄로 그린다
    pts = np.array([to_px(by_id[i]["x_mm"], by_id[i]["y_mm"], s) for i, k in defects.items()
                    if k.get("scratch", 0) > 0], float)
    if len(pts) >= 2:
        mean = pts.mean(0)
        axis = np.linalg.eigh(np.cov((pts - mean).T))[1][:, 1]
        t = (pts - mean) @ axis
        half = pitch * s * 0.45
        ends = mean + np.outer([t.min() - half, t.max() + half], axis)
        p1, p2 = tuple(ends[0].astype(int).tolist()), tuple(ends[1].astype(int).tolist())
        cv2.line(img, p1, p2, (40, 44, 52), 4, cv2.LINE_AA)
        cv2.line(img, p1, p2, (250, 248, 244), 2, cv2.LINE_AA)

    noise = np_rng.normal(0, 3, img.shape)
    img = (img.astype(float) + noise).clip(0, 255).astype(np.uint8)
    cv2.circle(img, (c, c), R, (70, 72, 78), 2, cv2.LINE_AA)
    notch = np.array([[c - 8, c + R + 1], [c + 8, c + R + 1], [c, c + R - 9]], np.int32)
    cv2.fillPoly(img, [notch], (22, 22, 22), cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()
