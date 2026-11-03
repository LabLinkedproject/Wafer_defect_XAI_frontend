"""
wafer_detect.py
합성 웨이퍼 맵에서 결함을 찾는다. 다이 44개는 모양이 같으므로 라벨 없이 다이끼리 비교할 수 있다.

표준 기준 다이 방식을 따른다.
1. 측정한 격자로 맵을 곧게 펴고 다이마다 잘라낸다.
2. 다이를 서로 미세 정렬한 뒤 화소별 중앙값으로 기준 다이를 만든다. 결함은 다이마다 위치가 달라 중앙값에서 사라진다.
3. 다이와 기준의 차이를 강건한 분산(MAD)으로 나누어, 기준값을 스캔마다 자동으로 정한다.

한계: 여러 다이의 같은 위치에 반복되는 결함은 중앙값에 흡수되어 검출되지 않는다.
웨이퍼 전체를 덮는 Near-full 도 기준이 오염되어 약하다.
"""
from __future__ import annotations
import json
import math
import os

import cv2
import numpy as np

import wafer_raster as R
from wafer_model import WAFER_MODEL, die_layout

SCRIBE_MARGIN = 12            # 스크라이브 라인은 정렬 오차에 민감해 다이 가장자리는 비교에서 뺀다
MIN_DIE_COVERAGE = 0.92       # 이보다 적게 찍힌 다이는 검사하지 않는다
MAX_SHIFT = 8                 # 미세 정렬 허용 폭 (px)
SHARP_Z = 7.0                 # 선명한 결함 판정선
STAIN_Z = 7.0                 # 번짐 결함 판정선
MIN_CONTRAST = 0.05           # 맵 합성 잔차보다 작은 밝기 차이는 결함으로 보지 않는다
MIN_STAIN_CONTRAST = 0.02
MIN_SHARP_AREA = 5            # px
MIN_STAIN_RATIO = 0.012       # 다이 면적 대비
SCRATCH_ELONGATION = 3.5
SCRATCH_MIN_MM = 4.0
HEAVY_QUADRAT_AREA = 0.05     # 구역 면적 중 이 비율 이상이면 heavy
HEAVY_QUADRAT_COUNT = 3
LINE_GUARD_MM = 5.0           # 다이 경계선에서 이 거리 안의 가는 성분은 격자선 잔차
LINE_RESIDUAL_ELONGATION = 3.0
GROW_RATIO = 0.6              # 이중 문턱의 낮은 쪽 (높은 문턱 대비)
DARK_BACKGROUND = 80          # 웨이퍼 안쪽 밝기 중앙값이 이보다 어두우면 색 차이 채널로 검출
CHROMA_GAIN = 3.0             # R−B 차이를 밝기 영상 범위로 늘리는 배율
CHROMA_Z = 6.0                # 색 차이 채널의 선명 결함 판정선 (흑백은 SHARP_Z)
MIN_CHROMA_AREA_MM2 = 8.0     # 색 차이 채널은 잡음이 거칠어 이보다 작은 점은 결함으로 세지 않는다 (지름 약 3 mm)
OUTLIER_RATIO = 0.25          # 국소 배경에서 이만큼 벗어난 픽셀은 배경 평균에서 뺀다
MAX_DARK_DEFECT_MM = 12        # 이보다 작은 어두운 구멍은 미촬영이 아니라 검은 결함
BACKGROUND_MAX = 26           # 합성 맵의 미촬영 영역 밝기 (18~20)


def _seen(gray: np.ndarray, pitch_px: float) -> np.ndarray:
    """촬영된 영역. 맵의 미촬영 부분은 어둡게 칠해지지만 검은 결함도 그만큼 어둡다.
    촬영 영역 안에 갇힌 작은 어두운 구멍은 결함으로 보고 촬영된 것으로 친다."""
    seen = gray > BACKGROUND_MAX
    n, lab, stats, _ = cv2.connectedComponentsWithStats((~seen).astype(np.uint8), connectivity=8)
    h, w = gray.shape
    limit = (MAX_DARK_DEFECT_MM * pitch_px / WAFER_MODEL["pitch_mm"]) ** 2
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area <= limit and x > 0 and y > 0 and x + bw < w and y + bh < h:
            seen[lab == i] = True
    return seen


def rectify(gray: np.ndarray, meta: dict, seen: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """맵을 웨이퍼 중심·격자 기준으로 곧게 편다. 반환: 영상, 유효 영역, 다이 한 칸 px
    seen: 맵 합성이 저장한 촬영 범위 마스크. 없으면 밝기로 추정한다(웨이퍼가 밝게 찍힐 때만 맞다)."""
    pitch = meta["grid"]["pitch_px"]
    P = int(round(pitch))
    side = R.GRID_DIES * P
    h = side / 2
    k = pitch / P
    a = math.radians(-meta["grid"]["angle_deg"])
    c, s = math.cos(a) * k, math.sin(a) * k
    cx, cy = meta["wafer"]["center_px"]
    M = np.array([[c, s, cx - h * (c + s)], [-s, c, cy - h * (c - s)]], np.float32)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    img = cv2.warpAffine(gray.astype(np.float32), M, (side, side), flags=flags)
    seen = _seen(gray, pitch) if seen is None else seen > 127
    seen = cv2.warpAffine(seen.astype(np.float32), M, (side, side), flags=flags) > 0.99
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    disc = np.hypot(xx - h + 0.5, yy - h + 0.5) < meta["wafer"]["radius_px"] / k * 0.985
    valid = cv2.erode((seen & disc).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0   # 촬영 경계는 합성 잔차가 크다
    return img, valid, P


def crop_dies(img, valid, P, dies):
    """검사할 수 있는 다이만 잘라낸다."""
    out = {}
    for d in dies:
        r, c = R.die_slot(d)
        y, x = r * P, c * P
        v = valid[y:y + P, x:x + P]
        if v.mean() >= MIN_DIE_COVERAGE:
            out[d["id"]] = {"die": d, "y": y, "x": x, "img": img[y:y + P, x:x + P].copy(), "valid": v.copy()}
    return out


def _flatten(a: np.ndarray, sigma: float) -> np.ndarray:
    """국소 배경으로 나눈다. 아주 어두운 결함이 배경 평균을 끌어내려 주변에 가짜 테두리가 생기지 않도록
    배경에서 크게 벗어난 픽셀을 빼고 다시 평균한다."""
    base = cv2.GaussianBlur(a, (0, 0), sigma)
    keep = (np.abs(a - base) < OUTLIER_RATIO * np.maximum(base, 1.0)).astype(np.float32)
    w = cv2.GaussianBlur(keep, (0, 0), sigma)
    base = np.where(w > 0.05, cv2.GaussianBlur(a * keep, (0, 0), sigma) / np.maximum(w, 1e-6), base)
    return a / np.maximum(base, 1.0)


def align(crops: dict, P: int):
    """다이를 임시 기준에 맞춘다. 합성 잔차가 1~2 px 남기 때문에 필요하다."""
    win = cv2.createHanningWindow((P, P), cv2.CV_32F)
    stack = np.stack([c["img"] for c in crops.values()])
    ref = _flatten(np.median(stack, 0), P / 6)
    for c in crops.values():
        (dx, dy), _ = cv2.phaseCorrelate(ref.astype(np.float32), _flatten(c["img"], P / 6).astype(np.float32), win)
        dx, dy = float(np.clip(dx, -MAX_SHIFT, MAX_SHIFT)), float(np.clip(dy, -MAX_SHIFT, MAX_SHIFT))
        M = np.array([[1, 0, -dx], [0, 1, -dy]], np.float32)
        c["img"] = cv2.warpAffine(c["img"], M, (P, P), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        c["valid"] = cv2.warpAffine(c["valid"].astype(np.uint8), M, (P, P), flags=cv2.INTER_NEAREST) > 0
        c["shift"] = (round(dx, 2), round(dy, 2))


def robust_sigma(a: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def analyse(crops: dict, P: int) -> dict:
    """다이별 z 지도를 만든다."""
    stack = np.stack([c["img"] for c in crops.values()])
    ref = np.median(stack, 0)
    inner = np.zeros((P, P), bool)
    inner[SCRIBE_MARGIN:-SCRIBE_MARGIN, SCRIBE_MARGIN:-SCRIBE_MARGIN] = True
    ref_sharp = _flatten(ref, P / 6)
    ref_soft = cv2.GaussianBlur(ref, (0, 0), P / 14) / np.median(ref[inner])
    sharp_all, soft_all = [], []
    for c in crops.values():
        m = c["valid"] & inner
        c["mask"] = m
        c["sharp"] = _flatten(c["img"], P / 6) - ref_sharp
        c["soft"] = cv2.GaussianBlur(c["img"], (0, 0), P / 14) / np.median(c["img"][m]) - ref_soft
        sharp_all.append(c["sharp"][m])
        soft_all.append(c["soft"][m])
    s1, s2 = robust_sigma(np.concatenate(sharp_all)), robust_sigma(np.concatenate(soft_all))
    return {"sharp_sigma": s1, "soft_sigma": s2, "reference": ref}


def _components(mask: np.ndarray, min_area: int):
    n, lab, stats, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(lab == i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def _grid_residual(xs, ys, w, h, area_mm2, P, mm) -> bool:
    """다이 경계선 옆 띠 안에 갇힌 길쭉한 성분은 격자선 위치 차이가 남은 것으로 본다.
    격자선이 다이마다 조금씩 어긋나 있으면 기준 다이와 비교할 때 선이 결함처럼 남는다."""
    guard = LINE_GUARD_MM / mm
    near_x = min(xs.min(), P - 1 - xs.max()) < guard and np.ptp(xs) < guard     # 세로 경계선 옆
    near_y = min(ys.min(), P - 1 - ys.max()) < guard and np.ptp(ys) < guard     # 가로 경계선 옆
    along = max(w, h) >= LINE_RESIDUAL_ELONGATION * max(min(w, h), 1.0)     # 경계선 방향으로 길쭉함
    return (near_x or near_y) and (along or area_mm2 < 1.0)


def extract(crops: dict, P: int, noise: dict, min_area_mm2: float = 0.0, sharp_z: float = SHARP_Z) -> list[dict]:
    """z 지도에서 결함 덩어리를 뽑아 유형과 웨이퍼 좌표를 붙인다."""
    mm = WAFER_MODEL["pitch_mm"] / P
    found = []
    for c in crops.values():
        thr = max(sharp_z * noise["sharp_sigma"], MIN_CONTRAST)
        z = np.abs(c["sharp"])
        # 이중 문턱: 확실한 픽셀에서 출발해 절반 문턱까지 이어 붙인다. 대비가 약한 면에서 결함 하나가
        # 조각나는 것을 막고, 떨어져 있는 두 결함은 합치지 않는다
        weak = cv2.morphologyEx(((z > thr * GROW_RATIO) & c["mask"]).astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8))
        n, lab = cv2.connectedComponents(weak, connectivity=8)
        seeds = np.unique(lab[(z > thr) & c["mask"] & (weak > 0)])
        sharp = np.isin(lab, seeds[seeds > 0])
        soft = (np.abs(c["soft"]) > max(STAIN_Z * noise["soft_sigma"], MIN_STAIN_CONTRAST)) & c["mask"]
        stain_min = int(MIN_STAIN_RATIO * P * P)
        c["pixels"] = np.zeros((P, P), bool)
        # 번짐 지도는 흐림을 거치므로 선명한 결함 둘레에도 반응한다. 그 둘레는 번짐 결함에서 뺀다
        halo = cv2.dilate(sharp.astype(np.uint8), np.ones((2 * (P // 14) + 1,) * 2, np.uint8)) > 0
        for region, kind_min, is_stain in ((sharp, MIN_SHARP_AREA, False), (soft & ~halo, stain_min, True)):
            for comp, area in _components(region, kind_min):
                if area * mm * mm < min_area_mm2:
                    continue
                ys, xs = np.where(comp)
                pts = np.column_stack([xs, ys]).astype(np.float32)
                (_, _), (w, h), _ = cv2.minAreaRect(pts)
                long_mm, short = max(w, h) * mm, max(min(w, h), 1.0)
                elong = max(w, h) / short
                if not is_stain and _grid_residual(xs, ys, w, h, area * mm * mm, P, mm):
                    continue
                kind = "stain" if is_stain else ("scratch" if elong >= SCRATCH_ELONGATION and long_mm >= SCRATCH_MIN_MM else "particle")
                x_mm = (c["x"] + xs.mean() - R.GRID_DIES / 2 * P) * mm
                y_mm = (R.GRID_DIES / 2 * P - c["y"] - ys.mean()) * mm
                found.append({"die": c["die"]["id"], "type": kind, "x_mm": round(float(x_mm), 2), "y_mm": round(float(y_mm), 2),
                              "area_mm2": round(float(area) * mm * mm, 2), "length_mm": round(float(long_mm), 2)})
                c["pixels"] |= comp
    return found


def quadrat_values(crops: dict, found: list[dict], P: int, per_die: int = 2) -> dict:
    """다이를 구역으로 나눈 판정. {(다이 id, 구역 행, 구역 열): 값}. 학습 입력의 light / heavy 와 같은 값이다."""
    out = {}
    for did, c in crops.items():
        q = P // per_die
        pts = [(d["x_mm"], d["y_mm"]) for d in found if d["die"] == did]
        for r in range(per_die):
            for k in range(per_die):
                part = c["pixels"][r * q:(r + 1) * q, k * q:(k + 1) * q]
                if not part.any():
                    continue
                n = _components(part, 1)
                heavy = part.mean() >= HEAVY_QUADRAT_AREA or len(n) >= HEAVY_QUADRAT_COUNT
                out[(did, r, k)] = R.HEAVY_VALUE if heavy else R.LIGHT_VALUE
    return out


def detection_channel(bgr: np.ndarray, meta: dict) -> tuple[np.ndarray, dict]:
    """검출에 쓸 한 장의 영상과 검출 설정 (channel, sharp_z, min_area_mm2).
    밝은 바탕(불투명면)은 흑백 그대로 쓴다. 투명면 너머 바닥이 어둡게 비치면 흑백으로는 결함과 바닥의 차이가
    작고, 표면 흠집·먼지가 하얗게 빛나 결함보다 진해진다. 그때는 색 차이(R−B)를 쓴다.
    붙인 결함은 따뜻한 색, 비친 바닥은 푸른색, 흠집과 반사는 무채색이라 결함만 도드라진다."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    (cx, cy), r = meta["wafer"]["center_px"], meta["wafer"]["radius_px"]
    yy, xx = np.mgrid[0:h, 0:w]
    disc = np.hypot(xx - cx, yy - cy) < r * 0.9
    if float(np.median(gray[disc])) >= DARK_BACKGROUND:
        return gray, {"channel": "gray", "sharp_z": SHARP_Z, "min_area_mm2": 0.0}
    b, _, rr = cv2.split(bgr.astype(np.float32))
    # 색 차이에서 결함은 가장자리(종이)만 따뜻한 색인 고리로 보여 흑백보다 신호가 약하다. 문턱을 한 단계 낮추고
    # 대신 작은 점은 세지 않는다
    return (np.clip(128 + CHROMA_GAIN * (rr - b), 0, 255).astype(np.uint8),
            {"channel": "chroma", "sharp_z": CHROMA_Z, "min_area_mm2": MIN_CHROMA_AREA_MM2})


def detect(map_path: str, meta: dict, seen: np.ndarray | None = None) -> dict:
    img, prof = detection_channel(cv2.imread(map_path), meta)
    return detect_array(img, meta, seen=seen, min_area_mm2=prof["min_area_mm2"], sharp_z=prof["sharp_z"])


def detect_array(gray: np.ndarray, meta: dict, per_die: int = 2, seen: np.ndarray | None = None,
                 min_area_mm2: float = 0.0, sharp_z: float = SHARP_Z) -> dict:
    img, valid, P = rectify(gray, meta, seen)
    return detect_rectified(img, valid, P, per_die, min_area_mm2, sharp_z)


def detect_rectified(img: np.ndarray, valid: np.ndarray, P: int, per_die: int = 2, min_area_mm2: float = 0.0,
                     sharp_z: float = SHARP_Z) -> dict:
    dies = die_layout()
    crops = crop_dies(img, valid, P, dies)
    if len(crops) < 8:
        raise ValueError(f"검사 가능한 다이가 {len(crops)}개뿐입니다. 기준 다이를 만들려면 8개 이상 필요합니다.")
    align(crops, P)
    noise = analyse(crops, P)
    found = extract(crops, P, noise, min_area_mm2, sharp_z)
    counts: dict[int, dict] = {}
    for d in found:
        counts.setdefault(d["die"], {}).setdefault(d["type"], 0)
        counts[d["die"]][d["type"]] += 1
    return {"inspected": sorted(crops), "defects": found, "die_counts": counts, "pitch_px": P,
            "noise": {k: round(v, 5) for k, v in noise.items() if k != "reference"},
            "shift_px": round(float(np.max([np.hypot(*c["shift"]) for c in crops.values()])), 2),
            "quadrats": quadrat_values(crops, found, P, per_die)}


def load_meta(scan_dir: str) -> dict:
    with open(os.path.join(scan_dir, "wafer_map.json")) as f:
        return json.load(f)
