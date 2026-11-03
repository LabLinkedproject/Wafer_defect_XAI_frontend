"""
wafer_stitch.py
다이별 촬영본을 한 장의 웨이퍼 맵으로 합성한다. 사진을 칸에 늘어놓지 않고, 스캔 자체에서 측정한
카메라 모델로 각 픽셀을 웨이퍼 좌표에 투영해 겹치는 부분을 함께 평균한다.

처리 순서
  1. 조명 평탄화   : 카메라에 고정된 조명·주변 감광을 전체 사진의 픽셀별 중앙값으로 구해 나눈다
  2. 카메라 모델   : 이웃 사진 사이의 영상 이동으로 갠트리 mm → 영상 px 변환(회전·축척)을 측정
  3. 왜곡 보정     : 합성 선명도가 가장 높은 렌즈 왜곡 계수를 찾는다 (체커보드 불필요)
  4. 노출 맞춤     : 겹침 영역을 기준으로 사진별 밝기 차이를 맞춘다
  5. 합성          : 영상 중심일수록 큰 가중치로 평균해 왜곡과 감광이 적은 픽셀을 우선한다
  6. 측정          : 합성 결과에서 웨이퍼 원(중심·반지름), 격자 기울기·피치를 측정한다
"""
from __future__ import annotations
import glob
import json
import math
import os
import time

import cv2
import numpy as np

from wafer_model import WAFER_MODEL

WORK_SCALE = 0.5           # 카메라 모델 측정용 축소
SIGMA_RATIO = 0.24         # 합성 가중치 폭 (영상 높이 대비)
MAX_RADIUS_RATIO = 0.62    # 합성에 쓰는 최대 반경 (영상 높이 대비). 바깥은 왜곡·감광이 크다
MIN_IMAGES = 8
WAFER_DIAMETER_MM = WAFER_MODEL["diameter_mm"]


def _flatfield(stack: np.ndarray, sigma: float) -> np.ndarray:
    """장면은 사진마다 움직이고 조명은 카메라에 고정되어 있으므로 중앙값이 조명을 남긴다."""
    return cv2.GaussianBlur(np.median(stack, 0), (0, 0), sigma)


def _edges(img: np.ndarray, win: np.ndarray) -> np.ndarray:
    a = cv2.GaussianBlur(img, (0, 0), 1.5)
    return np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)) * win


def estimate_camera(flat: list[np.ndarray], pos: np.ndarray, grid: list[tuple], prior: np.ndarray | None = None) -> dict:
    """이웃 사진 쌍의 영상 이동으로 mm→px 행렬을 구한다.
    다이 격자가 반복되므로 한 칸 어긋난 쌍이 섞인다. 큰 이동을 일관되게 내는 쌍만 쓴다.
    prior(이전 스캔의 행렬)를 주면 예상 이동만큼 먼저 옮긴 뒤 남은 차이만 잰다.
    스캔 간격과 격자 피치가 같고 반사 무늬가 없는 면에서는 이렇게 해야 한 칸 이동을 0 으로 읽지 않는다."""
    h, w = flat[0].shape
    yy, xx = np.mgrid[0:h, 0:w]
    win = cv2.GaussianBlur((np.hypot(xx - w / 2, yy - h / 2) < h * 0.46).astype(np.float32), (0, 0), 15)
    hann = cv2.createHanningWindow((w, h), cv2.CV_64F)
    edges = [_edges(f, win).astype(np.float64) for f in flat]
    samples = []
    for i, j in grid:
        d = pos[j] - pos[i]
        if prior is None:
            (sx, sy), resp = cv2.phaseCorrelate(edges[i], edges[j], hann)
        else:
            ex, ey = prior @ d
            back = cv2.warpAffine(edges[j], np.float64([[1, 0, -ex], [0, 1, -ey]]), (w, h))
            (rx, ry), resp = cv2.phaseCorrelate(edges[i], back, hann)
            sx, sy = ex + rx, ey + ry
        samples.append((d[0], d[1], sx, sy, resp))
    s = np.array(samples)
    if len(s) < 4:
        raise ValueError("이웃 사진 쌍이 부족합니다")

    def axis_vector(mask):
        """정합 응답이 강한 쌍만 쓴다. 격자에 속은 쌍은 응답이 약하거나 이동량이 작다."""
        rows = s[mask]
        mm = np.abs(rows[:, 0:2]).sum(1)
        v = rows[:, 2:4] * np.sign(rows[:, 0:2].sum(1, keepdims=True)) / np.maximum(mm[:, None], 1e-6)
        n = np.hypot(*v.T)
        order = np.argsort(-rows[:, 4])
        keep = [i for i in order if n[i] > 1.5][:max(3, len(order) // 5)]
        return np.median(v[keep], 0)

    is_x = np.abs(s[:, 0]) > np.abs(s[:, 1])
    if is_x.sum() < 2 or (~is_x).sum() < 2:
        raise ValueError("X, Y 양쪽으로 이동한 사진이 필요합니다")
    ax, ay = axis_vector(is_x), axis_vector(~is_x)
    # 두 축은 직교하고 축척이 같아야 한다. 측정값을 그 조건으로 다듬는다.
    scale = (np.hypot(*ax) + np.hypot(*ay)) / 2
    theta = math.atan2(ax[1], ax[0])
    sign = 1.0 if ax[0] * ay[1] - ax[1] * ay[0] > 0 else -1.0
    A = scale * np.array([[math.cos(theta), -sign * math.sin(theta)],
                          [math.sin(theta), sign * math.cos(theta)]])
    return {"A": A, "scale": float(scale), "theta_deg": math.degrees(theta), "mirrored": sign < 0}


def _camera_matrix(scale: float, theta_deg: float, mirrored: bool) -> np.ndarray:
    t, sign = math.radians(theta_deg), -1.0 if mirrored else 1.0
    return scale * np.array([[math.cos(t), -sign * math.sin(t)], [math.sin(t), sign * math.cos(t)]])


def camera_prior(scan_root: str, exclude: str) -> np.ndarray | None:
    """가장 최근에 합성에 성공한 다른 스캔의 카메라 행렬 (축소 영상 기준). 카메라 장착이 같다고 본다."""
    for d in sorted(glob.glob(os.path.join(scan_root, "*_scan")), reverse=True):
        meta = os.path.join(d, "wafer_map.json")
        if os.path.abspath(d) == os.path.abspath(exclude) or not os.path.isfile(meta):
            continue
        with open(meta) as f:
            cam = json.load(f).get("camera")
        if cam:
            return _camera_matrix(cam["px_per_mm_full"] * WORK_SCALE, cam["rotation_deg"], cam["mirrored"])
    return None


def _agrees(cam: dict, prior: np.ndarray) -> bool:
    """직접 측정이 이전 카메라와 축척 20 %, 회전 10° 안에서 맞는지"""
    if not np.isfinite(cam["A"]).all():
        return False
    p_scale = float(np.linalg.norm(prior[:, 0]))
    p_theta = math.degrees(math.atan2(prior[1, 0], prior[0, 0]))
    turn = abs((cam["theta_deg"] - p_theta + 180) % 360 - 180)
    return abs(cam["scale"] / p_scale - 1) < 0.2 and turn < 10


def _distort(x, y, q):
    """왜곡 없는 좌표 → 영상 좌표. 모델: r_u = r_d (1 + q r_d²), 고정점 반복으로 푼다."""
    if not q:
        return x, y
    r2 = x * x + y * y
    f = np.ones_like(r2)
    for _ in range(4):
        f = 1 / (1 + q * r2 * f * f)
    return x * f, y * f


def _undistort(x, y, q):
    f = 1 + q * (x * x + y * y)
    return x * f, y * f


def _project(shape_out, s, c_out, B, C, u0, v0, q):
    """모자이크 격자 → 원본 영상 좌표."""
    gy, gx = np.mgrid[0:shape_out[0], 0:shape_out[1]].astype(np.float32)
    px, py = (gx - c_out[0]) / s, -(gy - c_out[1]) / s
    x = B[0, 0] * (px - C[0]) + B[0, 1] * (py - C[1])
    y = B[1, 0] * (px - C[0]) + B[1, 1] * (py - C[1])
    x, y = _distort(x, y, q)
    return (x + u0).astype(np.float32), (y + v0).astype(np.float32)


def _compose(images, flat_ref, pos, A, s, size, k, gains=None, sigma=None, color=False):
    """가중 평균 합성. 반환 (모자이크, 누적 가중치)"""
    h, w = images[0].shape[:2]
    u0, v0 = w / 2, h / 2
    B = -A
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sigma = sigma or SIGMA_RATIO * h
    rr2 = (xx - u0) ** 2 + (yy - v0) ** 2
    w0 = np.exp(-rr2 / (2 * sigma ** 2)).astype(np.float32)
    # 조명이 약해 잡음이 큰 영역과 렌즈 외곽은 쓰지 않는다
    w0[(flat_ref < flat_ref.max() * 0.22) | (rr2 > (MAX_RADIUS_RATIO * h) ** 2)] = 0
    H, W = size
    c = (W / 2, H / 2)
    num = np.zeros((H, W, 3) if color else (H, W), np.float32)
    den = np.zeros((H, W), np.float32)
    ext = max(w, h) * 0.75 / np.hypot(*A[:, 0])     # 사진 하나가 닿는 범위 (mm)
    for i, img in enumerate(images):
        C = pos[i]
        # 사진이 닿는 영역만 계산
        rx = int(ext * s)
        cx, cy = int(c[0] + C[0] * s), int(c[1] - C[1] * s)
        x0, x1 = max(0, cx - rx), min(W, cx + rx)
        y0, y1 = max(0, cy - rx), min(H, cy + rx)
        if x1 <= x0 or y1 <= y0:
            continue
        mx, my = _project((y1 - y0, x1 - x0), s, (c[0] - x0, c[1] - y0), B, C, u0, v0, k)
        a = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0)
        wt = cv2.remap(w0, mx, my, cv2.INTER_LINEAR, borderValue=0)
        g = 1.0 if gains is None else gains[i]
        if color:
            num[y0:y1, x0:x1] += a * g * wt[..., None]
        else:
            num[y0:y1, x0:x1] += a * g * wt
        den[y0:y1, x0:x1] += wt
    d = np.maximum(den, 1e-5)
    return (num / d[..., None] if color else num / d), den


def _estimate_gains(images, flat_ref, pos, A, s, size, k):
    """사진별 밝기 맞춤: 모자이크 대비 비율로 두 번 갱신한다 (자동 노출 차이 보정)."""
    n = len(images)
    gains = np.ones(n, np.float32)
    h, w = images[0].shape
    u0, v0 = w / 2, h / 2
    B = -A
    H, W = size
    c = (W / 2, H / 2)
    for _ in range(2):
        mos, den = _compose(images, flat_ref, pos, A, s, size, k, gains)
        for i, img in enumerate(images):
            # 영상 중심부 픽셀 → 모자이크 좌표 (중심부는 왜곡이 작아 무시한다)
            ys, xs = np.mgrid[int(h * 0.3):int(h * 0.7):6, int(w * 0.3):int(w * 0.7):6]
            d = np.stack([xs - u0, ys - v0], -1).reshape(-1, 2).astype(np.float64)
            p = (np.linalg.inv(B) @ d.T).T + pos[i]
            mxs = (c[0] + p[:, 0] * s).astype(int)
            mys = (c[1] - p[:, 1] * s).astype(int)
            ok = (mxs >= 0) & (mxs < W) & (mys >= 0) & (mys < H)
            if ok.sum() < 20:
                continue
            ref = mos[mys[ok], mxs[ok]]
            cur = img[ys.ravel()[ok], xs.ravel()[ok]] * gains[i]
            r = ref[cur > 1e-3] / cur[cur > 1e-3]
            if len(r):
                gains[i] *= float(np.clip(np.median(r), 0.7, 1.4))
        gains /= np.median(gains)
    return gains


def _highpass(img: np.ndarray) -> np.ndarray:
    a = cv2.GaussianBlur(img, (0, 0), 1.2) - cv2.GaussianBlur(img, (0, 0), 9)
    return np.clip(a, -0.25, 0.25)


def refine_camera(flat, light, pos, A0, pairs) -> dict:
    """사진 쌍을 서로의 좌표로 투영해 겹침 일치도(NCC)가 최대인 축척·회전·왜곡을 찾는다.
    멀리 떨어진 쌍일수록 영상 외곽끼리 겹치므로 왜곡에 민감하다. 체커보드가 필요 없다."""
    h, w = flat[0].shape
    u0, v0 = w / 2, h / 2
    hp = [_highpass(f) for f in flat]
    yy, xx = np.mgrid[0:h:2, 0:w:2].astype(np.float32)
    dx, dy = xx - u0, yy - v0
    rr = np.hypot(dx, dy)
    use = (cv2.resize(light, (xx.shape[1], xx.shape[0])) > light.max() * 0.15) & (rr < 0.95 * h)

    def score(f, theta, q):
        c, sn = math.cos(theta), math.sin(theta)
        R = np.array([[c, -sn], [sn, c]])
        A = f * R @ A0
        px, py = _undistort(dx, dy, q)
        tot, n = 0.0, 0
        for i, j in pairs:
            sh = A @ (pos[j] - pos[i])
            xd, yd = _distort(px + sh[0], py + sh[1], q)
            mx, my = (xd + u0).astype(np.float32), (yd + v0).astype(np.float32)
            inside = (mx > 4) & (mx < w - 4) & (my > 4) & (my < h - 4) & use
            if inside.sum() < 1500:
                continue
            aimg = hp[i][yy.astype(int), xx.astype(int)]
            bimg = cv2.remap(hp[j], mx, my, cv2.INTER_LINEAR)
            a, b = aimg[inside], bimg[inside]
            a, b = a - a.mean(), b - b.mean()
            den = float(np.sqrt((a * a).sum() * (b * b).sum()))
            if den > 1e-9:
                tot += float((a * b).sum()) / den
                n += 1
        return tot / max(n, 1)

    # 축척과 왜곡은 서로 보상하는 관계라 축별 탐색은 지역 최적에 갇힌다.
    # 먼저 일부 쌍으로 두 값을 격자로 함께 훑어 시작점을 잡는다.
    qmax = 1.0 / (h * h)
    all_pairs = pairs
    pairs = all_pairs[::max(1, len(all_pairs) // 12)]
    grid = [(score(ff, 0.0, qq), ff, qq) for ff in np.arange(0.92, 1.1001, 0.02)
            for qq in np.arange(0.0, 0.7001, 0.1) / (h * h)]
    pairs = all_pairs
    f, th, q = 1.0, 0.0, 0.0
    base = score(f, th, q)
    _, f, q = max(grid)
    best = score(f, th, q)
    for step_f, step_t, step_q in ((0.012, 0.01, 0.06 * qmax), (0.005, 0.004, 0.025 * qmax), (0.002, 0.0015, 0.01 * qmax)):
        improved = True
        while improved:
            improved = False
            for df, dt, dq in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                               (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1)):
                nf, nt, nq = f + df * step_f, th + dt * step_t, min(qmax, max(0.0, q + dq * step_q))
                v = score(nf, nt, nq)
                if v > best + 1e-4:
                    best, f, th, q = v, nf, nt, nq
                    improved = True
    c, sn = math.cos(th), math.sin(th)
    A = f * np.array([[c, -sn], [sn, c]]) @ A0
    return {"A": A, "q": q, "ncc_before": round(base, 4), "ncc_after": round(best, 4)}


def fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    A = np.c_[2 * points, np.ones(len(points))]
    b = (points ** 2).sum(1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(sol[0]), float(sol[1]), float(math.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 0)))


def detect_wafer(mos: np.ndarray, den: np.ndarray, s: float) -> dict | None:
    """모자이크에서 웨이퍼 외곽원을 찾는다. 곧게 뻗은 격자선은 원에 맞지 않으므로 RANSAC 로 걸러진다."""
    img = cv2.GaussianBlur(mos, (0, 0), 3)
    valid = cv2.erode((den > 0.05).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    p5, p95 = np.percentile(img[valid], (2, 98))
    u8 = np.clip((img - p5) / max(p95 - p5, 1e-6) * 255, 0, 255).astype(np.uint8)
    edges = cv2.Canny(u8, 18, 45)
    edges[~valid] = 0
    # 곧은 선(격자·테이프)은 제거한다
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=70, minLineLength=int(50 * s / 4.8) , maxLineGap=6)
    if lines is not None:
        for l in lines[:, 0]:
            cv2.line(edges, (l[0], l[1]), (l[2], l[3]), 0, 5)
    ys, xs = np.nonzero(edges)
    if len(xs) < 80:
        return None
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
    pts = np.c_[xs, ys].astype(np.float64)
    if len(pts) > 6000:
        pts = pts[np.random.default_rng(1).choice(len(pts), 6000, replace=False)]
    rng = np.random.default_rng(3)
    H = mos.shape[0]
    best, best_n = None, 0
    for _ in range(4000):
        idx = rng.choice(len(pts), 3, replace=False)
        p = pts[idx]
        if min(np.hypot(*(p[0] - p[1])), np.hypot(*(p[1] - p[2])), np.hypot(*(p[0] - p[2]))) < 60:
            continue
        try:
            cx, cy, r = fit_circle(p)
        except np.linalg.LinAlgError:
            continue
        if not 0.3 * H < r < 1.0 * H:
            continue
        d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        near = pts[d < 2.5]
        if len(near) <= best_n:
            continue
        # 웨이퍼는 바닥보다 어둡다: 바깥쪽으로 갈수록 밝아지는 경계만 센다
        ix, iy = near[:, 0].astype(int), near[:, 1].astype(int)
        rad = gx[iy, ix] * (near[:, 0] - cx) + gy[iy, ix] * (near[:, 1] - cy)
        n = int((rad > 0).sum())
        if n > best_n:
            best, best_n = (cx, cy, r), n
    if best is None or best_n < 60:
        return None
    cx, cy, r = best
    for _ in range(5):
        d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        cx, cy, r = fit_circle(pts[d < 3])
    d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
    inl = pts[d < 3]
    ang = np.degrees(np.arctan2(inl[:, 1] - cy, inl[:, 0] - cx))
    arc = len(np.unique((ang // 5).astype(int))) * 5
    return {"cx": float(cx), "cy": float(cy), "r": float(r), "residual_px": round(float(np.median(d[d < 3])), 2),
            "points": int(len(inl)), "arc_deg": int(arc)}


def measure_grid(mos: np.ndarray, den: np.ndarray, wafer: dict | None) -> dict | None:
    """격자 기울기와 피치. 회전시키며 행·열 투영의 뾰족함이 최대인 각을 찾는다.
    테이프처럼 굵고 진한 구조가 투영을 지배하지 않도록 웨이퍼 원 안쪽만 쓴다."""
    H, W = mos.shape
    m = den > 0.3
    if wafer:
        yy, xx = np.mgrid[0:H, 0:W]
        m &= np.hypot(xx - wafer["cx"], yy - wafer["cy"]) < wafer["r"] * 0.95
    if m.sum() < 1000:
        return None
    img = mos.copy()
    img[~m] = float(np.median(img[m]))
    hp = cv2.GaussianBlur(img, (0, 0), 1) - cv2.GaussianBlur(img, (0, 0), 8)
    hp[~cv2.erode(m.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)] = 0
    hp = np.clip(hp, -0.2, 0)              # 격자선은 어두운 선
    best = (0.0, -1.0)
    center = (wafer["cx"], wafer["cy"]) if wafer else (W / 2, H / 2)
    for a in np.arange(-6, 6.01, 0.25):
        M = cv2.getRotationMatrix2D(center, float(a), 1)
        r = cv2.warpAffine(hp, M, (W, H))
        sc = float(r.sum(0).var() + r.sum(1).var())
        if sc > best[1]:
            best = (float(a), sc)
    angle = best[0]
    r = cv2.warpAffine(hp, cv2.getRotationMatrix2D(center, angle, 1), (W, H))
    out, lines_by = {}, {}
    for name, prof in (("x", r.sum(0)), ("y", r.sum(1))):
        # 격자선은 투영의 뾰족한 골로 나타난다. 골 사이 간격의 중앙값이 피치다.
        p = cv2.GaussianBlur(prof.reshape(1, -1).astype(np.float32), (0, 0), 2).ravel()
        th = p.min() * 0.35
        lines, i = [], 0
        while i < len(p):
            if p[i] < th:
                j = i
                while j < len(p) and p[j] < th:
                    j += 1
                lines.append(i + int(np.argmin(p[i:j])))
                i = j
            else:
                i += 1
        lines_by[name] = lines
        gaps = np.diff(lines)
        gaps = gaps[gaps > H * 0.04]
        if len(gaps) >= 2:
            out[name] = float(np.median(gaps))
    if not out:
        return None
    if "x" in lines_by and wafer:
        # 웨이퍼 중심 기준 격자선 위상. 0 이면 중심이 격자선 위(짝수 격자)
        out_phase = {n: float(min(((v - c) % out[n]) for v in lines_by[n]) / out[n])
                     for n, c in (("x", wafer["cx"]), ("y", wafer["cy"])) if n in out}
    else:
        out_phase = {}
    return {"angle_deg": angle, "pitch_px": float(np.mean(list(out.values()))), "phase": out_phase,
            "pitch_x_px": out.get("x"), "pitch_y_px": out.get("y")}


def build(scan_dir: str, out_dir: str | None = None, out_scale: float = 1.0, log=print) -> dict:
    """스캔 폴더(manifest.json + 사진)로 웨이퍼 맵을 만든다."""
    t0 = time.time()
    out_dir = out_dir or scan_dir
    with open(os.path.join(scan_dir, "manifest.json")) as f:
        man = json.load(f)
    dies = man["dies"]
    if len(dies) < MIN_IMAGES:
        raise ValueError(f"사진이 {len(dies)}장뿐입니다. 합성에는 최소 {MIN_IMAGES}장이 필요합니다.")
    root = os.path.dirname(scan_dir.rstrip("/"))
    pos = np.array([[d["pos"]["x"], d["pos"]["y"]] for d in dies], float)
    rc = {(d["row"], d["col"]): i for i, d in enumerate(dies)}

    log("사진 불러오기")
    bgr = [cv2.imread(os.path.join(root, d["file"])) for d in dies]
    if any(b is None for b in bgr):
        raise ValueError("사진을 읽지 못했습니다")
    small = [cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None, fx=WORK_SCALE, fy=WORK_SCALE,
                        interpolation=cv2.INTER_AREA).astype(np.float32) for b in bgr]

    log("조명 평탄화")
    light = _flatfield(np.stack(small), 10)
    floor = max(light.max() * 0.05, 6)
    flat = [s / np.maximum(light, floor) for s in small]

    log("카메라 축척·방향 측정")
    pairs = [(i, rc[(r + dr, c + dc)]) for (r, c), i in rc.items() for dr, dc in ((0, 1), (1, 0))
             if (r + dr, c + dc) in rc]
    prior = camera_prior(root, scan_dir)
    try:
        cam = estimate_camera(flat, pos, pairs)
    except ValueError:
        if prior is None:
            raise
        cam = {"A": np.full((2, 2), np.nan)}
    if prior is not None and not _agrees(cam, prior):
        log("격자 반복으로 이동량이 모호해 이전 카메라 값을 기준으로 다시 측정")
        cam = estimate_camera(flat, pos, pairs, prior)
    if not np.isfinite(cam["A"]).all():
        raise ValueError("사진 사이 이동량을 측정하지 못했습니다. 사진이 실제로 다른 위치에서 찍혔는지 확인하세요.")
    A = cam["A"]
    scale_mm = cam["scale"]                  # 축소 영상 기준 px/mm
    s = scale_mm * 0.985                     # 모자이크 px/mm (측정용)
    ext = float(np.ptp(pos, 0).max()) / 2 + flat[0].shape[0] * 0.75 / scale_mm + 12
    cpos = pos - (pos.max(0) + pos.min(0)) / 2
    size = (int(2 * ext * s), int(2 * ext * s))
    log(f"축척 {scale_mm / WORK_SCALE:.2f} px/mm, 회전 {cam['theta_deg']:.1f}°")

    log("렌즈 왜곡·축척 정밀 측정")
    # 정합은 다양한 거리의 쌍으로 재야 왜곡을 구분할 수 있다
    order = sorted(range(len(pos)), key=lambda i: tuple(pos[i]))
    far = [(a_, b_) for a_ in order[::3] for b_ in order[::3] if a_ < b_ and 20 < np.hypot(*(pos[a_] - pos[b_])) < 95]
    fine = refine_camera(flat, light, pos, A, far[:40])
    A, k = fine["A"], fine["q"]
    scale_mm = float(np.linalg.norm(A[:, 0]))
    s = scale_mm * 0.985
    ext = float(np.ptp(pos, 0).max()) / 2 + flat[0].shape[0] * MAX_RADIUS_RATIO / scale_mm + 12
    size = (int(2 * ext * s), int(2 * ext * s))
    log(f"축척 {scale_mm / WORK_SCALE:.2f} px/mm, 왜곡 {k * flat[0].shape[0] ** 2:.3f}, 일치도 {fine['ncc_before']} → {fine['ncc_after']}")

    log("밝기 맞춤")
    gains = _estimate_gains(flat, light, cpos, A, s, size, k)

    log("합성")
    mos_g, den_g = _compose(flat, light, cpos, A, s, size, k, gains)

    log("웨이퍼·격자 측정")
    wafer = detect_wafer(mos_g, den_g, s)
    grid = measure_grid(mos_g, den_g, wafer if wafer and wafer["arc_deg"] >= 45 else None)

    # 컬러 출력 (out_scale=1 이면 원본의 절반 해상도 기준)
    S = s * out_scale
    SZ = (int(size[0] * out_scale), int(size[1] * out_scale))
    if out_scale >= 1.5:
        col_scale = 1.0
        kc = k * (WORK_SCALE ** 2)         # 해상도가 2배면 q 는 1/4 배
        Ac = A / WORK_SCALE
        imgs = [b.astype(np.float32) for b in bgr]
        lc = cv2.resize(light, (bgr[0].shape[1], bgr[0].shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        col_scale = WORK_SCALE
        kc, Ac = k, A
        imgs = [cv2.resize(b, None, fx=col_scale, fy=col_scale, interpolation=cv2.INTER_AREA).astype(np.float32) for b in bgr]
        lc = light
    white = np.maximum(lc, floor)
    imgs = [x / white[..., None] for x in imgs]
    rgb, den = _compose(imgs, lc, cpos, Ac, S, SZ, kc, gains, color=True)
    mask = den > 0.02
    # 조명 색 보정: 채널별 밝기를 맞춰 LED 색이 웨이퍼 색으로 보이지 않게 한다
    med = np.median(rgb[mask], 0)
    rgb = rgb * (med.mean() / np.maximum(med, 1e-6))
    ref = rgb[mask]
    lo, hi = np.percentile(ref, 0.5), np.percentile(ref, 99.7)
    out = np.clip((rgb - lo) / max(hi - lo, 1e-6) * 245, 0, 255).astype(np.uint8)
    out[~mask] = (18, 18, 20)
    name = "wafer_map.jpg"
    cv2.imwrite(os.path.join(out_dir, name), out, [cv2.IMWRITE_JPEG_QUALITY, 90])

    center_g = (pos.max(0) + pos.min(0)) / 2
    result = {
        "images": len(dies), "map": name, "size_px": [SZ[1], SZ[0]],
        "camera": {"px_per_mm_full": round(scale_mm / WORK_SCALE, 3), "rotation_deg": round(cam["theta_deg"], 2),
                   "mirrored": cam["mirrored"], "fov_mm": [round(bgr[0].shape[1] / (scale_mm / WORK_SCALE), 1),
                                                          round(bgr[0].shape[0] / (scale_mm / WORK_SCALE), 1)]},
        "distortion": {"q_normalized": round(k * flat[0].shape[0] ** 2, 4)},
        "overlap_ncc": {"before": fine["ncc_before"], "after": fine["ncc_after"]},
        "exposure_gain_range": [round(float(gains.min()), 3), round(float(gains.max()), 3)],
        "px_per_gantry_mm": round(S, 4),
        "seconds": round(time.time() - t0, 1),
        "gantry_center_of_map": [round(float(center_g[0]), 2), round(float(center_g[1]), 2)],
    }
    if wafer and wafer["arc_deg"] < 45:
        wafer = None                         # 호가 너무 짧으면 원을 믿을 수 없다
    if wafer:
        r_mm = wafer["r"] / s
        yy, xx = np.mgrid[0:SZ[0], 0:SZ[1]].astype(np.float32)
        disc = np.hypot(xx - wafer["cx"] * out_scale, yy - wafer["cy"] * out_scale) < wafer["r"] * out_scale
        total = math.pi * (wafer["r"] * out_scale) ** 2
        result["coverage_pct"] = round(float((disc & mask).sum()) / total * 100, 1)
        cgx = (wafer["cx"] - size[1] / 2) / s + center_g[0]
        cgy = -(wafer["cy"] - size[0] / 2) / s + center_g[1]
        result["wafer"] = {"center_px": [round(wafer["cx"] * out_scale, 1), round(wafer["cy"] * out_scale, 1)],
                           "radius_px": round(wafer["r"] * out_scale, 1), "radius_gantry_mm": round(r_mm, 2),
                           "center_gantry_mm": [round(cgx, 2), round(cgy, 2)],
                           "fit_residual_px": wafer["residual_px"], "arc_deg": wafer["arc_deg"],
                           "gantry_mm_per_true_mm": round(r_mm / (WAFER_DIAMETER_MM / 2), 4)}
    if grid:
        ratio = (WAFER_DIAMETER_MM / 2) / wafer["r"] if wafer else None
        result["grid"] = {"angle_deg": grid["angle_deg"], "pitch_px": round(grid["pitch_px"] * out_scale, 1),
                          "phase": grid["phase"],
                          "pitch_gantry_mm": round(grid["pitch_px"] / s, 2),
                          "pitch_mm": round(grid["pitch_px"] * ratio, 2) if ratio else None}
    result = json.loads(json.dumps(result, default=float))      # numpy 값을 일반 수로
    with open(os.path.join(out_dir, "wafer_map.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def latest_map(scan_root: str = os.path.join("captures", "scans")) -> dict | None:
    """가장 최근에 합성된 웨이퍼 맵 정보."""
    for d in sorted(glob.glob(os.path.join(scan_root, "*_scan")), reverse=True):
        meta = os.path.join(d, "wafer_map.json")
        if os.path.isfile(meta):
            with open(meta) as f:
                m = json.load(f)
            m["dir"] = os.path.basename(d)
            m["url"] = f"/scans/{m['dir']}/{m['map']}"
            return m
    return None


if __name__ == "__main__":
    import sys
    print(json.dumps(build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None), ensure_ascii=False, indent=2))
