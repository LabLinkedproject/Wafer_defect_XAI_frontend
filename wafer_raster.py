"""
wafer_raster.py
실물 웨이퍼 기준의 분류기 입력 격자. 학습 데이터 가공과 실시간 추론이 같은 정의를 쓴다.

다이 한 칸을 6셀로 나눈 60×60 격자다 (셀 = 32/6 = 5.33 mm, 전체 320 mm).
- 격자 중심이 웨이퍼 중심이고, 스크라이브 라인이 셀 경계와 일치한다.
- 셀 크기가 공개 데이터의 다이(270/52 = 5.2 mm)와 거의 같아 재표본화 손실이 작다.

입력은 3채널이다.
  0: 불량 값 (0~1)
  1: 관측 영역 (전면 맵이면 웨이퍼 전체, 다이 단위면 검사 다이 44개)
  2: 중심 거리 r/R (전역 평균 풀링을 해도 위치 정보가 남도록)
"""
from __future__ import annotations

import numpy as np

from wafer_model import WAFER_MODEL, die_layout

CELLS_PER_DIE = 6
GRID_DIES = 10                        # 격자 한 변의 다이 칸 수 (웨이퍼 8칸 + 여유 1칸씩)
SIZE = CELLS_PER_DIE * GRID_DIES
CELL_MM = WAFER_MODEL["pitch_mm"] / CELLS_PER_DIE
RADIUS_CELLS = WAFER_MODEL["diameter_mm"] / 2 / CELL_MM
LIGHT_VALUE = 0.6                     # scoring.die_weight 의 light / heavy 값과 같다
HEAVY_VALUE = 1.0


def wafer_mask(spec=WAFER_MODEL, samples: int = 8) -> np.ndarray:
    """셀별 웨이퍼 점유율. 6시 방향 V 노치까지 도안대로 뺀다."""
    n = SIZE * samples
    u = (np.arange(n) + 0.5) / samples * CELL_MM - SIZE / 2 * CELL_MM
    xx, yy = np.meshgrid(u, -u)
    r = spec["diameter_mm"] / 2
    inside = xx ** 2 + yy ** 2 <= r ** 2
    depth, half = spec["notch_depth_mm"], spec["notch_width_mm"] / 2
    notch = (yy <= -r + depth * (1 - np.abs(xx) / half)) & (np.abs(xx) <= half)
    return (inside & ~notch).reshape(SIZE, samples, SIZE, samples).mean((1, 3)).astype(np.float32)


def radius_map() -> np.ndarray:
    c = (SIZE - 1) / 2
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    return np.hypot(xx - c, yy - c) / RADIUS_CELLS


def die_slot(d: dict) -> tuple[int, int]:
    """다이의 다이 칸 좌표 (격자 여유 칸 1을 더한 행, 열)."""
    return d["row"] + 1, d["col"] + 1


def die_mask(dies=None) -> np.ndarray:
    """검사 다이 44개가 차지하는 셀."""
    m = np.zeros((GRID_DIES, GRID_DIES), np.float32)
    for d in dies or die_layout():
        m[die_slot(d)] = 1.0
    return m.repeat(CELLS_PER_DIE, 0).repeat(CELLS_PER_DIE, 1)


def from_dies(dies: list[dict], values: dict[int, float]) -> np.ndarray:
    """다이 단위 판정 → 입력 3채널. values: {다이 id: 0~1}"""
    v = np.zeros((GRID_DIES, GRID_DIES), np.float32)
    by_id = {d["id"]: d for d in dies}
    for i, val in values.items():
        v[die_slot(by_id[i])] = val
    fill = v.repeat(CELLS_PER_DIE, 0).repeat(CELLS_PER_DIE, 1)
    return np.stack([fill, die_mask(dies), radius_map()])


def from_points(points_mm, spec=WAFER_MODEL) -> np.ndarray:
    """결함 좌표(웨이퍼 중심 기준 mm, 위쪽 +y) → 입력 3채널. 합성 맵에서 좌표를 얻을 때 쓴다."""
    fill = np.zeros((SIZE, SIZE), np.float32)
    mask = wafer_mask(spec)
    for x, y in points_mm:
        c, r = int(x / CELL_MM + SIZE / 2), int(SIZE / 2 - y / CELL_MM)
        if 0 <= r < SIZE and 0 <= c < SIZE:
            fill[r, c] = 1.0
    return np.stack([fill * (mask > 0), mask, radius_map()])
