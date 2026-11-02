"""
wafer_model.py
실물 웨이퍼 모형의 규격. 기준은 직접 제작한 도안(reports/wafer_model, DXF 생성 코드)이다.

은경 밀러 아크릴 2 mm, 지름 270 mm, 다이 32 mm, 6시 방향 노치(깊이 1.5 mm, 폭 2 mm).
스크라이브 라인이 웨이퍼 중심을 지나는 짝수 격자다.

검사 대상은 칸 면적의 99 % 이상이 웨이퍼 안에 있는 다이 44개다.
    행별 다이 수 (위 → 아래): 2, 6, 6, 8, 8, 6, 6, 2  = 44
온전한 다이는 40개이고, 나머지 4개는 바깥 모서리만 0.76 mm 잘린 다이(면적 99.9 %)다.
그 밖의 가장자리 칸은 면적 91 % 이하의 부분 다이라 제외한다.
"""
from __future__ import annotations
import math

import numpy as np

WAFER_MODEL = {
    "name":           "270 mm / 44-die model",
    "material":       "은경 밀러 아크릴 2 mm",
    "diameter_mm":    270.0,
    "pitch_mm":       32.0,                # 스크라이브 라인 간격 = 다이 크기
    "notch_depth_mm": 1.5,                 # 6시 방향 V 노치
    "notch_width_mm": 2.0,
    "notch_clock":    6,
    "min_die_area":   0.99,                # 다이 칸 면적 중 웨이퍼 안에 든 비율이 이 이상이면 검사 대상
    "grid":           8,                   # 짝수 격자 한 변 칸 수
    "expected_dies":  44,
}


def active_radius(spec=WAFER_MODEL) -> float:
    return spec["diameter_mm"] / 2


def die_area_fraction(cx: float, cy: float, spec=WAFER_MODEL, samples: int = 200) -> float:
    """다이 칸 면적 중 웨이퍼 원 안에 든 비율."""
    p = spec["pitch_mm"]
    u = (np.arange(samples) + 0.5) / samples - 0.5
    xx, yy = np.meshgrid(cx + u * p, cy + u * p)
    return float((xx ** 2 + yy ** 2 <= active_radius(spec) ** 2).mean())


def die_layout(spec=WAFER_MODEL) -> list[dict]:
    """다이 목록. 좌표는 웨이퍼 중심 기준 mm (오른쪽 +x, 위쪽 +y).
    row 0 = 맨 위, col 0 = 맨 왼쪽. id 는 위→아래, 왼→오른 순서로 1부터."""
    n, p = spec["grid"], spec["pitch_mm"]
    half = n / 2
    dies = []
    for row in range(n):
        cy = (half - row - 0.5) * p
        for col in range(n):
            cx = (col - half + 0.5) * p
            if die_area_fraction(cx, cy, spec) >= spec["min_die_area"]:
                dies.append({"row": row, "col": col, "x_mm": round(cx, 3), "y_mm": round(cy, 3)})
    for i, d in enumerate(dies, 1):
        d["id"] = i
    return dies


def row_counts(dies) -> list[int]:
    counts: dict[int, int] = {}
    for d in dies:
        counts[d["row"]] = counts.get(d["row"], 0) + 1
    return [counts[k] for k in sorted(counts)]


def scan_order(dies) -> list[dict]:
    """지그재그(서펜타인) 순서: 위 행은 왼→오, 다음 행은 오→왼. 갠트리 이동 거리를 최소화."""
    by_row: dict[int, list[dict]] = {}
    for d in dies:
        by_row.setdefault(d["row"], []).append(d)
    order = []
    for k, row in enumerate(sorted(by_row)):
        cells = sorted(by_row[row], key=lambda d: d["col"], reverse=(k % 2 == 1))
        order.extend(cells)
    return order


def summary(spec=WAFER_MODEL) -> dict:
    dies = die_layout(spec)
    xs = [d["x_mm"] for d in dies]
    ys = [d["y_mm"] for d in dies]
    return {
        **spec,
        "active_radius_mm": round(active_radius(spec), 2),
        "die_count": len(dies),
        "row_counts": row_counts(dies),
        "count_ok": len(dies) == spec["expected_dies"],
        "extent_mm": {"x": [min(xs), max(xs)], "y": [min(ys), max(ys)]},
    }


if __name__ == "__main__":
    s = summary()
    print(f"{s['name']}: 지름 {s['diameter_mm']:.1f} mm, 반경 {s['active_radius_mm']} mm, 피치 {s['pitch_mm']} mm")
    print(f"다이 {s['die_count']}개 (기대 {s['expected_dies']}) 행별 {s['row_counts']}")
    print(f"다이 중심 범위 X {s['extent_mm']['x']}  Y {s['extent_mm']['y']}")
