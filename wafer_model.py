"""
wafer_model.py
실물 스캔용 웨이퍼 모델 규격: 8.5인치, 엣지 제외 44다이.

다이 배치는 8×8 짝수 격자에서 '다이 네 모서리가 모두 유효 반경 안에 드는 칸'만 남긴 것이다.
    행별 다이 수 (위 → 아래): 2, 6, 6, 8, 8, 6, 6, 2  = 44
중앙 다이가 있는 홀수 격자로는 상하좌우 대칭 44개가 나올 수 없으므로(1 + 4k 꼴) 짝수 격자가 유일한 대칭 배치다.

⚠ PITCH_MM 은 실측 전 가정값이다. 모형의 다이 한 칸을 자로 재서 바꾸면 스캔 좌표가 모두 따라 바뀐다.
   유효 반경 / 피치 가 4.243 ~ 4.472 범위면 44개가 유지되고, 벗어나면 다이 수가 달라져 시작 시 경고한다.
"""
from __future__ import annotations
import math

INCH_MM = 25.4

WAFER_MODEL = {
    "name":           "8.5inch / 44-die model",
    "diameter_mm":    8.5 * INCH_MM,       # 215.9
    "edge_excl_mm":   3.0,                 # 가장자리 제외 폭 (가정)
    "pitch_mm":       24.0,                # 다이 피치 (가정, 실측 후 수정)
    "grid":           8,                   # 짝수 격자 한 변 칸 수
    "expected_dies":  44,
}


def active_radius(spec=WAFER_MODEL) -> float:
    return spec["diameter_mm"] / 2 - spec["edge_excl_mm"]


def die_layout(spec=WAFER_MODEL) -> list[dict]:
    """다이 목록. 좌표는 웨이퍼 중심 기준 mm (오른쪽 +x, 위쪽 +y).
    row 0 = 맨 위, col 0 = 맨 왼쪽. id 는 위→아래, 왼→오른 순서로 1부터."""
    n, p, r = spec["grid"], spec["pitch_mm"], active_radius(spec)
    half = n / 2
    dies = []
    for row in range(n):
        cy = (half - row - 0.5) * p
        for col in range(n):
            cx = (col - half + 0.5) * p
            # 다이의 가장 바깥 모서리까지 거리
            far = math.hypot(abs(cx) + p / 2, abs(cy) + p / 2)
            if far <= r:
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
    print(f"{s['name']}: 지름 {s['diameter_mm']:.1f} mm, 유효 반경 {s['active_radius_mm']} mm, 피치 {s['pitch_mm']} mm")
    print(f"다이 {s['die_count']}개 (기대 {s['expected_dies']}) 행별 {s['row_counts']}")
    print(f"다이 중심 범위 X {s['extent_mm']['x']}  Y {s['extent_mm']['y']}")
