"""training/model.py 분류 모델과 학습용 가공 연산. 다이 단위 입력은 wafer_raster.from_dies 와 같게 만든다."""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import wafer_raster as R


class WaferNet(nn.Module):
    """작은 CNN. 전역 평균 풀링 + 선형층이라 마지막 특징맵으로 CAM 을 바로 계산할 수 있다."""

    def __init__(self, classes: int = 8):
        super().__init__()
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.features = nn.Sequential(block(3, 16), nn.MaxPool2d(2), block(16, 32), nn.MaxPool2d(2),
                                      block(32, 48), nn.MaxPool2d(2), block(48, 64))
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        return self.head(self.drop(self.features(x).mean((2, 3))))


class Geometry:
    def __init__(self, device="cpu"):
        self.disc = torch.from_numpy(R.wafer_mask())[None, None].to(device)
        self.dies = torch.from_numpy(R.die_mask())[None, None].to(device)
        self.radius = torch.from_numpy(R.radius_map())[None, None].to(device)


def augment(x: torch.Tensor) -> torch.Tensor:
    """회전·뒤집기와 정렬 오차. 공개 데이터는 노치 방향이 없고, 패턴 라벨은 방향에 무관하다.
    정렬 오차 ±8 mm, ±6° 는 합성 맵의 원·격자 측정 오차를 넉넉히 덮는 범위다."""
    n = len(x)
    a = torch.randint(0, 4, (n,)) * math.pi / 2 + torch.empty(n).uniform_(-math.radians(6), math.radians(6))
    flip = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
    sh = torch.empty(n, 2).uniform_(-1.5, 1.5) * 2 / R.SIZE
    c, s = torch.cos(a), torch.sin(a)
    t = torch.stack([torch.stack([c * flip, -s, sh[:, 0]], 1), torch.stack([s * flip, c, sh[:, 1]], 1)], 1)
    return F.grid_sample(x, F.affine_grid(t, x.shape, align_corners=False), align_corners=False, padding_mode="zeros")


def full_map(fill: torch.Tensor, g: Geometry, noise: torch.Tensor | None = None) -> torch.Tensor:
    """전면 맵 입력: 웨이퍼 전체 셀. noise 는 표본별 무작위 잡음 밀도."""
    if noise is not None:
        fill = torch.maximum(fill, (torch.rand_like(fill) < noise[:, None, None, None]).float())
    fill = fill * (g.disc > 0.5)
    return torch.cat([fill, g.disc.expand_as(fill), g.radius.expand_as(fill)], 1)


def quadrat_map(fill: torch.Tensor, g: Geometry, per_die: int, light=0.06, heavy=0.35, stray=None, keep=None) -> torch.Tensor:
    """검사 다이 44개만 쓰는 입력. 다이를 per_die×per_die 구역으로 나눠 구역마다 판정한다.
    per_die=1 이면 다이당 값 하나, 6 이면 셀 그대로다.
    light, heavy: 구역 불량 면적비 기준. stray: 구역별 산발 결함 확률. keep: 판정을 남길 비율."""
    n = len(fill)
    k = R.CELLS_PER_DIE // per_die
    v = F.avg_pool2d(fill, k) if k > 1 else fill
    light = torch.as_tensor(light, dtype=v.dtype).reshape(-1, 1, 1, 1)
    heavy = torch.as_tensor(heavy, dtype=v.dtype).reshape(-1, 1, 1, 1)
    state = ((v >= light) * R.LIGHT_VALUE + (v >= heavy) * (R.HEAVY_VALUE - R.LIGHT_VALUE)).float()
    if keep is not None:
        state = state * (torch.rand_like(state) < keep[:, None, None, None])
    if stray is not None:
        state = torch.maximum(state, (torch.rand_like(state) < stray[:, None, None, None]) * R.LIGHT_VALUE)
    grid = F.interpolate(state, scale_factor=k, mode="nearest") * g.dies if k > 1 else state * g.dies
    return torch.cat([grid, g.dies.expand(n, -1, -1, -1), g.radius.expand(n, -1, -1, -1)], 1)
