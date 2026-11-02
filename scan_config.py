"""
scan_config.py
스캔 콘솔 설정 (웹에서 바꾸고 파일로 유지). 런타임 값이라 git 에는 기본값만 코드로 둔다.
"""
from __future__ import annotations
import json
import pathlib
import threading

CONFIG_FILE = pathlib.Path(__file__).with_name("scan_config.json")

DEFAULTS = {
    # 갠트리 좌표계(중앙 원점, mm)에서 웨이퍼 중심이 놓인 위치. '현재 위치를 웨이퍼 중심으로' 버튼으로 설정
    "wafer_center_x": 0.0,
    "wafer_center_y": 0.0,
    # 웨이퍼 좌표(+x 오른쪽, +y 위쪽)와 갠트리 축 방향이 반대면 -1
    "wafer_sign_x": 1,
    "wafer_sign_y": 1,
    "pitch_mm": None,          # None 이면 wafer_model.WAFER_MODEL 값 사용
    # 합성 맵에서 측정한 웨이퍼 반지름(갠트리 mm). 값이 있으면 웨이퍼 전체를 덮는 격자 경로로 스캔한다
    "map_radius_mm": None,
    "map_step_mm": 36.0,
    "rpm": 90.0,               # 프로토타입 기준 속도 (매핑 구현하며 점진 조정)
    "accel_rev": 0.5,
    "dwell_s": 0.6,            # 정지 후 흔들림이 가라앉을 때까지 대기
    "exposure": 20,            # 수동 노출. None 이면 자동 (이 카메라는 자동에서 포화되기 쉬움)
    "jpeg_quality": 92,
}

_lock = threading.Lock()


def load() -> dict:
    with _lock:
        try:
            data = json.loads(CONFIG_FILE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
    return {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}


def update(**changes) -> dict:
    cfg = load()
    for k, v in changes.items():
        if k in DEFAULTS:
            cfg[k] = v
    with _lock:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    return cfg
