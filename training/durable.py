"""
training/durable.py
끊겨도 잃지 않기 위한 공통 도구. 이 보드는 SD 카드 하나에 쓰기 지연이 10분이라 fsync 없이는 전원이 끊길 때 기록이 사라진다.

- 파일은 임시 파일에 쓰고 fsync 한 뒤 교체한다. 쓰다 끊겨도 이전 파일이 온전히 남는다.
- 체크포인트는 두 세대를 둔다. 최신 파일이 깨졌으면 그 전 것으로 돌아간다.
- 멈춤 요청은 models/STOP 또는 종료 신호로 받고, 작업은 체크포인트를 남긴 뒤 EXIT_PAUSED 로 끝난다.
"""
from __future__ import annotations
import io
import json
import os
import signal
import time

import training_status

BASE = training_status.BASE
MODELS = os.path.join(BASE, "models")
STOP_FILE = os.path.join(MODELS, "STOP")
EVENTS = os.path.join(MODELS, "events.log")
EXIT_PAUSED = 75
PAUSE_C = 80                  # RK3588 은 85 °C 부근에서 스스로 클럭을 내린다. 그 전에 쉰다
RESUME_C = 65
_signalled = False


def _fsync_dir(path: str):
    fd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_bytes(path: str, data: bytes):
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path)


def write_json(path: str, obj):
    write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2, default=float).encode("utf8"))


def append_line(path: str, line: str):
    with open(path, "a", encoding="utf8") as f:
        f.write(line.rstrip("\n") + "\n")
        f.flush()
        os.fsync(f.fileno())


def event(message: str):
    """나중에 무슨 일이 있었는지 볼 수 있도록 시각과 함께 남긴다. /var/log 는 메모리라 재부팅하면 지워진다."""
    os.makedirs(MODELS, exist_ok=True)
    append_line(EVENTS, f"{time.strftime('%m-%d %H:%M:%S')} {message}")


def save_checkpoint(obj, path: str):
    """최신 파일을 .prev 로 밀어 두 세대를 유지한다."""
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    tmp = f"{path}.new"
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        os.replace(path, f"{path}.prev")
    os.replace(tmp, path)
    _fsync_dir(path)


def load_checkpoint(path: str):
    """읽을 수 있는 가장 새 체크포인트. 없으면 None"""
    import torch
    for p in (path, f"{path}.prev"):
        if os.path.exists(p):
            try:
                return torch.load(p, weights_only=False)
            except Exception as e:                       # 끊긴 순간의 파일이 깨졌을 수 있다
                event(f"체크포인트 읽기 실패 {os.path.basename(p)}: {type(e).__name__}")
    return None


def catch_stop_signal():
    """시스템 종료·kill 신호를 받으면 다음 저장 지점에서 멈춘다."""
    def handle(*_):
        global _signalled
        _signalled = True
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGHUP, handle)


def stop_requested() -> bool:
    return _signalled or os.path.exists(STOP_FILE)


def wait_if_hot(job: str, done: int, total: int, unit: str):
    """코어가 PAUSE_C 를 넘으면 RESUME_C 까지 식을 때까지 쉰다. 쉬는 동안에도 멈춤 요청을 받는다."""
    t = training_status.temperature_c()
    if t is None or t < PAUSE_C:
        return
    event(f"과열 대기 시작 {t:.0f}°C ({job})")
    while t is not None and t > RESUME_C and not stop_requested():
        training_status.report(job, done, total, f"과열 대기 중 {t:.0f}°C, {RESUME_C}°C 까지 식으면 이어갑니다", unit=unit)
        time.sleep(10)
        t = training_status.temperature_c()
    event(f"과열 대기 끝 {t:.0f}°C ({job})")
