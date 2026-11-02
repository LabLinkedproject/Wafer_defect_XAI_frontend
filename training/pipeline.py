"""
training/pipeline.py
내려받기 → 가공 → 라벨 데이터 학습 → 평가 → 추가 라벨 수집 → 재학습 → 평가 → 기준 비교 → 내보내기.

끊겨도 이어지도록 만들었다.
- 끝난 단계는 models/pipeline_state.json 에 남기고 다시 하지 않는다. 기록 전에 디스크에 확실히 쓴다.
- 잠금 파일로 동시에 하나만 돈다.
- 단계가 실패하면 그 자리에서 다시 시도하고, MAX_FAILS 번 실패하면 멈춘 뒤 상태 화면에 남긴다.

실행: ./training_ctl.sh start
"""
from __future__ import annotations
import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import durable                           # noqa: E402
import training_status                                 # noqa: E402

EPOCHS = "20"
STEPS = [("fetch_wm811k", []), ("prepare_data", []), ("train", ["--epochs", EPOCHS]), ("evaluate", ["--tag", "a"]),
         ("pseudo_label", []), ("train", ["--epochs", EPOCHS, "--pseudo", "--out", "models/wafernet_b.pt"]),
         ("evaluate", ["--ckpt", "models/wafernet_b.pt", "--tag", "b"]), ("baseline", []), ("tune", []), ("export", []), ("compare", [])]
STATE = "models/pipeline_state.json"
LOCK = "models/pipeline.lock"
DONE = "models/PIPELINE_DONE"
HEARTBEAT = "models/heartbeat.log"
MAX_FAILS = 5
RETRY_SECONDS = 20


def load_state() -> dict:
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"done": 0, "fails": 0}


def heartbeat():
    """다음에 보드가 갑자기 꺼져도 직전 상태를 볼 수 있게 남긴다. /var/log 는 재부팅하면 사라진다."""
    while True:
        with open("/proc/meminfo") as f:
            avail = int(next(line for line in f if line.startswith("MemAvailable")).split()[1]) // 1024
        durable.append_line(HEARTBEAT, f"{time.strftime('%m-%d %H:%M:%S')} {training_status.temperature_c():.0f}C 여유 {avail}MB 부하 {os.getloadavg()[0]:.1f}")
        time.sleep(15)


def keep_v1():
    """MixedWM38 전용 모델은 교차 데이터셋 시험에 쓴다."""
    os.makedirs("models/v1", exist_ok=True)
    for name in ("wafernet.pt", "training_log.jsonl", "wafernet_metrics.json"):
        if os.path.exists(f"models/{name}") and not os.path.exists(f"models/v1/{name}"):
            shutil.copy(f"models/{name}", f"models/v1/{name}")


def main():
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return                                           # 이미 돌고 있다
    threading.Thread(target=heartbeat, daemon=True).start()
    st = load_state()
    if st["done"] == 0 and st["fails"] == 0:
        keep_v1()
    durable.event(f"절차 시작: {st['done']}/{len(STEPS)} 단계까지 완료된 상태")
    py = sys.executable
    for k in range(st["done"], len(STEPS)):
        mod, args = STEPS[k]
        cpus = "0-3" if mod == "fetch_wm811k" else "4-7"   # 8코어 동시 사용 단계에서 보드가 재부팅된 적이 있다
        while True:
            training_status.report("pipeline", k, len(STEPS), f"{training_status.JOB_NAMES[mod]} 진행 중", unit="단계")
            with open(f"models/{k}_{mod}.log", "a") as out:
                code = subprocess.call(["taskset", "-c", cpus, py, "-m", f"training.{mod}", *args], stdout=out, stderr=subprocess.STDOUT)
            if code == 0:
                break
            if code == durable.EXIT_PAUSED:
                durable.event("절차 멈춤: 저장 후 대기")
                return
            st["fails"] += 1
            durable.write_json(STATE, st)
            durable.event(f"{training_status.JOB_NAMES[mod]} 실패 {st['fails']}/{MAX_FAILS}회: models/{k}_{mod}.log")
            if st["fails"] >= MAX_FAILS:
                durable.write_bytes(durable.STOP_FILE, b"failed\n")
                sys.exit(code)
            time.sleep(RETRY_SECONDS)
        os.sync()                                        # 결과가 디스크에 닿은 뒤에야 완료로 기록한다
        st = {"done": k + 1, "fails": 0}
        durable.write_json(STATE, st)
    durable.write_bytes(DONE, b"")
    training_status.report("pipeline", len(STEPS), len(STEPS), "전체 완료", unit="단계")
    durable.event("전체 절차 완료")


if __name__ == "__main__":
    main()
