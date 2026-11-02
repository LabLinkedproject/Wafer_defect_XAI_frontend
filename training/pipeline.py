"""
training/pipeline.py
내려받기가 끝나기를 기다린 뒤 가공 → 라벨 데이터 학습 → 평가 → 추가 라벨 수집 → 재학습 → 평가
→ 기준 비교 → 내보내기를 차례로 실행한다.
한 단계가 실패하면 멈추고 그 사실을 상태 화면에 남긴다.

실행: ./venv-train/bin/python -m training.pipeline
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import training_status                                 # noqa: E402

EPOCHS = "20"
STEPS = [("prepare_data", []), ("train", ["--epochs", EPOCHS]), ("evaluate", ["--tag", "a"]), ("pseudo_label", []),
         ("train", ["--epochs", EPOCHS, "--pseudo", "--out", "models/wafernet_b.pt"]),
         ("evaluate", ["--ckpt", "models/wafernet_b.pt", "--tag", "b"]), ("baseline", []), ("export", [])]


def others_running() -> bool:
    return any(p["job"] != "pipeline" for p in training_status._processes())


def keep_v1():
    """MixedWM38 전용 모델은 교차 데이터셋 시험에 쓴다."""
    os.makedirs("models/v1", exist_ok=True)
    for name in ("wafernet.pt", "training_log.jsonl", "wafernet_metrics.json"):
        if os.path.exists(f"models/{name}") and not os.path.exists(f"models/v1/{name}"):
            shutil.copy(f"models/{name}", f"models/v1/{name}")


def main():
    while others_running():
        training_status.report("pipeline", 0, len(STEPS), "앞선 작업이 끝나기를 기다리는 중", unit="단계")
        time.sleep(15)
    if not os.path.exists("datasets/wm811k_labeled.npz"):
        training_status.report("pipeline", 0, len(STEPS), "내려받기 실패: WM-811K 파일이 없습니다", unit="단계")
        sys.exit(1)
    keep_v1()
    py = sys.executable
    for k, (mod, args) in enumerate(STEPS):
        training_status.report("pipeline", k, len(STEPS), f"{training_status.JOB_NAMES[mod]} 진행 중", unit="단계")
        cpus = "4-7" if mod == "train" else "0-7"
        code = subprocess.call(["taskset", "-c", cpus, py, "-m", f"training.{mod}", *args], stdout=open(f"models/{k}_{mod}.log", "w"), stderr=subprocess.STDOUT)
        if code:
            training_status.report("pipeline", k, len(STEPS), f"{training_status.JOB_NAMES[mod]} 실패: models/{k}_{mod}.log 확인", unit="단계", failed=True)
            sys.exit(code)
    training_status.report("pipeline", len(STEPS), len(STEPS), "전체 완료", unit="단계")


if __name__ == "__main__":
    main()
