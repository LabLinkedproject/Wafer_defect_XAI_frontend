"""
training/backup.py
학습 결과를 보드 밖으로 옮긴다. 30분마다 cron 이 부른다. 어느 경로든 없으면 조용히 넘어가고 다음에 다시 시도한다.

1. USB 메모리: /media 나 /mnt 아래 마운트된 곳에 바로 복사한다.
2. GitHub: 네트워크가 되면 training-backup 브랜치에 올린다. 작업 브랜치와 작업 폴더는 건드리지 않는다.

코드와 파일 크기가 작은 결과물만 옮긴다. 데이터셋은 다시 받을 수 있어 제외한다.
"""
from __future__ import annotations
import glob
import os
import shutil
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import training_status                                  # noqa: E402

BRANCH = "training-backup"
WORKTREE = os.path.expanduser("~/wafer_backup")
PATTERNS = ("*.json", "*.jsonl", "*.onnx", "*.pt", "resume_*.pt", "events.log", "*_*.log")
SKIP = ("progress_",)
LOG = os.path.join(BASE, "models", "events.log")


def note(msg: str):
    with open(LOG, "a", encoding="utf8") as f:
        f.write(f"{time.strftime('%m-%d %H:%M:%S')} {msg}\n")


def files() -> list[str]:
    out = []
    for p in PATTERNS:
        out += glob.glob(os.path.join(BASE, "models", p))
    return sorted({f for f in out if not os.path.basename(f).startswith(SKIP)})


def copy_to(root: str) -> int:
    os.makedirs(root, exist_ok=True)
    n = 0
    for f in files():
        d = os.path.join(root, os.path.basename(f))
        if not os.path.exists(d) or os.path.getmtime(f) > os.path.getmtime(d):
            shutil.copy2(f, d)
            n += 1
    return n


def to_usb():
    for root in glob.glob("/media/*/*") + glob.glob("/mnt/*"):
        if os.path.ismount(root) and os.access(root, os.W_OK):
            n = copy_to(os.path.join(root, "wafer_backup"))
            subprocess.run(["sync"], timeout=120)
            if n:
                note(f"USB 백업 {n}개 파일 → {root}")


def git(*args, timeout=60):
    return subprocess.run(["git", *args], cwd=WORKTREE, timeout=timeout, capture_output=True, text=True)


def to_github():
    if not os.path.isdir(WORKTREE):
        subprocess.run(["git", "-C", BASE, "worktree", "add", "-q", "-B", BRANCH, WORKTREE, "HEAD"], timeout=60, check=True)
    copy_to(os.path.join(WORKTREE, "backup"))
    git("add", "-f", "backup")
    if git("diff", "--cached", "--quiet").returncode == 0:
        return
    git("-c", "user.name=wafer-backup", "-c", "user.email=noreply@anthropic.com", "commit", "-q", "-m", f"학습 결과 자동 백업 {time.strftime('%m-%d %H:%M')}")
    r = git("push", "-q", "origin", BRANCH, timeout=180)
    note("GitHub 백업 완료" if r.returncode == 0 else "GitHub 백업 대기: 네트워크 없음")


def main():
    for fn in (to_usb, to_github):
        try:
            fn()
        except Exception as e:                           # 백업 실패가 학습을 방해하면 안 된다
            note(f"백업 건너뜀 {fn.__name__}: {type(e).__name__}")


if __name__ == "__main__":
    main()
