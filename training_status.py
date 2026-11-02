"""
training_status.py
학습 작업의 상태를 한곳에서 판단한다. 웹 화면과 터미널 명령이 같은 판단을 쓴다.

실행 여부는 로그가 아니라 프로세스 목록으로 판정한다. 로그만 보면 멈춘 작업도 도는 것처럼 보이기 때문이다.
작업 스크립트는 report() 로 진행도를 남긴다. 표준 라이브러리만 쓴다.

터미널 확인: python3 training_status.py
"""
from __future__ import annotations
import json
import os
import re
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PROGRESS_DIR = os.path.join(BASE, "models")
LOG = os.path.join(BASE, "models", "training_log.jsonl")
METRICS = os.path.join(BASE, "models", "wafernet_metrics.json")
JOB_NAMES = {"train": "모델 학습", "fetch_wm811k": "데이터 내려받기", "prepare_data": "데이터 가공",
             "baseline": "기준 모델 비교", "evaluate": "성능 평가", "export": "모델 내보내기",
             "pseudo_label": "추가 라벨 수집", "pipeline": "전체 학습 절차"}
STALL_SECONDS = 180           # 프로세스는 있는데 이 시간 동안 진행 기록이 없으면 멈춤 의심


def report(job: str, done: int, total: int, detail: str = "", **extra):
    """작업 진행도를 남긴다. 화면이 읽는 도중 깨지지 않도록 교체 방식으로 쓴다."""
    row = {"job": job, "done": done, "total": total, "detail": detail, "time": time.time(), **extra}
    path = os.path.join(PROGRESS_DIR, f"progress_{job}.json")
    prev = _read_json(path)
    row["started"] = prev["started"] if prev and done else time.time()
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(row, f, ensure_ascii=False)
    os.replace(tmp, path)


def _progress_files() -> list[str]:
    return [os.path.join(PROGRESS_DIR, n) for n in os.listdir(PROGRESS_DIR)
            if n.startswith("progress_") and n.endswith(".json")]


def _read_json(path):
    try:
        with open(path, encoding="utf8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _processes() -> list[dict]:
    """실행 중인 training.* 프로세스"""
    out = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().decode("utf8", "replace").split("\0")
        except OSError:
            continue
        # 셸 명령 안의 문자열에 속지 않도록 실제 인자 구조를 본다
        if "python" not in os.path.basename(argv[0]) or "-m" not in argv:
            continue
        module = argv[argv.index("-m") + 1] if argv.index("-m") + 1 < len(argv) else ""
        if module.startswith("training."):
            out.append({"pid": int(pid), "job": module.split(".", 1)[1], "cmd": " ".join(argv)})
    return out


def _cpu_ticks(pid: int) -> int:
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])


def _cpu_percent(pid: int, wait: float = 0.25) -> float | None:
    """코어 하나를 100 으로 친 실제 사용률. 살아 있어도 놀고 있으면 낮게 나온다."""
    try:
        a = _cpu_ticks(pid)
        time.sleep(wait)
        return round((_cpu_ticks(pid) - a) / os.sysconf("SC_CLK_TCK") / wait * 100)
    except (OSError, IndexError, ValueError):
        return None


def _history() -> list[dict]:
    rows = []
    try:
        with open(LOG) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return rows


def _target_epochs(cmd: str) -> int | None:
    m = re.search(r"--epochs[ =](\d+)", cmd)
    return int(m.group(1)) if m else None


def _job(p: dict, prog, history: list[dict], now: float, log_time: float) -> dict:
    """실행 중인 작업 하나의 진행 상황"""
    job = {"job": p["job"], "job_name": JOB_NAMES.get(p["job"], p["job"]), "pid": p["pid"],
           "cpu_percent": _cpu_percent(p["pid"])}
    fresh = prog if prog and prog.get("job") == p["job"] else None
    if p["job"] == "train":
        total = _target_epochs(p["cmd"]) or (fresh or {}).get("total")
        done = history[-1]["epoch"] if history else 0
        mins = history[-1]["minutes"] if history else 0
        job.update(unit="에폭", done=done, total=total, last_update_age=round(now - log_time))
        if done and total:
            job["eta_seconds"] = round(mins / done * (total - done) * 60)
            job["elapsed_seconds"] = round(mins * 60)
        if fresh:
            job["detail"] = fresh.get("detail", "")
            job["last_update_age"] = round(now - fresh["time"])
    elif fresh:
        job.update(unit=fresh.get("unit", ""), done=fresh["done"], total=fresh["total"], detail=fresh.get("detail", ""),
                   last_update_age=round(now - fresh["time"]), elapsed_seconds=round(now - fresh["started"]))
        if fresh["done"] and fresh["total"]:
            job["eta_seconds"] = round((now - fresh["started"]) / fresh["done"] * (fresh["total"] - fresh["done"]))
    # 전체 절차는 하위 작업이 도는 동안 스스로 갱신하지 않는다
    job["stalled"] = (p["job"] != "pipeline" and job.get("last_update_age", 0) > STALL_SECONDS
                      and (job.get("cpu_percent") or 0) < 20)
    return job


def status() -> dict:
    procs = _processes()
    history = _history()
    now = time.time()
    metrics_time = os.path.getmtime(METRICS) if os.path.exists(METRICS) else 0
    log_time = os.path.getmtime(LOG) if os.path.exists(LOG) else 0
    out = {"now": now, "history": history, "metrics": _read_json(METRICS), "jobs": []}
    if procs:
        progress = {}
        for path in _progress_files():
            row = _read_json(path)
            if row:
                progress[row.get("job")] = row
        # 학습이 있으면 화면 머리에 학습을 먼저 보인다
        procs.sort(key=lambda p: p["job"] != "train")
        out["jobs"] = [_job(p, progress.get(p["job"]), history, now, log_time) for p in procs]
        out["state"] = "stalled" if all(j["stalled"] for j in out["jobs"]) else "running"
        out["training"] = procs[0]["job"] == "train"
        if out["training"]:
            out["epochs_total"] = out["jobs"][0].get("total")
    elif history and metrics_time < log_time:
        out.update(state="stopped", message=f"에폭 {history[-1]['epoch']}에서 학습 프로세스가 끝났고 최종 평가 기록이 없습니다.")
    elif metrics_time:
        out.update(state="idle", message="실행 중인 작업이 없습니다. 마지막 학습은 완료되었습니다.")
    else:
        out.update(state="idle", message="실행 중인 작업이 없습니다.")
    return out


def text() -> str:
    s = status()
    if not s["jobs"]:
        return ("✖ 중단됨: " if s["state"] == "stopped" else "○ 실행 안 함: ") + s["message"]
    lines = []
    for j in s["jobs"]:
        count = f"{j.get('done', '?')}/{j.get('total', '?')} {j.get('unit', '')}".strip()
        eta = f", 남은 시간 약 {j['eta_seconds'] // 60}분" if "eta_seconds" in j else ""
        flag = "  ⚠ 멈춘 것 같음" if j["stalled"] else ""
        lines.append(f"● 실행 중: {j['job_name']} {count}{eta}, CPU {j['cpu_percent']}% (PID {j['pid']}){flag}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(text())
