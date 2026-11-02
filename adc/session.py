"""
adc/session.py
현재 웨이퍼 상태를 들고 있는 서버 세션. 시나리오·다이 수정·현장 확인 답변이 여기 쌓인다.
다이를 고치면 영상 근거는 다시 계산하되 현장 확인 답변은 유지한다. 새 사실이 들어오면 판단이 그대로 이어지게 하기 위해서다.
"""
from __future__ import annotations
import glob
import json
import os
import threading
from datetime import datetime

import scan_config
import wafer_model
from . import engine, render, scenarios, scoring
from .knowledge import CHECKS, DEFECT_TYPES, PATTERNS

FEEDBACK_FILE = "adc_feedback.jsonl"
SCAN_ROOT = os.path.join("captures", "scans")
_CHECK_IDS = {c["id"] for c in CHECKS}


class Session:
    def __init__(self):
        self._lock = threading.RLock()
        self.scenario = "donut"
        self.defects: dict[int, dict] = {}
        self.answers: list[dict] = []
        self.edited = False
        self.target = None
        self._cache = None
        self.load_scenario("donut")

    def _model(self):
        spec = dict(wafer_model.WAFER_MODEL)
        pitch = scan_config.load().get("pitch_mm")
        if pitch:
            spec["pitch_mm"] = float(pitch)
        return spec, scoring.annotate(wafer_model.die_layout(spec))

    def _invalidate(self):
        self._cache = None

    def load_scenario(self, name: str):
        with self._lock:
            if name not in scenarios.SCENARIOS:
                raise KeyError(name)
            _, dies = self._model()
            self.scenario = name
            self.defects = scenarios.generate(name, dies)
            self.answers, self.edited, self.target = [], False, None
            self._invalidate()

    def set_die(self, die_id: int, dtype: str | None, count: int):
        with self._lock:
            _, dies = self._model()
            if die_id not in {d["id"] for d in dies}:
                raise KeyError(die_id)
            if dtype is None or count <= 0:
                self.defects.pop(die_id, None)
            else:
                if dtype not in DEFECT_TYPES:
                    raise KeyError(dtype)
                self.defects[die_id] = {t: (count if t == dtype else 0) for t in DEFECT_TYPES}
            self.edited = True
            self._invalidate()

    def answer(self, check_id: str, answer: str):
        with self._lock:
            if check_id not in _CHECK_IDS or answer not in ("yes", "no", "unknown"):
                raise KeyError(check_id)
            self.answers = [a for a in self.answers if a["check"] != check_id]
            self.answers.append({"check": check_id, "answer": answer})
            self._invalidate()

    def undo_answer(self):
        with self._lock:
            if self.answers:
                self.answers.pop()
                self._invalidate()

    def set_target(self, target: str | None):
        with self._lock:
            self.target = target if target in PATTERNS else None
            self._invalidate()

    def latest_scan(self) -> dict | None:
        """가장 최근 실제 스캔 이미지. 다이 인스펙터에서 합성 이미지 대신 보여 준다."""
        for path in sorted(glob.glob(os.path.join(SCAN_ROOT, "*_scan", "manifest.json")), reverse=True):
            try:
                with open(path) as f:
                    m = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if m.get("dies"):
                return {"dir": os.path.basename(os.path.dirname(path)), "started": m.get("started"),
                        "count": len(m["dies"]),
                        "images": {str(d["id"]): {"file": d["file"], "thumb": d["thumb"]} for d in m["dies"]}}
        return None

    def state(self) -> dict:
        with self._lock:
            if self._cache is None:
                spec, dies = self._model()
                r = engine.analyze(dies, spec, self.defects, self.answers, self.target)
                r["session"] = {"scenario": self.scenario, "edited": self.edited,
                                "scenario_name": scenarios.SCENARIOS[self.scenario]["name"],
                                "scenario_note": scenarios.SCENARIOS[self.scenario]["note"],
                                "feedback_count": self.feedback_count()}
                r["scenarios"] = [{"id": k, **v} for k, v in scenarios.SCENARIOS.items()]
                self._cache = r
            res = dict(self._cache)
            res["scan"] = self.latest_scan()
            return res

    def input_image(self) -> bytes:
        with self._lock:
            spec, dies = self._model()
            return render.render_input(dies, self.defects, spec["pitch_mm"], spec["diameter_mm"])

    def feedback_count(self) -> int:
        try:
            with open(FEEDBACK_FILE) as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def add_feedback(self, verdict: str, labels: list[str], cause: str | None, note: str) -> int:
        """엔지니어 판정을 재학습 라벨 후보로 기록한다."""
        with self._lock:
            st = self.state()
            rec = {"time": datetime.now().isoformat(timespec="seconds"), "scenario": self.scenario,
                   "edited": self.edited, "verdict": verdict, "ai_labels": st["labels"],
                   "operator_labels": labels, "operator_cause": cause, "note": note[:500],
                   "die_defects": {str(k): v for k, v in self.defects.items()},
                   "answers": self.answers}
            with open(FEEDBACK_FILE, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._invalidate()
            return self.feedback_count()
