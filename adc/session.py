"""
adc/session.py
현재 웨이퍼 상태를 들고 있는 서버 세션. 입력(실측 스캔 또는 시나리오)·다이 수정·현장 확인 답변이 여기 쌓인다.
실측 스캔이면 검출 결과를 학습 모델로 판정하고, 시나리오면 규칙 기반으로 판정한다.
다이를 고치면 영상 근거는 다시 계산하되 현장 확인 답변은 유지한다. 새 사실이 들어오면 판단이 그대로 이어지게 하기 위해서다.
"""
from __future__ import annotations
import json
import os
import threading
from datetime import datetime

import scan_config
import wafer_model
import wafer_stitch
from . import cnn, engine, render, scan_input, scenarios, scoring
from .knowledge import CHECKS, DEFECT_TYPES, PATTERNS

FEEDBACK_FILE = "adc_feedback.jsonl"
SCAN_ROOT = os.path.join("captures", "scans")
_CHECK_IDS = {c["id"] for c in CHECKS}


class Session:
    def __init__(self):
        self._lock = threading.RLock()
        self.source = "scenario"
        self.scenario = "donut"
        self.scan_id = None
        self.detection = None
        self.quadrats: dict[tuple, float] = {}
        self.defects: dict[int, dict] = {}
        self.answers: list[dict] = []
        self.edited = False
        self.target = None
        self._cache = None
        latest = scan_input.scans()
        try:
            self.load_scan(latest[0]["id"]) if latest else self.load_scenario("donut")
        except Exception as e:                           # 맵이 깨졌어도 대시보드는 열린다
            print("[ADC] 최신 스캔을 불러오지 못했습니다:", e)
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
            self.source, self.scenario, self.scan_id, self.detection, self.quadrats = "scenario", name, None, None, {}
            self.defects = scenarios.generate(name, dies)
            self.answers, self.edited, self.target = [], False, None
            self._invalidate()

    def load_scan(self, scan_id: str):
        """실측 스캔의 맵에서 결함을 검출해 입력으로 쓴다."""
        with self._lock:
            det = scan_input.detect(scan_id)
            self.source, self.scan_id, self.detection = "scan", det["scan"], det
            self.defects = scan_input.die_defects(det)
            self.quadrats = scan_input.quadrats(det)
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
            if self.source == "scan":
                # 엔지니어가 고친 다이는 구역 넷 모두 그 정도로 본다
                self.quadrats = {k: v for k, v in self.quadrats.items() if k[0] != die_id}
                w = scoring.die_weight(self.defects.get(die_id, {}))
                for qr in range(2):
                    for qc in range(2):
                        if w:
                            self.quadrats[(die_id, qr, qc)] = w
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

    def state(self) -> dict:
        with self._lock:
            if self._cache is None:
                spec, dies = self._model()
                model = None
                if self.source == "scan" and cnn.available():
                    model = cnn.Model(dies, self.quadrats, scoring.RuleModel(dies, {k: v for k, v in self.defects.items()
                                                                                 if sum(v.values())}))
                r = engine.analyze(dies, spec, self.defects, self.answers, self.target, model)
                sess = {"source": self.source, "scenario": self.scenario, "edited": self.edited,
                        "scenario_name": scenarios.SCENARIOS[self.scenario]["name"],
                        "scenario_note": scenarios.SCENARIOS[self.scenario]["note"],
                        "feedback_count": self.feedback_count()}
                if self.source == "scan":
                    det = self.detection
                    sess.update(scan_id=self.scan_id, scan_name=scan_input.label(self.scan_id),
                                inspected=len(det["inspected"]), found=len(det["defects"]))
                    r["found"] = det["defects"]
                    r["wafer_map"] = scan_input.map_meta(self.scan_id)
                else:
                    r["found"] = []
                    r["wafer_map"] = wafer_stitch.latest_map(SCAN_ROOT)
                r["session"] = sess
                r["scenarios"] = [{"id": k, **v} for k, v in scenarios.SCENARIOS.items()]
                self._cache = r
            res = dict(self._cache)
            res["scans"] = scan_input.scans()
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
            rec = {"time": datetime.now().isoformat(timespec="seconds"), "source": self.source,
                   "scenario": self.scenario if self.source == "scenario" else None, "scan": self.scan_id,
                   "model": st["model"]["kind"], "edited": self.edited, "verdict": verdict, "ai_labels": st["labels"],
                   "operator_labels": labels, "operator_cause": cause, "note": note[:500],
                   "die_defects": {str(k): v for k, v in self.defects.items()},
                   "detections": st.get("found", []),
                   "answers": self.answers}
            with open(FEEDBACK_FILE, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._invalidate()
            return self.feedback_count()
