"""ADC + XAI 대시보드 API."""
from __future__ import annotations

from flask import Blueprint, Response, jsonify, request

from .knowledge import PATTERNS
from .session import Session


def create_adc_blueprint() -> Blueprint:
    bp = Blueprint("adc", __name__)
    s = Session()

    def body():
        return request.get_json(silent=True) or {}

    def bad(msg):
        return jsonify({"success": False, "error": msg}), 400

    @bp.route("/api/adc/state")
    def state():
        return jsonify(s.state())

    @bp.route("/api/adc/input.jpg")
    def input_image():
        return Response(s.input_image(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @bp.route("/api/adc/scenario", methods=["POST"])
    def scenario():
        try:
            s.load_scenario(body().get("id", ""))
        except KeyError:
            return bad("알 수 없는 시나리오")
        return jsonify(s.state())

    @bp.route("/api/adc/die", methods=["POST"])
    def die():
        d = body()
        try:
            s.set_die(int(d["id"]), d.get("type"), int(d.get("count", 0)))
        except (KeyError, TypeError, ValueError):
            return bad("다이 id, 유형, 개수를 확인하세요")
        return jsonify(s.state())

    @bp.route("/api/adc/answer", methods=["POST"])
    def answer():
        d = body()
        try:
            s.answer(d.get("check", ""), d.get("answer", ""))
        except KeyError:
            return bad("확인 항목 또는 답변이 올바르지 않습니다")
        return jsonify(s.state())

    @bp.route("/api/adc/answer/undo", methods=["POST"])
    def undo():
        s.undo_answer()
        return jsonify(s.state())

    @bp.route("/api/adc/target", methods=["POST"])
    def target():
        s.set_target(body().get("target"))
        return jsonify(s.state())

    @bp.route("/api/adc/feedback", methods=["POST"])
    def feedback():
        d = body()
        verdict = d.get("verdict")
        labels = [x for x in d.get("labels", []) if x in PATTERNS]
        if verdict not in ("confirm", "correct"):
            return bad("verdict 는 confirm 또는 correct")
        n = s.add_feedback(verdict, labels, d.get("cause"), str(d.get("note", "")))
        return jsonify({"success": True, "count": n})

    return bp
