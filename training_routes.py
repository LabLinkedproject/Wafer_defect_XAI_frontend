"""
training_routes.py
학습 상태 화면 (/training) 과 상태 API. 메인 서버와 전용 모니터가 함께 쓴다.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, render_template

import training_status


def create_training_blueprint() -> Blueprint:
    bp = Blueprint("training", __name__)

    @bp.route("/training")
    def training_page():
        return render_template("training.html")

    @bp.route("/api/training/status")
    def api_status():
        return jsonify(training_status.status())

    return bp
