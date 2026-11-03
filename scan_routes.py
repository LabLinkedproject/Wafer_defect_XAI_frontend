"""
scan_routes.py
웨이퍼 스캔 콘솔 (/scan) 과 갠트리·촬영 API.
"""
from __future__ import annotations
import os

from flask import Blueprint, jsonify, render_template, request, send_from_directory

import scan_config
from scan_service import ScanService, JobError, SCAN_ROOT


def create_scan_blueprint(camera) -> Blueprint:
    bp = Blueprint("scan", __name__)
    svc = ScanService(camera)

    def ok(**kw):
        return jsonify({"success": True, **kw})

    def fail(msg, code=400):
        return jsonify({"success": False, "error": msg}), code

    def call(fn, *a, **kw):
        try:
            res = fn(*a, **kw)
            return ok(result=res)
        except JobError as e:
            return fail(str(e), 409)

    @bp.route("/scan")
    def scan_page():
        return render_template("scan.html")

    @bp.route("/api/scan/status")
    def status():
        return jsonify(svc.status())

    @bp.route("/api/scan/plan")
    def plan():
        return jsonify(svc.plan())

    @bp.route("/api/gantry/jog", methods=["POST"])
    def jog():
        d = request.get_json(silent=True) or {}
        axis = d.get("axis")
        try:
            mm = float(d.get("mm", 0))
        except (TypeError, ValueError):
            return fail("mm 값이 숫자가 아닙니다")
        if axis not in ("x", "y") or not (-400 <= mm <= 400) or mm == 0:
            return fail("axis 는 x/y, mm 는 0 이 아닌 ±400 이내")
        return call(svc.jog, axis, mm)

    @bp.route("/api/gantry/goto", methods=["POST"])
    def goto():
        d = request.get_json(silent=True) or {}
        try:
            return call(svc.goto, float(d["x"]), float(d["y"]))
        except (KeyError, TypeError, ValueError):
            return fail("x, y (mm) 가 필요합니다")

    @bp.route("/api/gantry/origin", methods=["POST"])
    def origin():
        return call(svc.goto, 0.0, 0.0, "갠트리 원점")

    @bp.route("/api/gantry/wafer-center", methods=["POST"])
    def wafer_center():
        return call(svc.goto_wafer_center)

    @bp.route("/api/gantry/stop", methods=["POST"])
    def stop():
        svc.stop()
        return ok()

    @bp.route("/api/scan/set-center", methods=["POST"])
    def set_center():
        return call(svc.set_wafer_center_here)

    @bp.route("/api/scan/bounds", methods=["POST"])
    def bounds():
        return call(svc.preview_bounds)

    @bp.route("/api/scan/start", methods=["POST"])
    def start():
        return call(svc.start_scan)

    @bp.route("/api/scan/map", methods=["POST"])
    def build_map():
        return call(svc.build_latest_map)

    @bp.route("/api/scan/align", methods=["POST"])
    def align():
        return call(svc.apply_alignment)

    @bp.route("/api/scan/map")
    def get_map():
        return jsonify(svc.latest_map() or {})

    @bp.route("/api/scan/config", methods=["POST"])
    def config():
        d = request.get_json(silent=True) or {}
        changes = {}
        try:
            if "rpm" in d:
                changes["rpm"] = max(10.0, min(240.0, float(d["rpm"])))
            if "dwell_s" in d:
                changes["dwell_s"] = max(0.0, min(5.0, float(d["dwell_s"])))
            if "pitch_mm" in d:
                changes["pitch_mm"] = None if d["pitch_mm"] in (None, "") else max(5.0, min(60.0, float(d["pitch_mm"])))
            for k in ("wafer_sign_x", "wafer_sign_y"):
                if k in d:
                    changes[k] = -1 if int(d[k]) < 0 else 1
            for k in ("exposure", "scan_exposure"):
                if k in d:
                    changes[k] = None if d[k] in (None, "", "auto") else max(1, min(5000, int(d[k])))
        except (TypeError, ValueError):
            return fail("설정 값 형식이 올바르지 않습니다")
        cfg = scan_config.update(**changes)
        if "exposure" in changes:
            camera.set_exposure(cfg["exposure"])
        return ok(config=cfg)

    @bp.route("/scans/<path:relpath>")
    def scan_file(relpath):
        return send_from_directory(os.path.abspath(SCAN_ROOT), relpath)

    return bp
