import os
import time
import threading
import subprocess
from datetime import datetime

import cv2
from flask import Flask, Response, render_template, jsonify, request, send_from_directory

from die_segmentation import segment_die, draw_grid, MODE_GRID, MODE_CONTOUR
from defect_detection import detect_defects, draw_defects_on_frame
from wafer_map import WaferMap
from ai_inference import run_ai_detection
from process_analysis import analyze as analyze_process
from simulator import GENERATORS as SIM_GENERATORS
from mock_detector import generate as mock_detect, DEFAULT_IMAGE_SIZE
from wafer_spec import WAFER_SPEC
import scan_config
from scan_routes import create_scan_blueprint
from adc.routes import create_adc_blueprint


CAPTURES_DIR = "captures"
JPEG_QUALITY = 85
CAMERA_DEVICE = "/dev/video0"          # Arducam 1080P Low Light (USB, 어안 렌즈)
CAMERA_SIZE = (1920, 1080)             # 스캔 이미지 품질을 위해 1080p MJPG 로 받는다
STREAM_WIDTH = 960                     # 브라우저 스트림은 축소해서 보냄 (CPU·대역폭 절약)
AI_WIDTH = 640                         # AI 미리보기는 기존과 비슷한 크기로 축소해 입력
STREAM_FPS_LIMIT = 20
AI_FPS_LIMIT = 8

DEFAULT_MODE = MODE_GRID
DEFAULT_ROWS = 9
DEFAULT_COLS = 9

app = Flask(__name__)
os.makedirs(CAPTURES_DIR, exist_ok=True)

seg_params = {
    "mode": DEFAULT_MODE,
    "rows": DEFAULT_ROWS,
    "cols": DEFAULT_COLS,
    "enabled": True,
}

seg_lock = threading.Lock()
wafer_lock = threading.Lock()
wafer_map = WaferMap(DEFAULT_ROWS, DEFAULT_COLS)


class CameraManager:
    def __init__(self):
        self._cap = None
        self._raw_frame = None
        self._raw_frame_id = 0
        self._plain_jpeg = None
        self._seg_jpeg = None
        self._ai_jpeg = None
        self._lock = threading.Lock()
        self._running = False
        self._subs = {"plain": 0, "seg": 0, "ai": 0}
        self._subs_lock = threading.Lock()
        self._ai_last_id = 0

    def start(self, exposure=None):
        self._cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        if not self._cap.isOpened():
            raise RuntimeError(f"카메라 열기 실패: {CAMERA_DEVICE}")
        self.set_exposure(exposure)

        self._running = True
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._ai_worker, daemon=True).start()
        print("[Camera] started")

    def subscribe(self, name):
        with self._subs_lock:
            self._subs[name] = self._subs.get(name, 0) + 1

    def unsubscribe(self, name):
        with self._subs_lock:
            self._subs[name] = max(0, self._subs.get(name, 0) - 1)

    def _has_subs(self, name):
        with self._subs_lock:
            return self._subs.get(name, 0) > 0

    def _reader(self):
        interval = 1.0 / STREAM_FPS_LIMIT
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

        while self._running:
            cycle_start = time.monotonic()

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # raw frame은 항상 갱신 (캡처 엔드포인트/AI worker가 사용)
            # cap.read()는 매번 새 ndarray를 반환하므로 참조만 저장해도 안전하다.
            with self._lock:
                self._raw_frame = frame
                self._raw_frame_id += 1

            plain_bytes = None
            seg_bytes = None

            small = _shrink(frame, STREAM_WIDTH)

            if self._has_subs("plain"):
                ok, buf = cv2.imencode(".jpg", small, encode_params)
                if ok:
                    plain_bytes = buf.tobytes()

            if self._has_subs("seg"):
                with seg_lock:
                    params = dict(seg_params)
                try:
                    seg_vis = (
                        draw_grid(small, params["rows"], params["cols"])
                        if params["enabled"] else small
                    )
                except Exception:
                    seg_vis = small

                ok, buf = cv2.imencode(".jpg", seg_vis, encode_params)
                if ok:
                    seg_bytes = buf.tobytes()

            if plain_bytes is not None or seg_bytes is not None:
                with self._lock:
                    if plain_bytes is not None:
                        self._plain_jpeg = plain_bytes
                    if seg_bytes is not None:
                        self._seg_jpeg = seg_bytes

            # 데드라인 기반 sleep: 처리가 빠르면 그만큼 더 빠르게 다음 프레임
            remain = interval - (time.monotonic() - cycle_start)
            if remain > 0:
                time.sleep(remain)

    def _ai_worker(self):
        interval = 1.0 / AI_FPS_LIMIT
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

        while self._running:
            cycle_start = time.monotonic()

            if not self._has_subs("ai"):
                time.sleep(0.1)
                continue

            with self._lock:
                if self._raw_frame is None or self._raw_frame_id == self._ai_last_id:
                    frame = None
                else:
                    frame = self._raw_frame
                    self._ai_last_id = self._raw_frame_id

            if frame is None:
                time.sleep(0.02)
                continue

            try:
                _, ai_vis = run_ai_detection(_shrink(frame, AI_WIDTH))
            except Exception as e:
                print("[AI Error]", e)
                time.sleep(0.05)
                continue

            ok, buf = cv2.imencode(".jpg", ai_vis, encode_params)
            if ok:
                with self._lock:
                    self._ai_jpeg = buf.tobytes()

            remain = interval - (time.monotonic() - cycle_start)
            if remain > 0:
                time.sleep(remain)

    def get_jpeg_frame(self):
        with self._lock:
            return self._plain_jpeg

    def get_seg_jpeg_frame(self):
        with self._lock:
            return self._seg_jpeg

    def get_ai_jpeg_frame(self):
        with self._lock:
            return self._ai_jpeg

    def capture_raw_frame(self):
        with self._lock:
            return self._raw_frame.copy() if self._raw_frame is not None else None

    def frame_id(self):
        with self._lock:
            return self._raw_frame_id

    def wait_new_frame(self, after_id, timeout=2.0):
        """after_id 이후에 들어온 프레임을 기다려 복사본을 돌려준다.
        갠트리 정지 후 이 함수로 받으면 이동 중에 찍힌(흔들린) 프레임을 피할 수 있다."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._lock:
                if self._raw_frame is not None and self._raw_frame_id > after_id:
                    return self._raw_frame.copy()
            time.sleep(0.01)
        return None

    def set_exposure(self, value=None):
        """value=None 이면 자동 노출, 정수면 수동 노출 (1~5000, 작을수록 어두움).
        이 카메라는 자동 노출에서 밝은 실내에 쉽게 포화되므로 10~30 수동을 권장."""
        if value is None:
            args = ["--set-ctrl=auto_exposure=3"]
        else:
            v = max(1, min(5000, int(value)))
            args = ["--set-ctrl=auto_exposure=1", "--set-ctrl=backlight_compensation=0",
                    f"--set-ctrl=exposure_time_absolute={v}"]
        try:
            r = subprocess.run(["v4l2-ctl", "-d", CAMERA_DEVICE, *args],
                               capture_output=True, text=True, timeout=3)
            return r.returncode == 0
        except Exception as e:
            print("[Camera] exposure 설정 실패:", e)
            return False

    def is_ok(self):
        return self._cap is not None and self._cap.isOpened()


def _shrink(frame, width):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


camera = CameraManager()
app.register_blueprint(create_scan_blueprint(camera))   # /scan 콘솔 + 갠트리·촬영 API
app.register_blueprint(create_adc_blueprint())          # /analysis ADC + XAI 대시보드 API


def _mjpeg_stream(get_frame_fn, stream_name):
    interval = 1.0 / STREAM_FPS_LIMIT
    last_sent = None

    camera.subscribe(stream_name)
    try:
        while True:
            cycle_start = time.monotonic()
            frame = get_frame_fn()

            if frame is None or frame is last_sent:
                # 새 프레임 없으면 짧게 대기 (busy loop 방지)
                time.sleep(0.02)
                continue

            last_sent = frame
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

            remain = interval - (time.monotonic() - cycle_start)
            if remain > 0:
                time.sleep(remain)
    finally:
        camera.unsubscribe(stream_name)


def ai_detections_to_die_defects(ai_detections, frame_shape, rows, cols):
    h, w = frame_shape[:2]
    cell_h = h // rows
    cell_w = w // cols

    defects = []

    for d in ai_detections:
        x, y, bw, bh = d["bbox"]
        cx = x + bw // 2
        cy = y + bh // 2

        col = min(cols - 1, max(0, cx // cell_w))
        row = min(rows - 1, max(0, cy // cell_h))

        defects.append({
            "die": [int(row), int(col)],
            "type": d.get("type", d.get("label", "particle")),
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "confidence": float(d["confidence"]),
        })

    return defects


@app.route("/")
def index():
    with seg_lock:
        params = dict(seg_params)

    return render_template(
        "index.html",
        camera_ok=camera.is_ok(),
        seg_mode=params["mode"],
        seg_rows=params["rows"],
        seg_cols=params["cols"],
        seg_enabled=params["enabled"],
    )


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_stream(camera.get_jpeg_frame, "plain"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed_seg")
def video_feed_seg():
    return Response(_mjpeg_stream(camera.get_seg_jpeg_frame, "seg"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed_ai")
def video_feed_ai():
    return Response(_mjpeg_stream(camera.get_ai_jpeg_frame, "ai"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/capture")
def capture():
    if not camera.is_ok():
        return jsonify({"success": False, "error": "camera error"}), 503

    frame = camera.capture_raw_frame()
    if frame is None:
        return jsonify({"success": False, "error": "no frame"}), 503

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    with seg_lock:
        params = dict(seg_params)

    rows = params["rows"]
    cols = params["cols"]

    fname_raw = f"capture_{ts}_raw.jpg"
    cv2.imwrite(os.path.join(CAPTURES_DIR, fname_raw), frame)

    seg_result = segment_die(frame, mode=params["mode"], rows=rows, cols=cols)

    ai_detections, ai_vis = run_ai_detection(frame)
    ai_defects = ai_detections_to_die_defects(ai_detections, frame.shape, rows, cols)

    with wafer_lock:
        global wafer_map
        if wafer_map.rows != rows or wafer_map.cols != cols:
            wafer_map = WaferMap(rows, cols)

        wafer_map.add_defects(ai_defects)
        wafer_info = wafer_map.to_json()

    fname_seg = f"capture_{ts}_seg.jpg"
    cv2.imwrite(os.path.join(CAPTURES_DIR, fname_seg), seg_result.vis_frame)

    fname_ai = f"capture_{ts}_ai.jpg"
    cv2.imwrite(os.path.join(CAPTURES_DIR, fname_ai), ai_vis)

    defect_vis = draw_defects_on_frame(frame, ai_defects, rows, cols)
    fname_defect = f"capture_{ts}_defect.jpg"
    cv2.imwrite(os.path.join(CAPTURES_DIR, fname_defect), defect_vis)

    return jsonify({
        "success": True,
        "filename_raw": fname_raw,
        "filename_seg": fname_seg,
        "filename_ai": fname_ai,
        "filename_defect": fname_defect,
        "die_count": seg_result.die_count,
        "die_coords": seg_result.die_coords,
        "ai_detections": ai_detections,
        "ai_count": len(ai_detections),
        "defects": ai_defects,
        "defect_count": len(ai_defects),
        "wafer_map": wafer_info,
    })


@app.route("/set_seg", methods=["POST"])
def set_seg():
    data = request.get_json(silent=True) or {}

    with seg_lock:
        if "mode" in data and data["mode"] in (MODE_GRID, MODE_CONTOUR):
            seg_params["mode"] = data["mode"]
        if "rows" in data:
            seg_params["rows"] = max(1, min(int(data["rows"]), 20))
        if "cols" in data:
            seg_params["cols"] = max(1, min(int(data["cols"]), 20))
        if "enabled" in data:
            seg_params["enabled"] = bool(data["enabled"])

        params = dict(seg_params)

    return jsonify({"success": True, "params": params})


@app.route("/wafer_map")
def get_wafer_map():
    with wafer_lock:
        return jsonify(wafer_map.to_json())


@app.route("/reset_wafer_map", methods=["POST"])
def reset_wafer_map():
    with wafer_lock:
        wafer_map.reset()

    return jsonify({"success": True})


@app.route("/captures/<filename>")
def serve_capture(filename):
    return send_from_directory(CAPTURES_DIR, filename)


# ---------- Process Analysis Dashboard ----------

@app.route("/analysis")
def analysis_page():
    return render_template("analysis.html")


@app.route("/api/analysis")
def api_analysis():
    with wafer_lock:
        snapshot = wafer_map.to_json()
    snapshot["process"] = analyze_process(snapshot)
    return jsonify(snapshot)


@app.route("/api/simulate/<pattern>", methods=["POST"])
def api_simulate(pattern):
    gen = SIM_GENERATORS.get(pattern)
    if gen is None:
        return jsonify({
            "success": False,
            "error": f"unknown pattern: {pattern}",
            "available": list(SIM_GENERATORS.keys()),
        }), 400

    with wafer_lock:
        wafer_map.reset()
        wafer_map.add_defects(gen(wafer_map.rows, wafer_map.cols))
        snapshot = wafer_map.to_json()

    snapshot["process"] = analyze_process(snapshot)
    return jsonify({"success": True, "wafer": snapshot})


# 시나리오별 입력 이미지 / 메타 정보 매핑 (mock — 실제 deep learning 도입 전 데모용)
SOURCE_IMAGES = {
    "donut": {
        "file": "wafer_donut_input.jpg",
        "url":  "/static/img/wafer_donut_input.jpg",
        "scenario": "CVD/PVD 증착 균일도 불량에 의한 Donut Pattern",
        "expected_defect": "Stain",
        "expected_pattern": "Donut",
        "expected_zone": "Middle",
        "model": "Mock-YOLO v0.1 (placeholder)",
        "inference_ms": 12.4,
        "note": "실사 wafer 이미지에 대한 mock 분석 결과입니다. "
                "실제 deep learning 모델은 다음 단계에서 적용됩니다.",
    },
}


@app.route("/api/source")
def api_source():
    """현재 wafer 상태와 매칭되는 입력 이미지 메타를 반환한다.
    pattern을 보고 가장 적합한 시나리오의 이미지를 고른다.
    AI mock detection bbox도 함께 반환 (이미지 위 overlay 용)."""
    with wafer_lock:
        snapshot = wafer_map.to_json()
        rows, cols = wafer_map.rows, wafer_map.cols

    pattern = snapshot.get("pattern", "Normal")
    key = "donut"
    meta = dict(SOURCE_IMAGES[key])

    img_abs = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "static", "img", meta["file"],
    )
    meta["exists"] = os.path.isfile(img_abs)
    if not meta["exists"]:
        meta["url"] = None

    meta["current_pattern"] = pattern
    meta["captured_at"] = datetime.now().isoformat(timespec="seconds")

    # AI mock detection bbox 추가
    snap_process = analyze_process(snapshot)
    detections = mock_detect(
        snap_process["die_grid"],
        rows=rows, cols=cols,
        image_size=DEFAULT_IMAGE_SIZE,
    )
    meta["detections"]   = detections
    meta["detection_count"] = len(detections)
    meta["image_size"]   = DEFAULT_IMAGE_SIZE
    meta["wafer_spec"]   = WAFER_SPEC

    return jsonify(meta)


if __name__ == "__main__":
    try:
        camera.start(exposure=scan_config.load()["exposure"])
    except Exception as e:
        print("[Camera Error]", e)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False,
    )