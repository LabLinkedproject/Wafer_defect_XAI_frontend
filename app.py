"""
app.py
웹 서버. 화면은 두 개다.
  /scan      웨이퍼 스캔 콘솔: 카메라 미리보기, 갠트리 이동, 자동 스캔, 웨이퍼 맵 합성
  /analysis  분석 대시보드: 검출 → ADC 패턴 판정 → XAI 근거 → 능동 추론 → 공정 피드백

실행: ./venv/bin/python app.py   (http://<보드 IP>:5000)
"""
import subprocess
import threading
import time

import cv2
from flask import Flask, Response, redirect, render_template

import scan_config
from adc.routes import create_adc_blueprint
from scan_routes import create_scan_blueprint

JPEG_QUALITY = 85
CAMERA_DEVICE = "/dev/video0"          # Arducam 1080P Low Light (USB, 어안 렌즈)
CAMERA_SIZE = (1920, 1080)             # 스캔 이미지 품질을 위해 1080p MJPG 로 받는다
STREAM_WIDTH = 960                     # 브라우저 스트림은 축소해서 보냄 (CPU·대역폭 절약)
STREAM_FPS_LIMIT = 20

app = Flask(__name__)


class CameraManager:
    """카메라 하나를 읽어 최신 원본 프레임과 미리보기 JPEG 를 유지한다."""

    def __init__(self):
        self._cap = None
        self._raw_frame = None
        self._raw_frame_id = 0
        self._jpeg = None
        self._lock = threading.Lock()
        self._running = False
        self._viewers = 0

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
        print("[Camera] started")

    def _reader(self):
        interval = 1.0 / STREAM_FPS_LIMIT
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        while self._running:
            t0 = time.monotonic()
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            # 원본은 항상 갱신한다 (스캔 촬영이 사용). cap.read() 는 매번 새 배열이라 참조만 저장해도 안전하다.
            with self._lock:
                self._raw_frame = frame
                self._raw_frame_id += 1
            if self._viewers > 0:
                ok, buf = cv2.imencode(".jpg", _shrink(frame, STREAM_WIDTH), params)
                if ok:
                    with self._lock:
                        self._jpeg = buf.tobytes()
            remain = interval - (time.monotonic() - t0)
            if remain > 0:
                time.sleep(remain)

    def stream(self):
        """미리보기 MJPEG. 보는 사람이 있을 때만 인코딩한다."""
        self._viewers += 1
        last = None
        try:
            while True:
                t0 = time.monotonic()
                with self._lock:
                    jpeg = self._jpeg
                if jpeg is None or jpeg is last:
                    time.sleep(0.02)
                    continue
                last = jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                remain = 1.0 / STREAM_FPS_LIMIT - (time.monotonic() - t0)
                if remain > 0:
                    time.sleep(remain)
        finally:
            self._viewers -= 1

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
        """value=None 이면 자동 노출, 정수면 수동 노출 (1~5000, 작을수록 어두움)."""
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
app.register_blueprint(create_adc_blueprint())          # /analysis 대시보드 API


@app.route("/")
def index():
    return redirect("/scan")


@app.route("/video_feed")
def video_feed():
    return Response(camera.stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/analysis")
def analysis_page():
    return render_template("analysis.html")


if __name__ == "__main__":
    try:
        camera.start(exposure=scan_config.load()["exposure"])
    except Exception as e:
        print("[Camera Error]", e)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
