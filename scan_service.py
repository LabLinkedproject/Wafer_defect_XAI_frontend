"""
scan_service.py
웹에서 갠트리를 움직이고 웨이퍼 모델을 다이별로 촬영하는 서비스.

· 모터 구동은 /home/orangepi/motor_test 의 gantry_core / gantry 모듈을 그대로 쓴다 (핀 매핑 단일 정의).
· 한 번에 하나의 작업만 백그라운드 스레드로 실행하고, GPIO 는 작업 동안만 점유한다
  → 작업이 없을 때는 터미널의 gantry.sh 도 그대로 쓸 수 있다.
· 위치는 gantry_state.json (CLI 와 공유) 에 저장된다.

환경변수:
    MOTOR_DIR          모터 코드 위치 (기본 /home/orangepi/motor_test)
    WAFER_GANTRY_DRY=1 GPIO 없이 동작 흉내 (화면·촬영 흐름 점검용, 위치 파일도 저장 안 함)
"""
from __future__ import annotations
import os
import sys
import json
import time
import threading
import traceback
from datetime import datetime

import cv2

import scan_config
import wafer_model

MOTOR_DIR = os.environ.get("MOTOR_DIR", "/home/orangepi/motor_test")
if MOTOR_DIR not in sys.path:
    sys.path.insert(0, MOTOR_DIR)
import gantry as gcli                      # noqa: E402  (motor_test/gantry.py)
from gantry_core import Gantry, Profile    # noqa: E402

DRY = os.environ.get("WAFER_GANTRY_DRY") == "1"
SCAN_ROOT = os.path.join("captures", "scans")
GPIO_CHIPS = [f"/dev/gpiochip{i}" for i in range(5)]


class JobError(Exception):
    pass


class ScanService:
    def __init__(self, camera):
        self.camera = camera
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._g: Gantry | None = None
        self._stop = False
        self.job = {"name": None, "busy": False, "message": "대기 중", "error": None,
                    "progress": 0, "total": 0, "started": None, "finished": None}
        self.scan = {"dir": None, "results": [], "preview": []}
        # DRY 에서는 실제 위치 파일을 지키기 위해 저장하지 않는다.
        # 단, GANTRY_STATE 로 별도(테스트용) 위치 파일을 지정했으면 그 파일에는 저장해 흐름을 그대로 재현한다.
        gcli.DRY = DRY and not os.environ.get("GANTRY_STATE")

    # ── 공통 ───────────────────────────────────────────────────────
    def gpio_access(self) -> bool:
        return DRY or all(os.access(c, os.R_OK | os.W_OK) for c in GPIO_CHIPS)

    def state_writable(self) -> bool:
        f = gcli.STATE_FILE
        return DRY or (os.access(f, os.W_OK) if f.exists() else os.access(f.parent, os.W_OK))

    def _state(self) -> dict:
        st = gcli.load_state()
        if st is None:
            raise JobError("원점이 지정되지 않았습니다. 갠트리를 중앙에 두고 터미널에서 "
                           "'gantry.sh set-center --micro 8' 을 먼저 실행하세요.")
        return st

    def _profile(self, st, cfg) -> Profile:
        return Profile(st["micro"], float(cfg["rpm"]), min(15.0, float(cfg["rpm"])), float(cfg["accel_rev"]))

    @staticmethod
    def _in_limits(x, y):
        (xl, xh), (yl, yh) = gcli.limits("x"), gcli.limits("y")
        return xl <= x <= xh and yl <= y <= yh

    def _move(self, g, st, prof, x, y):
        if not self._in_limits(x, y):
            (xl, xh), (yl, yh) = gcli.limits("x"), gcli.limits("y")
            raise JobError(f"({x:+.1f}, {y:+.1f}) mm 는 갠트리 허용 범위 "
                           f"X {xl:+.0f}~{xh:+.0f}, Y {yl:+.0f}~{yh:+.0f} 밖입니다.")
        ok = gcli.move_to(st, prof, x, y, g=g, quiet=True)
        if not ok and not g.stop_requested:
            raise JobError("이동 실패")
        return ok

    def status(self) -> dict:
        st = gcli.load_state()
        cfg = scan_config.load()
        (xl, xh), (yl, yh) = gcli.limits("x"), gcli.limits("y")
        return {
            "dry_run": DRY,
            "gpio_access": self.gpio_access(),
            "state_writable": self.state_writable(),
            "origin_set": st is not None,
            "position": None if st is None else {"x": st["x_mm"], "y": st["y_mm"]},
            "calibrated": None if st is None else {a: bool(st.get(f"calibrated_{a}")) for a in "xy"},
            "limits": {"x": [xl, xh], "y": [yl, yh]},
            "config": cfg,
            "job": dict(self.job),
            "scan": {"dir": self.scan["dir"], "count": len(self.scan["results"]),
                     "results": self.scan["results"], "preview": self.scan["preview"]},
        }

    def plan(self) -> dict:
        cfg = scan_config.load()
        spec = dict(wafer_model.WAFER_MODEL)
        if cfg.get("pitch_mm"):
            spec["pitch_mm"] = float(cfg["pitch_mm"])
        dies = wafer_model.die_layout(spec)
        order = wafer_model.scan_order(dies)
        cx, cy = float(cfg["wafer_center_x"]), float(cfg["wafer_center_y"])
        sx, sy = int(cfg["wafer_sign_x"]), int(cfg["wafer_sign_y"])
        for d in order:
            d["gx"] = round(cx + sx * d["x_mm"], 2)
            d["gy"] = round(cy + sy * d["y_mm"], 2)
            d["in_limits"] = self._in_limits(d["gx"], d["gy"])
        p = spec["pitch_mm"]
        ext = max(max(abs(d["x_mm"]), abs(d["y_mm"])) for d in dies) + p / 2   # 다이 바깥 경계
        corners = [(sx * ex, sy * ey) for ex, ey in ((-ext, ext), (ext, ext), (ext, -ext), (-ext, -ext))]
        return {
            "spec": wafer_model.summary(spec),
            "order": order,
            "bounds_mm": round(ext, 2),
            "corners": [{"name": n, "gx": round(cx + x, 2), "gy": round(cy + y, 2),
                         "in_limits": self._in_limits(cx + x, cy + y)}
                        for n, (x, y) in zip(("좌상", "우상", "우하", "좌하"), corners)],
            "all_in_limits": all(d["in_limits"] for d in order),
        }

    # ── 작업 실행 ──────────────────────────────────────────────────
    def _run(self, name, fn, total=0):
        with self._lock:
            if self.job["busy"]:
                raise JobError(f"'{self.job['name']}' 작업이 진행 중입니다.")
            if not self.gpio_access():
                raise JobError("GPIO 권한이 없습니다. 터미널에서 한 번만 실행하세요: "
                               "sudo /home/orangepi/motor_test/install_gpio_access.sh")
            if not self.state_writable():
                raise JobError(f"위치 파일({gcli.STATE_FILE})에 쓸 수 없습니다. "
                               "install_gpio_access.sh 를 실행하면 소유권이 정리됩니다.")
            st = self._state()
            self._stop = False
            self.job.update(name=name, busy=True, message="시작", error=None, progress=0, total=total,
                            started=datetime.now().isoformat(timespec="seconds"), finished=None)

        def worker():
            try:
                self._g = Gantry(dry=DRY, invert_m3=True, consumer="wafer_web")
                fn(self._g, st, scan_config.load())
                if self._stop:
                    self.job["message"] = "정지됨 (위치는 저장됨)"
                elif self.job["message"] == "시작":
                    self.job["message"] = "완료"
            except JobError as e:
                self.job.update(error=str(e), message="실패")
            except Exception as e:
                traceback.print_exc()
                self.job.update(error=f"{type(e).__name__}: {e}", message="실패")
            finally:
                if self._g is not None:
                    self._g.close()
                self._g = None
                self.job.update(busy=False, finished=datetime.now().isoformat(timespec="seconds"))

        self._thread = threading.Thread(target=worker, daemon=True, name=f"job-{name}")
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._g is not None:
            self._g.stop_requested = True

    # ── 단순 이동 ──────────────────────────────────────────────────
    def _precheck(self, x, y):
        """작업을 띄우기 전에 범위를 확인해, 범위 밖 요청은 즉시 거부한다."""
        if not self._in_limits(x, y):
            (xl, xh), (yl, yh) = gcli.limits("x"), gcli.limits("y")
            raise JobError(f"({x:+.1f}, {y:+.1f}) mm 는 갠트리 허용 범위 "
                           f"X {xl:+.0f}~{xh:+.0f}, Y {yl:+.0f}~{yh:+.0f} 밖입니다.")

    def jog(self, axis: str, mm: float):
        st0 = self._state()
        self._precheck(st0["x_mm"] + (mm if axis == "x" else 0.0), st0["y_mm"] + (mm if axis == "y" else 0.0))

        def fn(g, st, cfg):
            x = st["x_mm"] + (mm if axis == "x" else 0.0)
            y = st["y_mm"] + (mm if axis == "y" else 0.0)
            self.job["message"] = f"{axis.upper()} {mm:+g} mm 이동"
            self._move(g, st, self._profile(st, cfg), x, y)
            self.job["message"] = "완료"
        self._run("jog", fn)

    def goto(self, x: float, y: float, label="이동"):
        self._precheck(x, y)

        def fn(g, st, cfg):
            self.job["message"] = f"{label} → ({x:+.1f}, {y:+.1f})"
            self._move(g, st, self._profile(st, cfg), x, y)
            self.job["message"] = "완료"
        self._run("goto", fn)

    def goto_wafer_center(self):
        cfg = scan_config.load()
        self.goto(float(cfg["wafer_center_x"]), float(cfg["wafer_center_y"]), "웨이퍼 중심")

    def set_wafer_center_here(self) -> dict:
        if self.job["busy"]:
            raise JobError("작업 중에는 바꿀 수 없습니다.")
        st = self._state()
        return scan_config.update(wafer_center_x=round(st["x_mm"], 2), wafer_center_y=round(st["y_mm"], 2))

    # ── 촬영 ───────────────────────────────────────────────────────
    def _snap(self, out_dir, stem, cfg, dwell):
        """정지 후 dwell 초 기다렸다가 그 이후에 들어온 새 프레임을 저장. (원본, 썸네일) 상대 경로."""
        time.sleep(dwell)
        fid = self.camera.frame_id()
        frame = self.camera.wait_new_frame(fid, timeout=2.0)
        if frame is None:
            raise JobError("카메라 프레임을 받지 못했습니다 (카메라 연결 확인).")
        full = os.path.join(out_dir, f"{stem}.jpg")
        thumb = os.path.join(out_dir, f"{stem}_t.jpg")
        cv2.imwrite(full, frame, [cv2.IMWRITE_JPEG_QUALITY, int(cfg["jpeg_quality"])])
        h, w = frame.shape[:2]
        cv2.imwrite(thumb, cv2.resize(frame, (320, int(h * 320 / w)), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        rel = os.path.relpath(out_dir, SCAN_ROOT)
        return f"{rel}/{stem}.jpg", f"{rel}/{stem}_t.jpg"

    def _new_dir(self, kind):
        d = os.path.join(SCAN_ROOT, f"{datetime.now():%Y%m%d_%H%M%S}_{kind}")
        os.makedirs(d, exist_ok=True)
        return d

    def preview_bounds(self):
        """스캔 영역 네 모서리(다이 바깥 경계)를 돌며 한 장씩 찍고 웨이퍼 중심으로 복귀.
        검정 테이프로 표시한 스캔 범위가 각 모서리 사진에 맞게 보이는지 확인하는 용도."""
        plan = self.plan()
        bad = [c["name"] for c in plan["corners"] if not c["in_limits"]]
        if bad:
            raise JobError(f"모서리 {', '.join(bad)} 가 갠트리 범위 밖입니다. 웨이퍼 중심 설정을 확인하세요.")

        def fn(g, st, cfg):
            prof = self._profile(st, cfg)
            out = self._new_dir("bounds")
            self.scan.update(dir=out, preview=[], results=[])
            self.job["total"] = len(plan["corners"])
            for i, c in enumerate(plan["corners"], 1):
                if self._stop:
                    return
                self.job.update(message=f"범위 확인: {c['name']} 모서리 ({c['gx']:+.0f}, {c['gy']:+.0f})", progress=i - 1)
                if not self._move(g, st, prof, c["gx"], c["gy"]):
                    return
                full, thumb = self._snap(out, f"corner_{i}", cfg, max(1.0, float(cfg["dwell_s"])))
                self.scan["preview"].append({"name": c["name"], "file": full, "thumb": thumb,
                                             "gx": c["gx"], "gy": c["gy"]})
                self.job["progress"] = i
            self.job["message"] = "웨이퍼 중심으로 복귀"
            self._move(g, st, prof, float(cfg["wafer_center_x"]), float(cfg["wafer_center_y"]))
            self.job["message"] = "범위 확인 완료"
        self._run("bounds", fn, total=4)

    def start_scan(self):
        """원클릭 스캔: 웨이퍼 중심 → 44개 다이를 지그재그로 돌며 정지·촬영 → 웨이퍼 중심 복귀."""
        plan = self.plan()
        if not plan["all_in_limits"]:
            raise JobError("일부 다이 위치가 갠트리 범위 밖입니다. 웨이퍼 중심 설정을 확인하세요.")
        order = plan["order"]

        def fn(g, st, cfg):
            prof = self._profile(st, cfg)
            out = self._new_dir("scan")
            self.scan.update(dir=out, results=[], preview=[])
            manifest = {"started": datetime.now().isoformat(timespec="seconds"),
                        "spec": plan["spec"], "config": cfg, "dies": []}
            self.job.update(total=len(order), message="웨이퍼 중심으로 이동")
            if not self._move(g, st, prof, float(cfg["wafer_center_x"]), float(cfg["wafer_center_y"])):
                return
            t0 = time.monotonic()
            for i, d in enumerate(order, 1):
                if self._stop:
                    break
                self.job["message"] = f"다이 {d['id']} (행 {d['row'] + 1}, 열 {d['col'] + 1}) 촬영 중 [{i}/{len(order)}]"
                if not self._move(g, st, prof, d["gx"], d["gy"]):
                    break
                full, thumb = self._snap(out, f"die{d['id']:02d}_r{d['row']}c{d['col']}", cfg, float(cfg["dwell_s"]))
                rec = {**d, "file": full, "thumb": thumb,
                       "pos": {"x": round(st["x_mm"], 3), "y": round(st["y_mm"], 3)},
                       "t": round(time.monotonic() - t0, 2)}
                self.scan["results"].append(rec)
                manifest["dies"].append(rec)
                self.job["progress"] = i
                with open(os.path.join(out, "manifest.json"), "w") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)
            manifest["finished"] = datetime.now().isoformat(timespec="seconds")
            manifest["stopped"] = self._stop
            with open(os.path.join(out, "manifest.json"), "w") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            if self._stop:
                return
            self.job["message"] = "웨이퍼 중심으로 복귀"
            self._move(g, st, prof, float(cfg["wafer_center_x"]), float(cfg["wafer_center_y"]))
            self.job["message"] = f"스캔 완료: {len(self.scan['results'])}/{len(order)}장, {time.monotonic() - t0:.0f}초"
        self._run("scan", fn, total=len(order))
