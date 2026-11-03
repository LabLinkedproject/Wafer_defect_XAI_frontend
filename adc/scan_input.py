"""
adc/scan_input.py
실측 스캔 → 분석 입력. 합성 맵에서 결함을 검출하고 다이별 개수와 구역 판정을 만든다.
검출은 몇 초 걸리므로 결과를 스캔 폴더의 detection.json 에 남겨 두고, 맵이 바뀌지 않았으면 다시 쓴다.
"""
from __future__ import annotations
import glob
import json
import os
from datetime import datetime

import cv2

import scan_config
import wafer_detect
from .knowledge import DEFECT_TYPES

SCAN_ROOT = os.path.join("captures", "scans")
CACHE = "detection.json"
VERSION = 7                     # 검출 방식이 바뀌면 올린다. 이전 결과는 다시 계산한다


def scan_dir(scan_id: str) -> str:
    d = os.path.join(SCAN_ROOT, os.path.basename(scan_id))
    if not os.path.isfile(os.path.join(d, "wafer_map.json")):
        raise KeyError(scan_id)
    return d


def map_meta(scan_id: str) -> dict:
    d = scan_dir(scan_id)
    with open(os.path.join(d, "wafer_map.json")) as f:
        m = json.load(f)
    m["dir"] = os.path.basename(d)
    m["url"] = f"/scans/{m['dir']}/{m['map']}"
    return m


def label(scan_id: str) -> str:
    try:
        t = datetime.strptime(scan_id[:15], "%Y%m%d_%H%M%S").strftime("%m-%d %H:%M")
    except ValueError:
        t = scan_id
    return f"{t} 스캔"


def scans() -> list[dict]:
    """맵이 합성된 스캔, 최신순"""
    out = []
    for meta in sorted(glob.glob(os.path.join(SCAN_ROOT, "*_scan", "wafer_map.json")), reverse=True):
        sid = os.path.basename(os.path.dirname(meta))
        out.append({"id": sid, "name": label(sid)})
    return out


def detect(scan_id: str) -> dict:
    """스캔 맵의 결함 검출 결과. 맵이 새로 합성되면 다시 계산한다."""
    d = scan_dir(scan_id)
    path = os.path.join(d, CACHE)
    map_path = os.path.join(d, map_meta(scan_id)["map"])
    if os.path.isfile(path) and os.path.getmtime(path) >= os.path.getmtime(map_path):
        with open(path) as f:
            cached = json.load(f)
        if cached.get("version") == VERSION:
            return cached
    meta = wafer_detect.load_meta(d)
    gray, prof = wafer_detect.detection_channel(cv2.imread(map_path), meta,
                                                use_chroma=bool(scan_config.load().get("detect_chroma")))
    mask_path = os.path.join(d, meta.get("mask", ""))
    seen = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if meta.get("mask") and os.path.isfile(mask_path) else None
    r = wafer_detect.detect_array(gray, meta, seen=seen, min_area_mm2=prof["min_area_mm2"], sharp_z=prof["sharp_z"],
                                  dark_only=prof["dark_only"])
    out = {"version": VERSION, "scan": os.path.basename(d), "defects": r["defects"], "inspected": r["inspected"],
           "per_die": 2, "quadrats": [[k[0], k[1], k[2], v] for k, v in sorted(r["quadrats"].items())],
           "noise": r["noise"], "shift_px": r["shift_px"], "channel": prof["channel"]}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, ensure_ascii=False, default=float)
    os.replace(tmp, path)
    return out


def die_defects(det: dict) -> dict[int, dict]:
    """검출 목록 → {다이 id: {유형: 개수}} (분석 엔진 입력)"""
    out: dict[int, dict] = {}
    for x in det["defects"]:
        c = out.setdefault(int(x["die"]), {t: 0 for t in DEFECT_TYPES})
        c[x["type"]] = c.get(x["type"], 0) + 1
    return out


def quadrats(det: dict) -> dict[tuple, float]:
    return {(int(a), int(b), int(c)): float(v) for a, b, c, v in det["quadrats"]}
