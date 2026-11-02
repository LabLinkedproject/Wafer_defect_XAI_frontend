"""기준 다이 결함 추출 시험. 실행: ./venv/bin/python -m unittest tests.test_detect -v"""
import unittest

import cv2
import numpy as np

import wafer_detect as W
import wafer_raster as R
from wafer_model import die_layout

P = 170
SIDE = R.GRID_DIES * P


def clean_map(seed=0):
    """스크라이브 라인과 조명 기울기, 잡음이 있는 곧은 웨이퍼 맵"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIDE, 0:SIDE].astype(np.float32)
    g = 150 + 0.01 * xx + 0.008 * yy
    for k in range(R.GRID_DIES + 1):
        g[:, max(k * P - 2, 0):k * P + 2] -= 25
        g[max(k * P - 2, 0):k * P + 2, :] -= 25
    g = g + rng.normal(0, 0.4, g.shape)
    meta = {"wafer": {"center_px": [SIDE / 2 - 0.5, SIDE / 2 - 0.5], "radius_px": 135 / 32 * P},
            "grid": {"pitch_px": float(P), "angle_deg": 0.0}}
    return np.clip(g, 0, 255).astype(np.uint8), meta


def die_at(i):
    return next(d for d in die_layout() if d["id"] == i)


class Detect(unittest.TestCase):
    def test_clean_map_has_no_defects(self):
        g, meta = clean_map()
        r = W.detect_array(g, meta)
        self.assertEqual(len(r["inspected"]), 44)
        self.assertEqual(r["defects"], [])

    def test_particle_is_found_in_the_right_die(self):
        g, meta = clean_map()
        a, b = R.die_slot(die_at(20))
        cv2.circle(g, (b * P + 80, a * P + 70), 4, 60, -1)
        r = W.detect_array(g, meta)
        self.assertEqual({d["die"] for d in r["defects"]}, {20})
        self.assertEqual(r["defects"][0]["type"], "particle")
        self.assertTrue(r["quadrats"])

    def test_scratch_is_long(self):
        g, meta = clean_map()
        a, b = R.die_slot(die_at(25))
        cv2.line(g, (b * P + 30, a * P + 40), (b * P + 140, a * P + 120), 90, 2)
        r = W.detect_array(g, meta)
        self.assertIn("scratch", {d["type"] for d in r["defects"] if d["die"] == 25})

    def test_coordinates_are_in_mm(self):
        g, meta = clean_map()
        d = die_at(20)
        a, b = R.die_slot(d)
        cv2.circle(g, (b * P + 85, a * P + 85), 4, 60, -1)
        f = W.detect_array(g, meta)["defects"][0]
        self.assertAlmostEqual(f["x_mm"], d["x_mm"], delta=1.5)
        self.assertAlmostEqual(f["y_mm"], d["y_mm"], delta=1.5)

    def test_too_few_dies_is_an_error(self):
        g, meta = clean_map()
        with self.assertRaises(ValueError):
            W.detect_array(np.zeros_like(g), meta)


class Quadrats(unittest.TestCase):
    def test_matches_die_value_layout(self):
        dies = die_layout()
        x = R.from_quadrats(dies, {(1, 0, 1): 1.0}, 2)
        r, c = R.die_slot(dies[0])
        self.assertEqual(x[0, r * 6 + 0, c * 6 + 3], 1.0)
        self.assertEqual(x[0, r * 6 + 3, c * 6 + 3], 0.0)
        self.assertEqual(x.shape, (3, R.SIZE, R.SIZE))


if __name__ == "__main__":
    unittest.main()
