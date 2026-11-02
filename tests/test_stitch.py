"""웨이퍼 맵 합성 회귀 테스트. 실행: ./venv/bin/python -m unittest tests.test_stitch -v"""
import math
import unittest

import cv2
import numpy as np

import wafer_stitch as W


def synthetic_scan(step=30.0, n=3, s=4.0, rot_deg=88.0, size=(540, 960), q=0.3):
    """알려진 카메라 모델로 평면 장면을 촬영한 사진을 만든다."""
    h, w = size
    rng = np.random.default_rng(1)
    scene = cv2.GaussianBlur(rng.random((1400, 1400)).astype(np.float32), (0, 0), 3)
    scene = (scene - scene.min()) / np.ptp(scene)
    c, sn = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    A = s * np.array([[c, -sn], [sn, c]])
    Ainv = np.linalg.inv(-A)
    qn = q / (h * h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - w / 2, yy - h / 2
    ux, uy = W._undistort(dx, dy, qn)
    images, pos = [], []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            C = np.array([i * step, j * step])
            px = Ainv[0, 0] * ux + Ainv[0, 1] * uy + C[0]
            py = Ainv[1, 0] * ux + Ainv[1, 1] * uy + C[1]
            mx, my = (700 + px * 2).astype(np.float32), (700 - py * 2).astype(np.float32)
            images.append(cv2.remap(scene, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
            pos.append(C)
    return images, np.array(pos), A, qn


class LensModel(unittest.TestCase):
    def test_distort_inverts_undistort(self):
        x, y = np.array([120.0, -260.0, 40.0]), np.array([80.0, 170.0, -300.0])
        q = 0.4 / 540 ** 2
        ux, uy = W._undistort(x, y, q)
        rx, ry = W._distort(ux, uy, q)
        np.testing.assert_allclose(rx, x, atol=0.5)
        np.testing.assert_allclose(ry, y, atol=0.5)


class CameraEstimate(unittest.TestCase):
    def test_recovers_rotation_scale_and_distortion(self):
        images, pos, A, q = synthetic_scan()
        light = np.full(images[0].shape, 200, np.float32)
        far = [(i, j) for i in range(0, len(pos), 3) for j in range(0, len(pos), 3)
               if i < j and 20 < np.hypot(*(pos[i] - pos[j])) < 95][:50]
        start = 1.04 * A
        out = W.refine_camera(images, light, pos, start, far)
        scale = np.linalg.norm(out["A"][:, 0])
        self.assertAlmostEqual(scale / np.linalg.norm(A[:, 0]), 1.0, delta=0.01)
        self.assertGreater(out["ncc_after"], 0.8)
        self.assertAlmostEqual(out["q"] * 540 ** 2, 0.3, delta=0.08)


class WaferMeasure(unittest.TestCase):
    def test_circle_fit(self):
        t = np.deg2rad(np.arange(10, 90, 3))
        p = np.c_[300 + 500 * np.cos(t), 200 + 500 * np.sin(t)]
        cx, cy, r = W.fit_circle(p)
        self.assertAlmostEqual(cx, 300, places=3)
        self.assertAlmostEqual(cy, 200, places=3)
        self.assertAlmostEqual(r, 500, places=3)


if __name__ == "__main__":
    unittest.main()
