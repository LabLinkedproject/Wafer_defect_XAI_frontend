"""끊김 대비 도구 시험. 실행: ./venv-train/bin/python -m unittest tests.test_durable -v"""
import os
import tempfile
import unittest

import torch

from training import durable


class Atomic(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "ck.pt")
        durable.EVENTS = os.path.join(self.dir.name, "events.log")

    def tearDown(self):
        self.dir.cleanup()

    def test_keeps_two_generations(self):
        durable.save_checkpoint({"v": 1}, self.path)
        durable.save_checkpoint({"v": 2}, self.path)
        self.assertEqual(durable.load_checkpoint(self.path)["v"], 2)
        self.assertEqual(torch.load(self.path + ".prev", weights_only=False)["v"], 1)

    def test_corrupt_latest_falls_back(self):
        durable.save_checkpoint({"v": 1}, self.path)
        durable.save_checkpoint({"v": 2}, self.path)
        with open(self.path, "wb") as f:
            f.write(b"\0" * 64)
        self.assertEqual(durable.load_checkpoint(self.path)["v"], 1)

    def test_truncated_latest_falls_back(self):
        durable.save_checkpoint({"v": 1}, self.path)
        durable.save_checkpoint({"v": torch.zeros(1000)}, self.path)
        with open(self.path, "r+b") as f:
            f.truncate(os.path.getsize(self.path) // 2)
        self.assertEqual(durable.load_checkpoint(self.path)["v"], 1)

    def test_missing_is_none(self):
        self.assertIsNone(durable.load_checkpoint(self.path))

    def test_interrupted_write_leaves_old_file(self):
        durable.write_json(self.path, {"v": 1})
        with open(self.path + ".tmp", "w") as f:
            f.write('{"v"')                              # 쓰는 도중 끊긴 상태
        with open(self.path) as f:
            self.assertIn('"v": 1', f.read())


if __name__ == "__main__":
    unittest.main()
