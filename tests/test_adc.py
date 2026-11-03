"""ADC 엔진 회귀 테스트. 실행: ./venv/bin/python -m unittest tests.test_adc -v

합성 시나리오로 동작을 고정하는 테스트이며, 실제 웨이퍼 분류 정확도를 뜻하지 않는다.
"""
import unittest

from adc import belief, engine, scenarios, scoring
from wafer_model import WAFER_MODEL, die_layout

DIES = scoring.annotate(die_layout())
EXPECT = {
    "normal": [], "center": ["Center"], "donut": ["Donut"], "edge_ring": ["Edge-Ring"],
    "edge_loc": ["Edge-Loc"], "local": ["Local"], "random": ["Random"], "scratch": ["Scratch"],
    "near_full": ["Near-full"], "donut_scratch": ["Donut", "Scratch"], "ring_random": ["Edge-Ring", "Random"],
}


def run(name, answers=None):
    return engine.analyze(DIES, WAFER_MODEL, scenarios.generate(name, DIES), answers)


class Classification(unittest.TestCase):
    def test_all_scenarios(self):
        for name, labels in EXPECT.items():
            with self.subTest(name):
                self.assertEqual(run(name)["labels"], labels)

    def test_zone_sizes(self):
        self.assertEqual([sum(d["zone"] == z for d in DIES) for z in scoring.ZONES], [4, 20, 20])


class Explanation(unittest.TestCase):
    def test_additive_contributions_sum_to_logit(self):
        x = run("donut")["xai"]
        total = x["bias"] + sum(c["impact"] for c in x["contributions"])
        self.assertAlmostEqual(total, x["logit"], places=2)

    def test_counterfactual_flips_decision(self):
        defects = scenarios.generate("donut", DIES)
        cf = run("donut")["xai"]["counterfactual"]
        self.assertTrue(cf["found"])
        rest = {k: v for k, v in defects.items() if k not in cf["dies"]}
        self.assertLess(scoring.predict(DIES, rest)["Donut"]["score"], 0.5)


class ActiveInference(unittest.TestCase):
    def test_answers_change_top_cause(self):
        before = run("donut")["belief"]["hypotheses"][0]["id"]
        after = run("donut", [{"check": "mfc_log", "answer": "no"}, {"check": "rf_log", "answer": "yes"}])
        self.assertEqual(before, "gas_flow")
        self.assertEqual(after["belief"]["hypotheses"][0]["id"], "plasma")
        self.assertTrue(after["belief"]["converged"])

    def test_probabilities_sum_to_one(self):
        d = {"Donut": 0.9}
        self.assertAlmostEqual(sum(belief.prior_from_evidence(d, {"stain": 1.0}).values()), 1.0)

    def test_unknown_answer_changes_nothing(self):
        a = run("donut")["belief"]["hypotheses"]
        b = run("donut", [{"check": "mfc_log", "answer": "unknown"}])["belief"]["hypotheses"]
        self.assertEqual([h["p"] for h in a], [h["p"] for h in b])

    def test_normal_wafer_has_no_hypothesis(self):
        self.assertEqual(run("normal")["belief"]["hypotheses"], [])


class ScanPath(unittest.TestCase):
    """실측 스캔 경로: 검출 구역 판정 → 학습 모델 판정 → 가림·반사실 → 원인 추론.
    입력은 시연용 인쇄 배치(다이 4등분 구역 중앙에 5 mm 파티클)와 같은 구역 판정이다."""

    @staticmethod
    def quadrats(inside):
        out = {}
        for d in DIES:
            for qr in (0, 1):
                for qc in (0, 1):
                    x, y = d["x_mm"] + (qc - 0.5) * 16, d["y_mm"] - (qr - 0.5) * 16
                    if inside(x, y):
                        out[(d["id"], qr, qc)] = 1.0
        return out

    def analyze(self, q):
        from adc import cnn
        if not cnn.available():
            self.skipTest("학습 모델 파일 없음")
        defects = {}
        for (i, _, _) in q:
            defects.setdefault(i, {"particle": 0, "scratch": 0, "stain": 0})["particle"] += 1
        model = cnn.Model(DIES, q, scoring.RuleModel(DIES, defects))
        return engine.analyze(DIES, WAFER_MODEL, defects, None, None, model)

    def test_center_layout(self):
        r = self.analyze(self.quadrats(lambda x, y: (x * x + y * y) ** 0.5 < 30))
        self.assertEqual(r["labels"], ["Center"])
        self.assertEqual(r["model"]["kind"], "cnn")
        self.assertTrue(r["xai"]["counterfactual"]["found"])
        self.assertTrue(r["belief"]["hypotheses"])
        self.assertIsNotNone(r["xai"]["rule"])

    def test_donut_layout(self):
        r = self.analyze(self.quadrats(lambda x, y: 45 < (x * x + y * y) ** 0.5 < 75))
        self.assertEqual(r["labels"], ["Donut"])
        self.assertTrue(r["belief"]["checks"])


if __name__ == "__main__":
    unittest.main()
