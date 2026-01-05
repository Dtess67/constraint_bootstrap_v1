import unittest
from pathlib import Path
from constraint_bootstrap.run_metrics_v1 import compute_metrics, sig_to_set

class TestRunMetricsV1(unittest.TestCase):
    def test_sig_to_set(self):
        self.assertEqual(sig_to_set("[]"), set())
        self.assertEqual(sig_to_set("∅"), set())
        self.assertEqual(sig_to_set("[5]"), {5})
        self.assertEqual(sig_to_set("[5 7]"), {5, 7})
        self.assertEqual(sig_to_set("[7 5]"), {5, 7})

    def test_compute_metrics_basic(self):
        # 4 steps:
        # 1. pred=[], act=[] -> exact match, empty
        # 2. pred=[5], act=[5] -> exact match, non-empty
        # 3. pred=[], act=[7] -> miss, non-empty
        # 4. pred=[5], act=[7] -> miss, non-empty
        steps = [
            {"pred_sig": "[]", "act_sig": "[]", "err": "0.0"},
            {"pred_sig": "[5]", "act_sig": "[5]", "err": "0.0"},
            {"pred_sig": "[]", "act_sig": "[7]", "err": "2.0"},
            {"pred_sig": "[5]", "act_sig": "[7]", "err": "2.0"},
        ]
        m = compute_metrics(steps)
        
        # Overall exact: 2/4 = 0.5
        self.assertEqual(m["acc_exact"], 0.5)
        
        # Non-empty steps: 3 (steps 2,3,4)
        # Non-empty exact: 1 (step 2) -> 1/3 = 0.3333
        self.assertAlmostEqual(m["acc_nonempty_exact"], 1/3)
        
        # Silence baseline: act=[] is 1/4 = 0.25
        self.assertEqual(m["acc_always_silent"], 0.25)
        
        # Skill over silence: 0.5 - 0.25 = 0.25
        self.assertEqual(m["skill_over_silence"], 0.25)

    def test_bitwise_metrics(self):
        steps = [
            {"pred_sig": "[5]", "act_sig": "[5 7]"}, # TP 5, FN 7
            {"pred_sig": "[7]", "act_sig": "[5]"},   # FP 7, FN 5
            {"pred_sig": "[5 7]", "act_sig": "[7]"}, # TP 7, FP 5
        ]
        m = compute_metrics(steps)
        
        # Label 5:
        # Step 1: pred=T, act=T -> TP
        # Step 2: pred=F, act=T -> FN
        # Step 3: pred=T, act=F -> FP
        # TP=1, FP=1, FN=1 -> Prec=0.5, Rec=0.5, F1=0.5
        self.assertEqual(m["bitwise"]["5"]["f1"], 0.5)

        # Label 7:
        # Step 1: pred=F, act=T -> FN
        # Step 2: pred=T, act=F -> FP
        # Step 3: pred=T, act=T -> TP
        # TP=1, FP=1, FN=1 -> Prec=0.5, Rec=0.5, F1=0.5
        self.assertEqual(m["bitwise"]["7"]["f1"], 0.5)

if __name__ == "__main__":
    unittest.main()
