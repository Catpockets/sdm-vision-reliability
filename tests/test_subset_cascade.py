import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_subset_cascade import (
    admission_mask, confidence_at_budget, evaluate, indexes_for_ids, summarize, threshold_at_budget, verify_file,
)


class CascadeAuditTests(unittest.TestCase):
    def test_join_aligns_permuted_rows(self):
        source = np.array([44, 2, 13, 0])
        np.testing.assert_array_equal(indexes_for_ids(source, [0, 44, 13]), [3, 0, 2])

    def test_join_rejects_duplicate_missing_and_noninteger_ids(self):
        for source, wanted in (([1, 1], [1]), ([1, 3], [1, 1]), ([3, 1], [2]),
                               ([3, 1], [4]), ([3., 1.], [1])):
            with self.subTest(source=source, wanted=wanted), self.assertRaises(ValueError):
                indexes_for_ids(source, wanted)

    def test_ties_share_the_same_decision(self):
        confidence = [0.5, 0.9, 0.5, 0.1]
        np.testing.assert_array_equal(confidence_at_budget(confidence, 2), [False, True, False, False])
        np.testing.assert_array_equal(confidence_at_budget(confidence, 3), [True, True, True, False])

    def test_zero_and_full_budget(self):
        confidence = [0.2, 0.2]
        np.testing.assert_array_equal(confidence_at_budget(confidence, 0), [False, False])
        np.testing.assert_array_equal(confidence_at_budget(confidence, 2), [True, True])
        with self.assertRaises(ValueError):
            confidence_at_budget([0.2, np.nan], 1)

    def test_frozen_calibration_cutoff_allows_held_out_budget_drift(self):
        cutoff = threshold_at_budget([0.3, 0.5, 0.7, 0.9], 2)
        self.assertEqual(cutoff, 0.7)
        np.testing.assert_array_equal(admission_mask([0.6, 0.65, 0.8, 0.55], cutoff), [False, False, True, False])
        self.assertIsNone(threshold_at_budget([0.3, 0.5], 0))

    def test_calibration_finishes_before_opening_test_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = {"mode": "benchmark", "draws": [{"seed": seed} for seed in range(5)]}
            payload = json.dumps(plan).encode()
            (root / "plan.json").write_bytes(payload)
            (root / "verified-results.json").write_text(json.dumps({
                "draws": plan["draws"], "sha256": {"plan.json": hashlib.sha256(payload).hexdigest()},
            }))
            with patch("torchvision.datasets.CIFAR100") as dataset, patch(
                    "evaluate_subset_cascade.freeze_calibration_budgets", side_effect=ValueError("calibration blocked"),
            ) as freeze:
                with self.assertRaisesRegex(ValueError, "calibration blocked"):
                    evaluate(root, root, root, root)
                dataset.assert_called_once_with(root, train=True, download=False)
                freeze.assert_called_once()

    def test_error_changes_reconcile_with_fallback_only(self):
        primary = np.array([0, 0, 1, 1])
        fallback = np.array([1, 0, 0, 1])
        truth = np.array([0, 1, 0, 1])
        result = summarize(primary, fallback, truth, np.array([True, False, True, False]))
        self.assertEqual(result["primary_calls"], 2)
        self.assertEqual(result["fallback_calls"], 2)
        self.assertEqual(result["gains_vs_fallback_only"], 1)
        self.assertEqual(result["losses_vs_fallback_only"], 1)
        self.assertEqual(result["correct"], 2)

    def test_digest_detects_replaced_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "data"
            file.write_bytes(b"saved")
            expected = hashlib.sha256(b"saved").hexdigest()
            self.assertEqual(verify_file(file, expected), expected)
            file.write_bytes(b"replaced")
            with self.assertRaises(ValueError):
                verify_file(file, expected)


if __name__ == "__main__":
    unittest.main()
