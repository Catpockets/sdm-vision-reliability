import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import cifar100_subsets as subsets


class SubsetProtocolTests(unittest.TestCase):
    def test_five_reproducible_draws_from_all_classes(self):
        names = [f"class-{i}" for i in range(100)]
        plan = subsets.experiment_plan(names, smoke=False)
        self.assertEqual(plan, subsets.experiment_plan(names, smoke=False))
        self.assertEqual(plan["draws"], subsets.experiment_plan(names, smoke=True)["draws"])
        self.assertEqual(plan["per_class"], {"train": 375, "standard": 25, "enlarged": 100, "test": 100})
        selections = [tuple(draw["class_ids"]) for draw in plan["draws"]]
        self.assertEqual(len(set(selections)), 5)
        for selection, draw in zip(selections, plan["draws"]):
            self.assertEqual(len(set(selection)), 10)
            self.assertEqual(draw["class_names"], [names[i] for i in selection])
        self.assertGreater(max(map(max, selections)), 90)

    def test_disjoint_balanced_splits_and_consistent_overlap(self):
        labels = np.tile(np.arange(100), 500)
        class_ids = [3, 19, 65]
        split = subsets.split_training(labels, class_ids, train_per_class=375)
        self.assertEqual(len(set(split["train"]) & set(split["enlarged"])), 0)
        self.assertLessEqual(set(split["standard"]), set(split["enlarged"]))
        for name, size in (("train", 375), ("standard", 25), ("enlarged", 100)):
            np.testing.assert_array_equal(np.bincount(subsets.remap_labels(labels[split[name]], class_ids)), [size] * 3)
            other = subsets.split_training(labels, [3, 25], train_per_class=375)[name]
            np.testing.assert_array_equal(other[labels[other] == 3], split[name][labels[split[name]] == 3])

    def test_label_remapping_rejects_excluded_and_invalid_labels(self):
        np.testing.assert_array_equal(subsets.remap_labels([82, 2, 19, 2], [2, 19, 82]), [2, 0, 1, 0])
        for invalid in (-1, 3, 100):
            with self.assertRaises(ValueError):
                subsets.remap_labels([invalid], [2, 19, 82])

    def test_ties_and_empty_admissions(self):
        scores = np.full(10, 0.8)
        correct = np.arange(10) < 9
        self.assertIsNone(subsets.threshold_for_target(scores, correct, min_admitted=1))
        masks = subsets.admission_mask(scores, None)
        result = subsets.selective_statistics(np.zeros(10), np.zeros(10), masks)
        self.assertEqual(result["admitted"], 0)
        self.assertEqual(result["coverage"], 0)
        self.assertTrue(np.isnan(result["accepted_accuracy"]))

    def test_exact_pixel_matches_are_excluded_in_all_method_denominators(self):
        training = np.array([[11, 22], [33, 44]], dtype=np.uint8)
        held_out = np.array([[33, 44], [33, 45], [22, 11]], dtype=np.uint8)
        seen = subsets.seen_test_pixels(training, held_out)
        np.testing.assert_array_equal(seen, [True, False, False])

        # The excluded item is a correct admission. The remaining admitted item is wrong.
        result = subsets.summarize_admissions(
            np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([True, True, False]), seen
        )
        self.assertEqual(result["official_examples"], 3)
        self.assertEqual(result["excluded_seen_pixels"], 1)
        self.assertEqual(result["examples"], 2)
        self.assertEqual(result["admitted"], 1)
        self.assertEqual(result["correct_admitted"], 0)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["accepted_accuracy"], 0.)
        only_seen_admitted = subsets.summarize_admissions(
            np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([True, False, False]), seen
        )
        self.assertEqual(only_seen_admitted["admitted"], 0)
        self.assertIsNone(only_seen_admitted["accepted_accuracy"])

    def test_sdm_region_fits_actual_width(self):
        region_for = subsets.metric_functions()["fit_high_reliability_region"]
        region = region_for(np.eye(10), np.full(10, 2.0), np.arange(10))
        self.assertTrue(region["found"])
        self.assertEqual(len(region["thresholds"]), 10)
        absent = region_for(np.eye(10)[:-1], np.full(9, 2.0), np.arange(9))
        self.assertFalse(absent["found"])
        self.assertEqual(len(absent["thresholds"]), 10)

    def test_aggregation_keeps_all_draws_and_separates_modes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "smoke"
            rows = [
                {"track": track, "method": method, "admitted": 0, "accepted_accuracy": None}
                for track in subsets.TRACKS for method in subsets.METHODS
            ]
            for draw, status in enumerate(
                [{"status": "complete", "rows": rows}, {"status": "failed", "reason": "example"}], start=1,
            ):
                folder = root / f"draw-{draw}"
                folder.mkdir(parents=True)
                (folder / "status.json").write_text(json.dumps(status))
            report = subsets.summary_rows(temp)
            self.assertEqual([row["status"] for row in report], ["complete"] * 6 + ["failed"] + ["missing"] * 3)
            self.assertEqual(report[0]["admitted"], 0)
            self.assertIsNone(report[0]["accepted_accuracy"])
            self.assertEqual(report[6]["reason"], "example")
            self.assertEqual([row["status"] for row in subsets.summary_rows(temp, smoke=False)], ["missing"] * 5)


if __name__ == "__main__":
    unittest.main()
