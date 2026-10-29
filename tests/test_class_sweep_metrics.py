"""Class-sweep numerical regressions; no training or downloads."""
import importlib.util
from pathlib import Path
import sys
import unittest
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sdm_vision import metrics


class ClassSweepMetricsTest(unittest.TestCase):
    def test_constant_confidence_cannot_split_ties(self):
        score = np.full(20, .9)
        for correct in [np.array([True] * 18 + [False] * 2), np.array([False] * 2 + [True] * 18)]:
            for conservative in [False, True]:
                self.assertIsNone(metrics.choose_threshold(score, correct, .95, conservative))
            self.assertEqual(metrics.choose_threshold(score, correct, .85), .9)
            coverage, accuracy = metrics.coverage_accuracy_curve(score, correct)
            np.testing.assert_allclose(coverage, [1.])
            np.testing.assert_allclose(accuracy, [.9])
            self.assertAlmostEqual(metrics.aurc(score, correct), .1)

    def test_grouped_aurc_uses_interval_widths(self):
        score = np.array([.9, .9, .7, .7, .7, .7, .7, .7])
        correct = np.array([True, True, True, True, True, False, False, False])
        self.assertAlmostEqual(metrics.aurc(score, correct), .75 * 3 / 8)
        order = [1, 0, 7, 6, 5, 4, 3, 2]
        self.assertAlmostEqual(metrics.aurc(score[order], correct[order]), .75 * 3 / 8)
        self.assertGreaterEqual(metrics.e_aurc(score, correct), 0)
        self.assertAlmostEqual(metrics.e_aurc(np.array([.9, .8, .2, .1]), np.array([True, True, False, False])), 0)

    def test_empty_metrics_and_valid_wilson_threshold(self):
        score = np.array([]); correct = np.array([], dtype=bool)
        self.assertIsNone(metrics.choose_threshold(score, correct, .95))
        self.assertTrue(np.isnan(metrics.aurc(score, correct)))
        score = np.r_[np.full(100, .99), np.full(20, .5)]
        correct = np.r_[np.ones(100, dtype=bool), np.zeros(20, dtype=bool)]
        for conservative in [False, True]:
            self.assertEqual(metrics.choose_threshold(score, correct, .95, conservative), .99)

    def test_coverage_threshold_never_splits_ties_or_exceeds_budget(self):
        score = np.array([.9, .9, .7, .7])
        self.assertIsNone(metrics.threshold_for_coverage(score, 0))
        self.assertIsNone(metrics.threshold_for_coverage(score, .25))
        self.assertEqual(metrics.threshold_for_coverage(score, .5), .9)
        self.assertEqual(metrics.threshold_for_coverage(score, .75), .9)
        self.assertEqual(metrics.threshold_for_coverage(score, 1), .7)

    def test_runner_freezes_matching_on_calibration_and_exports_its_scores(self):
        spec = importlib.util.spec_from_file_location('sweep', ROOT / 'experiments/class_sweep.py')
        sweep = importlib.util.module_from_spec(spec); spec.loader.exec_module(sweep)
        labels = np.repeat(np.arange(10), 4)
        images = np.arange(40).reshape(-1, 1)
        split = {'train': np.arange(0, 40, 4), 'calibration': np.sort(np.r_[np.arange(1, 40, 4), np.arange(2, 40, 4)]),
                 'validation': np.arange(3, 40, 4)}
        confidences = {1: .95, 2: .85, 5: .75, 6: .65}
        def predict(model, pixels, values):
            conf = np.array([confidences.get(int(i), .8) for i in values[:, 0]])
            return torch.from_numpy(np.log(np.stack([conf, 1 - conf], axis=1))), values
        validation_admissions = [1, 0]
        def score(artifact, embeddings, device):
            ids = embeddings[:, 0]
            if np.array_equal(ids, [1, 2, 5, 6]):
                alpha = np.array([.95, .95, 0, 0])
            elif np.array_equal(ids, [3, 7]):
                alpha = .95 * np.array(validation_admissions)
            else:
                alpha = np.zeros(len(ids))
            return {'prediction': np.zeros(len(ids), dtype=int), 'probability': np.full(len(ids), .8),
                    'centroid_alpha': alpha, 'lower_alpha': alpha, 'is_ood': np.zeros(len(ids), dtype=bool)}
        artifact = SimpleNamespace(manifest={'regions': [{'alpha': .95}]})
        args = SimpleNamespace(steps=None, epochs=1, sdm_epochs=1)
        with patch.object(sweep.data, 'class_order', return_value=np.arange(10)), \
             patch.object(sweep.train, 'train_resnet', return_value=(None, None)), \
             patch.object(sweep.train, 'predict', side_effect=predict), \
             patch.object(sweep.train, 'fit_temperature', return_value=1.), \
             patch.object(sweep.sdm, 'fit_sdm', return_value=artifact), \
             patch.object(sweep.sdm, 'ladder_is_consistent', return_value=True), \
             patch.object(sweep.sdm, 'score', side_effect=score), \
             patch.object(sweep.metrics, 'paired_bootstrap', return_value=[np.nan, np.nan]):
            first, scores = sweep.run_one('cifar10', images, labels, list(map(str, range(10))), split, 0, 2, args, 'cpu')
            validation_admissions[:] = [1, 1]
            second, _ = sweep.run_one('cifar10', images, labels, list(map(str, range(10))), split, 0, 2, args, 'cpu')
        for result in [first, second]:
            self.assertEqual(result['matched_softmax_threshold'], .85)
            self.assertEqual(result['matched_calibration_sdm_coverage'], .5)
            self.assertEqual(result['matched_calibration_softmax_coverage'], .5)
            self.assertEqual(result['evaluation_version'], metrics.EVALUATION_VERSION)
        self.assertIn('sdm_cal_centroid_alpha', scores)
        np.testing.assert_allclose(scores['sdm_cal_centroid_alpha'], [.95, .95, 0, 0])

    def test_cached_results_require_current_evaluation_version(self):
        spec = importlib.util.spec_from_file_location('sweep_cache', ROOT / 'experiments/class_sweep.py')
        sweep = importlib.util.module_from_spec(spec); spec.loader.exec_module(sweep)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cifar10/chain0_K2/metrics.json'
            path.parent.mkdir(parents=True)
            argv = ['class_sweep.py', '--dataset', 'cifar10', '--chains', '1', '--k', '2', '--out', tmp]
            with patch.object(sys, 'argv', argv), \
                 patch.dict(sweep.data.LOADERS, {'cifar10': lambda _: (np.zeros((1, 1)), np.array([0]), list(map(str, range(10))))}), \
                 patch.object(sweep.data, 'fixed_split', return_value={}), \
                 patch.object(sweep.train, 'pick_device', return_value='cpu'), \
                 patch.object(sweep, 'run_one') as run:
                original = json.dumps({'legacy': True})
                path.write_text(original)
                with self.assertRaisesRegex(ValueError, 'legacy evaluation'):
                    sweep.main()
                self.assertEqual(path.read_text(), original)
                path.write_text(json.dumps({'evaluation_version': metrics.EVALUATION_VERSION}))
                sweep.main()
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
