"""Numerical checks for notebook 04 without training, torch, or dataset downloads.

Run: python -m unittest discover -s tests -p 'test_notebook04_metrics.py'
"""
import ast
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/04_cifar100_sdm_calibration_comparison.ipynb'


def notebook_functions():
    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {'np': np, 'pd': pd, 'TARGET_ALPHA': .95}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        compile(source, cell['id'], 'exec')
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if cell['id'] in {'temperature-scaling-fit', 'classification-metric-functions', 'recall-coverage-plot', '7b7ffceb'}:
            exec(compile(ast.Module(body=functions, type_ignores=[]), cell['id'], 'exec'), namespace)
    return notebook, namespace


class NotebookMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook, cls.functions = notebook_functions()

    def test_temperature_is_frozen_before_test_data(self):
        ids = [cell['id'] for cell in self.notebook['cells']]
        self.assertLess(ids.index('temperature-scaling-fit'), ids.index('85ef06d2'))
        fit_cell = next(c for c in self.notebook['cells'] if c['id'] == 'temperature-scaling-fit')
        tree = ast.parse(''.join(fit_cell['source']))
        referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertTrue({'test', 'test_base', 'test_loader'}.isdisjoint(referenced))

    def test_analytic_temperature_optimum(self):
        # softmax([2, 0] / T)[0] must equal the empirical 75% correctness.
        logits = np.tile([2., 0.], (100, 1))
        labels = np.r_[np.zeros(75, dtype=int), np.ones(25, dtype=int)]
        result = self.functions['fit_temperature'](logits, labels)
        np.testing.assert_allclose(result['temperature'], 2 / np.log(3), rtol=1e-10)
        self.assertLess(result['calibration_nll_after'], result['calibration_nll_before'])
        self.assertFalse(result['at_search_bound'])

    def test_temperature_bounds_and_uninformative_logits(self):
        fit = self.functions['fit_temperature']
        logits = np.tile([2., 0.], (100, 1))
        for label, expected in [(0, .01), (1, 100.)]:
            result = fit(logits, np.full(100, label, dtype=int))
            np.testing.assert_allclose(result['temperature'], expected)
            self.assertTrue(result['at_search_bound'])
        self.assertEqual(fit(np.zeros((4, 3)), np.array([0, 1, 2, 0]))['temperature'], 1)
        with self.assertRaises(ValueError):
            fit(np.zeros((0, 3)), np.array([], dtype=int))

    def test_temperature_numerics_and_predictions(self):
        probabilities = self.functions['temperature_probabilities']
        logits = np.array([[1e4, 0, -1e4], [1, 3, -2]])
        for temperature in [.01, .5, 1, 2, 100]:
            result = probabilities(logits, temperature)
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_allclose(result.sum(axis=1), 1)
            np.testing.assert_array_equal(result.argmax(axis=1), [0, 1])
        for invalid in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                probabilities(logits, invalid)

    def test_precision_recall_class_counts(self):
        labels = np.array([0, 0, 0, 1, 1, 2])
        predictions = np.array([0, 1, 1, 1, 0, 1])
        classes, summary = self.functions['classification_metrics'](labels, predictions, 4)
        np.testing.assert_allclose(classes.precision, [.5, .25, 0, 0])
        np.testing.assert_allclose(classes.recall[:3], [1 / 3, .5, 0])
        self.assertTrue(np.isnan(classes.recall.iloc[3]))
        np.testing.assert_allclose(
            [summary[key] for key in ['macro_precision', 'macro_recall', 'micro_precision', 'micro_recall']],
            [.25, 5 / 18, 1 / 3, 1 / 3],
        )

    def test_abstention_and_no_admissions(self):
        metrics = self.functions['classification_metrics']
        labels = np.array([0, 0, 0, 1, 1, 2])
        predictions = np.array([0, 1, 1, 1, 0, 1])
        admitted = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
        _, summary = metrics(labels, predictions, 4, admitted)
        np.testing.assert_allclose(
            [summary[key] for key in ['coverage', 'micro_precision', 'micro_recall']],
            [1 / 3, .5, 1 / 6],
        )
        _, summary = metrics(labels, predictions, 4, np.zeros(6, dtype=bool))
        self.assertTrue(np.isnan(summary['micro_precision']))
        self.assertEqual(summary['micro_recall'], 0)
        self.assertEqual(summary['macro_recall'], 0)
        _, empty = metrics(np.array([], dtype=int), np.array([], dtype=int), 4)
        self.assertTrue(np.isnan(empty['macro_recall']))

    def test_recall_coverage_ties_and_imbalanced_classes(self):
        curve = self.functions['macro_recall_coverage']
        metrics = self.functions['classification_metrics']
        labels = np.array([0, 0, 0, 1, 1, 2])
        predictions = np.array([0, 1, 1, 1, 0, 1])
        confidence = np.array([.9, .9, .5, .5, .1, .1])
        coverage, recall = curve(confidence, labels, predictions, 4)
        np.testing.assert_allclose(coverage, [0, 1 / 3, 2 / 3, 1])
        for threshold, actual in zip([.9, .5, .1], recall[1:]):
            expected = metrics(labels, predictions, 4, confidence >= threshold)[1]['macro_recall']
            np.testing.assert_allclose(actual, expected)
        permutation = np.array([1, 0, 3, 2, 5, 4])
        x, y = curve(confidence[permutation], labels[permutation], predictions[permutation], 4)
        np.testing.assert_allclose(x, coverage)
        np.testing.assert_allclose(y, recall)
        x, y = curve(np.array([]), np.array([], dtype=int), np.array([], dtype=int), 4)
        np.testing.assert_array_equal(x, [0.])
        np.testing.assert_array_equal(y, [0.])

    def test_uniform_confidence_metrics_are_permutation_invariant(self):
        metrics = self.functions['probability_metrics']
        probabilities = np.full((4, 2), .5)
        predictions = np.zeros(4, dtype=int)
        for labels in [np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])]:
            result = metrics(probabilities, labels, predictions)
            np.testing.assert_allclose([result['error_auroc'], result['error_auprc'], result['aurc']], [.5, .5, .5])
            self.assertEqual(result['coverage_at_95pct_accuracy'], 0)
            np.testing.assert_array_equal(result['curve_coverage'], [0, 1])
            for budget in [10, 25, 50, 75]:
                self.assertTrue(np.isnan(result[f'accuracy_at_{budget}pct_coverage']))
                self.assertEqual(result[f'actual_coverage_at_{budget}pct_budget'], 0)
            self.assertEqual(result['accuracy_at_100pct_coverage'], .5)

    def test_risk_metrics_use_attainable_thresholds_and_actual_budget_coverage(self):
        metrics = self.functions['probability_metrics']
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        predictions = np.zeros(8, dtype=int)
        probabilities = np.array([[.9, .1]] * 4 + [[.7, .3]] * 4)
        result = metrics(probabilities, labels, predictions)
        np.testing.assert_array_equal(result['curve_coverage'], [0, .5, 1])
        np.testing.assert_allclose(result['curve_risk'], [0, 0, 3 / 8])
        self.assertEqual(result['aurc'], 3 / 16)
        self.assertEqual(result['coverage_at_95pct_accuracy'], .5)
        self.assertEqual(result['accuracy_at_75pct_coverage'], 1)
        self.assertEqual(result['actual_coverage_at_75pct_budget'], .5)
        permutation = np.array([3, 0, 2, 1, 7, 5, 4, 6])
        permuted = metrics(probabilities[permutation], labels[permutation], predictions[permutation])
        for key in result:
            np.testing.assert_allclose(result[key], permuted[key], equal_nan=True, err_msg=key)

    def test_binary_ranking_ties_and_single_outcome(self):
        ranking = self.functions['binary_ranking_metrics']
        # First group: one positive/one negative; last group: one positive.
        auroc, average_precision = ranking([True, False, True], [.9, .9, .1])
        np.testing.assert_allclose([auroc, average_precision], [.25, 7 / 12])
        for labels in [[True, True], [False, False], []]:
            auroc, average_precision = ranking(labels, np.ones(len(labels)))
            self.assertTrue(np.isnan(auroc))
            if any(labels):
                self.assertEqual(average_precision, 1)
            else:
                self.assertTrue(np.isnan(average_precision))


if __name__ == '__main__':
    unittest.main()
