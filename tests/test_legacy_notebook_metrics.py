"""Exercise older notebook definitions without training, downloads or SDM scoring."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
FILES = sorted(p for p in (ROOT / 'notebooks').glob('*.ipynb')
               if p.name[:2] in {'04', '05', '06', '07', '08', '09', '10'}
               and 'sdm_calibration_comparison' not in p.name)


def definitions(path):
    notebook = json.loads(path.read_text())
    ns = {'np': np, 'pd': pd, 'torch': torch, 'F': F, 'device': 'cpu', 'TARGET_ALPHA': .95}
    wanted = {'wilson_interval', 'selective_curve', 'choose_threshold', 'coverage_accuracy_curve', 'aurc',
              'region_blockers', 'region_ladder', 'top_alpha', 'deployment_metrics', 'sdm_rows', 'with_interval', 'headline', 'calibration_bins'}
    for c in notebook['cells']:
        source = ''.join(c['source'])
        if c['cell_type'] != 'code' or source.lstrip().startswith('%'): continue
        for node in ast.parse(source).body:
            if (isinstance(node, ast.ImportFrom) and node.module == 'sdm_vision.metrics') or (isinstance(node, ast.FunctionDef) and node.name in wanted):
                exec(compile(ast.Module(body=[node], type_ignores=[]), path.name, 'exec'), ns)
    return notebook, ns


class LegacyNotebookMetricsTest(unittest.TestCase):
    def test_corrected_thresholds_retain_whole_score_groups(self):
        for path in FILES:
            with self.subTest(notebook=path.name):
                _, ns = definitions(path)
                for correct in [np.array([True] * 18 + [False] * 2), np.array([False] * 2 + [True] * 18)]:
                    self.assertIsNone(ns['choose_threshold'](np.full(20, .9), correct, .95))
                    self.assertIsNone(ns['choose_threshold'](np.full(20, .9), correct, .95, True))
                self.assertIsNone(ns['choose_threshold'](np.array([]), np.array([], dtype=bool), .95))
                if 'selective_curve' in ns:
                    _, counts, hits = ns['selective_curve'](np.full(20, .9), correct)
                    np.testing.assert_array_equal(counts, [20]); np.testing.assert_array_equal(hits, [18])
                if 'aurc' in ns:
                    for correct in [np.array([True, False, True, False]), np.array([False, True, False, True])]:
                        self.assertAlmostEqual(ns['aurc'](np.full(4, .5), correct), .5)

    def test_selected_class_deployment_handles_zero_admissions(self):
        path = next(p for p in FILES if p.name.startswith('07_'))
        _, ns = definitions(path);ns['y_val'] = np.array([0, 1])
        row = ns['deployment_metrics'](np.zeros(2, dtype=bool), np.zeros(2, dtype=int), np.ones(2), np.array([0, 1]))
        self.assertEqual(row['answered'], 0);self.assertEqual(row['selective accuracy'], '-')
        self.assertEqual(ns['top_alpha']({'artifact': SimpleNamespace(manifest={'regions': []})}), np.inf)

    def test_summary_matching_is_invariant_to_validation_admission_changes(self):
        path = next(p for p in FILES if p.name.startswith('10_'))
        _, ns = definitions(path)
        cal_conf = np.array([.95, .85, .75, .65])
        run = {'sdm': {'centroid_alpha': np.array([.95, 0]), 'is_ood': np.array([False, False])},
               'calibration_sdm': {'centroid_alpha': np.array([.95, .95, 0, 0])},
               'artifact': SimpleNamespace(manifest={'regions': [{'alpha': .95}]}),
               'validation': {'prediction': np.array([0, 0]), 'confidence': np.array([.9, .8]), 'correct': np.array([True, False])},
               'calibration': {'confidence': cal_conf}}
        prediction = np.array([0, 0]); labels = np.array([0, 1])
        first = ns['sdm_rows']('fixture', run, prediction, labels)
        run['sdm']['centroid_alpha'][:] = .95
        second = ns['sdm_rows']('fixture', run, prediction, labels)
        key = ('fixture', 'α ≥ 0.95')
        self.assertEqual(first[key]['calibration-matched softmax'], second[key]['calibration-matched softmax'])
        self.assertEqual(first[key]['calibration-matched softmax'], '100.0% (20.7% – 100.0%)')

    def test_no_legacy_quantile_matching_or_saved_outputs_remain(self):
        for path in FILES:
            notebook, _ = definitions(path)
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    self.assertFalse(cell['outputs']);self.assertIsNone(cell['execution_count'])
                    self.assertNotIn('np.quantile(', ''.join(cell['source']))

    def test_cached_fit_rejects_changed_calibration_membership_or_labels(self):
        from sdm_vision.sdm import validate_calibration_identity
        artifact = SimpleNamespace(calibration_rows=[{'id': 'train-1', 'label': 0}, {'id': 'train-2', 'label': 1}],
                                   manifest={'configuration': {'classNames': ['a', 'b']}})
        validate_calibration_identity(artifact, ['train-2', 'train-1'], [1, 0], ['a', 'b'])
        for ids, labels, names in [(['train-1'], [0], ['a', 'b']), (['train-1', 'train-2'], [1, 0], ['a', 'b']),
                                    (['train-1', 'train-2'], [0, 1], ['b', 'a'])]:
            with self.assertRaisesRegex(ValueError, 'calibration population'):
                validate_calibration_identity(artifact, ids, labels, names)

    def test_coarse_gallery_handles_no_admissions_and_small_pools(self):
        path = next(p for p in FILES if p.name.startswith('05_'))
        notebook = json.loads(path.read_text())
        source = ''.join(next(c for c in notebook['cells'] if c['cell_type'] == 'code' and 'admitted_ids =' in ''.join(c['source']))['source'])
        for alpha in [np.array([0., 0.]), np.array([.95, .95]), np.array([.95, 0.])]:
            ns = {'np': np, 'plt': plt, 'TARGET_ALPHA': .95,
                  'coarse': {'centroid_alpha': alpha, 'prediction': np.zeros(2, dtype=int),
                             'scores': [SimpleNamespace(nearest_support_matches=[]) for _ in range(2)]},
                  'data': {'train': {'index': np.arange(2)}, 'validation': {'index': np.arange(2)}},
                  'all_images': np.zeros((2, 3, 32, 32), dtype=np.uint8), 'group_names': ['a'], 'y_coarse': np.zeros(2, dtype=int)}
            try: exec(compile(source, 'coarse-gallery', 'exec'), ns)
            finally: plt.close('all')
            self.assertEqual(len(ns['admitted_ids']), (alpha >= .95).sum())

    def test_empty_calibration_support_is_reported_in_all_sdm_notebooks(self):
        artifact = SimpleNamespace(calibration_rows=[{'qPrime': 0.}], manifest={'configuration': {'oodLimit': 0}})
        for path in FILES:
            if path.name[:2] not in {'05', '06', '07', '09'}: continue
            _, ns = definitions(path)
            table = ns['region_blockers'](artifact, 2)
            self.assertEqual(table['labels with no images left'].iloc[0], 2)
            self.assertEqual(table['labels with too many misses'].iloc[0], 0)


if __name__ == '__main__': unittest.main()
