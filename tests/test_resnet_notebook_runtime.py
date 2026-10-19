"""Regression checks for the existing ResNet baseline and SDM notebooks."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def functions(filename):
    notebook = json.loads((ROOT / 'notebooks' / filename).read_text())
    namespace = {'np': np, 'pd': pd}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code' or ''.join(cell['source']).lstrip().startswith('%'):
            continue
        nodes = [n for n in ast.parse(''.join(cell['source'])).body
                 if isinstance(n, ast.FunctionDef) and n.name in
                 {'wilson_interval', 'selective_curve', 'choose_threshold', 'region_blockers'}]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), namespace)
    return notebook, namespace


class ResNetRuntimeTest(unittest.TestCase):
    def test_sdm_setup_uses_shared_requirements(self):
        notebook, _ = functions('03_sdm_resnet18.ipynb')
        install = [c for c in notebook['cells'] if '%pip install' in ''.join(c['source'])]
        self.assertEqual(''.join(install[0]['source']).strip(), '%pip install -r ../requirements.txt')

    def test_thresholds_cannot_split_equal_confidence_groups(self):
        for filename in ['02_cifar100_cnn_resnet18.ipynb', '03_sdm_resnet18.ipynb']:
            with self.subTest(notebook=filename):
                _, namespace = functions(filename)
                correct = np.array([True] * 18 + [False] * 2)
                scores = np.full(20, .9)
                choose = namespace['choose_threshold']
                self.assertIsNone(choose(scores, correct, .95))
                self.assertIsNone(choose(scores[::-1], correct[::-1], .95))
                self.assertIsNone(choose(np.array([]), np.array([], dtype=bool), .95))
                self.assertEqual(choose(scores, correct, .85), .9)
                # An attainable high group works for both ordinary and Wilson rules.
                scores = np.r_[np.full(100, .99), np.full(20, .5)]
                correct = np.r_[np.ones(100, dtype=bool), np.zeros(20, dtype=bool)]
                for conservative in [False, True]:
                    threshold = choose(scores, correct, .95, conservative=conservative)
                    self.assertEqual(threshold, .99)
                    self.assertEqual(correct[scores >= threshold].mean(), 1)

    def test_no_eligible_calibration_rows_have_explicit_blocker_counts(self):
        _, namespace = functions('03_sdm_resnet18.ipynb')
        artifact = SimpleNamespace(
            calibration_rows=[{'sdm': [.5, .5], 'qPrime': 0., 'label': i} for i in [0, 1, 0, 1]],
            manifest={'configuration': {'oodLimit': 0}},
        )
        result = namespace['region_blockers'](artifact, 2)
        self.assertTrue((result['labels with no images left'] == 2).all())
        self.assertTrue((result['labels with too many misses'] == 0).all())

    def test_exemplar_gallery_handles_fewer_than_three_cases(self):
        notebook, _ = functions('03_sdm_resnet18.ipynb')
        source = ''.join(next(c for c in notebook['cells'] if c.get('id') == 'cell-24')['source'])
        for alphas in [np.array([0., 0.]), np.array([.95, .95]), np.array([.95, 0.])]:
            namespace = {
                'np': np, 'plt': plt, 'TARGET_ALPHA': .95,
                'coarse': {'centroid_alpha': alphas, 'prediction': np.zeros(2, dtype=int),
                           'scores': [SimpleNamespace(nearest_support_matches=[]) for _ in range(2)]},
                'data': {'train': {'index': np.arange(2)}, 'validation': {'index': np.arange(2)}},
                'all_images': np.zeros((2, 3, 32, 32), dtype=np.uint8),
                'group_names': ['group'], 'y_coarse': np.zeros(2, dtype=int),
            }
            try:
                exec(compile(source, 'resnet-gallery', 'exec'), namespace)
            finally:
                plt.close('all')
            self.assertEqual(len(namespace['admitted_ids']), (alphas >= .95).sum())
            self.assertEqual(len(namespace['rejected_ids']), (alphas == 0).sum())


if __name__ == '__main__':
    unittest.main()
