"""Protocol parity, dataset adapters, and synthetic end-to-end notebook checks."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from benchmark_datasets import DATASETS, dataset_transforms, load_dataset, stratified_splits, transpose_emnist

module_spec = importlib.util.spec_from_file_location('notebook_builder', ROOT / 'scripts/build_dataset_notebooks.py')
builder = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(builder)
PATHS = [ROOT / f'notebooks/{number}_{key}_sdm_calibration_comparison.ipynb' for number, key, *_ in builder.CATALOG]


def function_nodes(notebook):
    return {
        node.name: node
        for cell in notebook['cells'] if cell['cell_type'] == 'code'
        for node in ast.parse(''.join(cell['source'])).body if isinstance(node, ast.FunctionDef)
    }


class DatasetProtocolTest(unittest.TestCase):
    def test_generated_notebooks_schema_and_metric_parity(self):
        template = json.loads(builder.TEMPLATE.read_text())
        reference = function_nodes(template)
        metrics = ['classification_metrics', 'binary_ranking_metrics', 'probability_metrics',
                   'macro_recall_coverage', 'fit_temperature', 'temperature_probabilities',
                   'sdm_activation', 'rescaled_similarity', 'conservative_distance_quantile']
        for path, spec in zip(PATHS, builder.CATALOG):
            with self.subTest(dataset=spec[1]):
                notebook = json.loads(path.read_text())
                nbformat.validate(nbformat.from_dict(notebook))
                self.assertEqual(notebook, builder.build(template, *spec))
                nodes = function_nodes(notebook)
                for name in metrics:
                    self.assertEqual(ast.dump(nodes[name]), ast.dump(reference[name]), name)
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        compile(''.join(cell['source']), cell['id'], 'exec')
                        self.assertIsNone(cell['execution_count'])
                        self.assertEqual(cell['outputs'], [])
                ids = [cell['id'] for cell in notebook['cells']]
                self.assertLess(ids.index('temperature-scaling-fit'), ids.index('85ef06d2'))
                for cell in notebook['cells'][:ids.index('85ef06d2')]:
                    if cell['cell_type'] == 'code':
                        self.assertNotIn('train=False', ''.join(cell['source']))

    def test_full_stratification_preserves_imbalance_and_no_leakage(self):
        labels = np.repeat(np.arange(4), [200, 220, 250, 300])
        parts = stratified_splits(labels, 4)
        again = stratified_splits(labels, 4)
        for key in parts:
            np.testing.assert_array_equal(parts[key], again[key])
        np.testing.assert_array_equal(np.bincount(labels[parts['train']]), [75, 95, 125, 175])
        np.testing.assert_array_equal(np.bincount(labels[parts['standard']]), [25] * 4)
        np.testing.assert_array_equal(np.bincount(labels[parts['enlarged']]), [100] * 4)
        self.assertTrue(set(parts['standard']) <= set(parts['enlarged']))
        sets = [set(parts[key]) for key in ['train', 'validation', 'enlarged']]
        for i in range(3):
            for j in range(i): self.assertFalse(sets[i] & sets[j])
        self.assertEqual(set.union(*sets), set(range(len(labels))))
        with self.assertRaises(ValueError): stratified_splits(np.array([0, 0]), 2)

    def test_smoke_contains_every_class(self):
        for classes in [10, 47]:
            labels = np.repeat(np.arange(classes), 8)
            parts = stratified_splits(labels, classes, smoke=True)
            for key, count in [('train', 2), ('validation', 1), ('standard', 1), ('enlarged', 2)]:
                np.testing.assert_array_equal(np.bincount(labels[parts[key]]), [count] * classes)

    def test_preprocessing_orientation_channels_and_flips(self):
        torch.manual_seed(42)
        for spec in DATASETS.values():
            train, evaluation = dataset_transforms(spec, 32)
            image = Image.fromarray(np.arange(28 * 28, dtype=np.uint8).reshape(28, 28))
            if not spec.grayscale: image = image.convert('RGB')
            self.assertEqual(tuple(train(image).shape), (3, 32, 32))
            np.testing.assert_array_equal(evaluation(image), evaluation(image))
            self.assertEqual(tuple(evaluation(image).shape), (3, 32, 32))
            self.assertEqual(any(isinstance(t, transforms.RandomHorizontalFlip) for t in train.transforms), spec.key == 'cifar10')
        image = Image.fromarray(np.arange(12, dtype=np.uint8).reshape(3, 4))
        np.testing.assert_array_equal(np.asarray(transpose_emnist(image)), np.asarray(image).T)

    def test_dataset_loader_api_and_labels(self):
        for spec in DATASETS.values():
            for train in [True, False]:
                with self.subTest(dataset=spec.key, train=train):
                    fake = fake_dataset(spec, train=train)
                    with patch.object(datasets, spec.loader, return_value=fake) as factory:
                        result = load_dataset(spec, '/tmp/unused-data', train=train, transform=None)
                    options = factory.call_args.kwargs
                    self.assertEqual(options['root'], Path('/tmp/unused-data') / spec.key)
                    self.assertEqual(len(result.classes), spec.classes)
                    self.assertGreaterEqual(result.targets.min(), 0)
                    self.assertLess(result.targets.max(), spec.classes)
                    if spec.key == 'svhn':
                        self.assertEqual(options['split'], 'train' if train else 'test')
                        self.assertNotIn('train', options)
                        self.assertEqual(result.classes[0], '0')
                    else: self.assertEqual(options['train'], train)
                    if spec.key == 'emnist_balanced': self.assertEqual(options['split'], 'balanced')

    def test_streamed_sdm_neighbors_match_full_search(self):
        notebook = json.loads(PATHS[0].read_text())
        nodes = function_nodes(notebook)
        rng = np.random.default_rng(28)
        support = rng.normal(size=(12, 5)).astype('float32')
        query = rng.normal(size=(7, 5)).astype('float32')
        namespace = {'torch': torch, 'np': np, 'DEVICE': torch.device('cpu'),
                     'reference_embeddings': support, 'neighbors': 8,
                     'reference': {'targets': np.array([0] * 6 + [1] * 6), 'predictions': np.array([0] * 5 + [1] * 7)}}
        for name in ['exact_squared_l2_neighbors', 'consecutive_correct_match_count', 'similarity_distance_signals']:
            exec(compile(ast.Module(body=[nodes[name]], type_ignores=[]), name, 'exec'), namespace)
        predictions = np.array([0, 1, 0, 1, 0, 1, 0])
        distances, indices, q = namespace['similarity_distance_signals'](query, predictions, batch_size=2)
        brute_distances = ((query[:, None] - support[None]) ** 2).sum(axis=2)
        nearest = np.argsort(brute_distances, axis=1)[:, :8]
        expected_q = []
        for row, prediction in zip(nearest, predictions):
            count = 0
            for neighbor in row:
                if namespace['reference']['targets'][neighbor] != prediction or namespace['reference']['predictions'][neighbor] != prediction: break
                count += 1
            expected_q.append(count)
        np.testing.assert_array_equal(indices[:, 0], nearest[:, 0])
        np.testing.assert_allclose(distances[:, 0], brute_distances[np.arange(7), nearest[:, 0]], rtol=1e-5)
        np.testing.assert_array_equal(q, expected_q)


class FakeImages(Dataset):
    def __init__(self, spec, *, train=True, transform=None):
        self.spec, self.transform = spec, transform
        targets = np.repeat(np.arange(spec.classes), 8)
        if spec.key == 'svhn': self.labels = targets
        else:
            self.targets = targets
            self.classes = [f'class_{i}' for i in range(spec.classes)]
        self._targets = targets
        self.offset = 0 if train else 17

    def __len__(self): return len(self._targets)

    def __getitem__(self, index):
        # Independent test pixels; predictable class patterns keep fixtures deterministic.
        label = int(self._targets[index])
        pixels = np.full((32, 32), (label * 17 + index % 8 + self.offset) % 256, dtype=np.uint8)
        image = Image.fromarray(pixels)
        if not self.spec.grayscale: image = image.convert('RGB')
        return (self.transform(image) if self.transform else image), label


def fake_dataset(spec, *, train=True, transform=None, **kwargs):
    return FakeImages(spec, train=train, transform=transform)


class TinyConvNeXt(nn.Module):
    """Compatible feature/head interface for exercising the complete notebook path."""
    def __init__(self, **kwargs):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, stride=16), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Identity(), nn.Flatten(1), nn.Linear(4, 1000))

    def forward(self, inputs):
        return self.classifier(self.avgpool(self.features(inputs)))


class NotebookPipelineTest(unittest.TestCase):
    def test_all_six_synthetic_smoke_pipelines_and_exports(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix='sdm28-fixtures-') as directory:
            for path, spec in zip(PATHS, DATASETS.values()):
                with self.subTest(dataset=spec.key):
                    def factory(*, split=None, train=True, transform=None, **kwargs):
                        return fake_dataset(spec, train=train if spec.key != 'svhn' else split == 'train', transform=transform)
                    environment = {'SDM_RUN_MODE': 'smoke', 'SDM_DOWNLOAD': '1',
                                   'SDM_RUN_DIR': directory, 'SDM_DATA_DIR': directory,
                                   'SDM_FORCE_RETRAIN': '0'}
                    namespace = {'display': lambda *args: None, '__name__': '__main__'}
                    with patch.dict(os.environ, environment), patch.object(datasets, spec.loader, side_effect=factory), \
                         patch('torchvision.models.convnext_tiny', TinyConvNeXt), patch('torch.cuda.is_available', return_value=False), \
                         patch.object(plt, 'show', side_effect=lambda: plt.close('all')), contextlib.redirect_stdout(io.StringIO()):
                        notebook = json.loads(path.read_text())
                        for cell in notebook['cells']:
                            if cell['cell_type'] == 'code': exec(compile(''.join(cell['source']), cell['id'], 'exec'), namespace)
                    run = namespace['RUN_DIR']
                    self.assertIn(spec.key, run.parts)
                    self.assertEqual(len(namespace['metrics_table']), 5)
                    self.assertEqual(len(namespace['test']['targets']), spec.classes * 2)
                    self.assertEqual(len(namespace['class_metrics_table']), spec.classes)
                    self.assertEqual(namespace['metadata']['status'], 'pipeline_check')
                    self.assertEqual(namespace['metadata']['split_sizes']['train_reference'], 2 * spec.classes)
                    for name in ['reference_outputs.npz', 'calibration_outputs.npz', 'test_outputs.npz', 'split_indices.npz',
                                 'test_reliability_metrics_smoke.csv', 'test_class_precision_recall_smoke.csv',
                                 'test_selective_precision_recall_smoke.csv', 'temperature_fits_smoke.csv',
                                 'test_recall_coverage_smoke.csv', 'confusion_matrix_smoke.png', 'recall_coverage_smoke.png',
                                 'risk_coverage_and_hr.png', 'experiment_metadata_smoke.json']:
                        self.assertTrue((run / name).is_file(), name)
                    curves = namespace['recall_curves']
                    self.assertEqual(set(curves), set(namespace['metrics_table'].index))
                    for x, y in curves.values():
                        self.assertEqual(x[-1], 1)
                        np.testing.assert_allclose(y[-1], namespace['metrics_table']['macro_recall'].iloc[0], atol=1e-12)
                    arrays = np.load(run / 'test_outputs.npz')
                    np.testing.assert_array_equal(arrays['dataset_indices'], namespace['test_indices'])
                    for temperature in namespace['temperature_table'].itertuples():
                        self.assertLessEqual(temperature.calibration_nll_after, temperature.calibration_nll_before + 1e-12)


if __name__ == '__main__':
    unittest.main()
