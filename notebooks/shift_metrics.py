"""Shared calibration/provenance helpers for the CIFAR baseline and shift runs."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    paths = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    for item in paths:
        if path.is_dir():
            digest.update(item.relative_to(path).as_posix().encode())
        with item.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
    return digest.hexdigest()


def artifact_identity(run_dir):
    run_dir = Path(run_dir)
    return {name: fingerprint(run_dir / name) for name in (
        'best_checkpoint.pt', 'calibration_outputs_full.npz', 'test_outputs_full.npz')}


def verify_identity(run_dir, saved):
    if not saved or artifact_identity(run_dir) != saved:
        raise ValueError('Checkpoint/output identity changed. Rerun notebook 11 on matching artifacts first.')


def threshold_for_target(confidences, correct, target=.95, min_admitted=100):
    """Largest attainable calibration coverage at the target; None admits nothing."""
    scores, correct = np.asarray(confidences), np.asarray(correct, dtype=bool)
    if scores.ndim != 1 or correct.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError('Expected aligned finite one-dimensional confidence/correctness arrays')
    if not 0 < target <= 1 or min_admitted < 1:
        raise ValueError('Target and minimum admission count must be positive')
    if not len(scores):
        return None
    order = np.argsort(-scores, kind='stable')
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores) - 1]
    counts = ends + 1
    accuracy = np.cumsum(correct[order])[ends] / counts
    eligible = np.flatnonzero((accuracy >= target) & (counts >= min_admitted))
    return float(sorted_scores[ends[eligible[-1]]]) if len(eligible) else None


def admission_mask(confidence, threshold):
    return np.zeros(len(confidence), dtype=bool) if threshold is None else np.asarray(confidence) >= threshold


def notebook_definitions(path, names, namespace):
    """Reuse reviewed, visible notebook code without executing fitting/data cells."""
    notebook = json.loads(Path(path).read_text())
    found = set()
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code' or ''.join(cell['source']).lstrip().startswith('%'):
            continue
        nodes = [n for n in ast.parse(''.join(cell['source'])).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        if nodes:
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
            found.update(n.name for n in nodes)
    if found != set(names):
        raise ValueError(f'Missing reviewed definitions in {path}: {set(names) - found}')
    return namespace


def metric_functions():
    import pandas as pd
    import torch
    names = ['classification_metrics', 'binary_ranking_metrics', 'probability_metrics',
             'temperature_probabilities', 'fit_temperature', 'normalize_embeddings',
             'exact_squared_l2_neighbors', 'consecutive_correct_match_count',
             'build_class_distance_cdfs', 'conservative_distance_quantile', 'sdm_activation',
             'rescaled_similarity', 'upstream_empirical_quantile', 'fit_high_reliability_region',
             'apply_high_reliability_region', 'fit_track']
    return notebook_definitions(Path(__file__).with_name('04_cifar100_sdm_calibration_comparison.ipynb'),
                                names, {'np': np, 'pd': pd, 'torch': torch, 'TARGET_ALPHA': .95})


def selective_statistics(predictions, labels, admitted):
    from math import sqrt
    admitted = np.asarray(admitted, dtype=bool)
    count = int(admitted.sum())
    result = {'examples': len(admitted), 'admitted': count,
              'coverage': float(admitted.mean()) if len(admitted) else 0.,
              'accepted_accuracy': np.nan, 'accepted_accuracy_low95': np.nan,
              'accepted_accuracy_high95': np.nan}
    if labels is not None and count:
        p = float((np.asarray(predictions)[admitted] == np.asarray(labels)[admitted]).mean())
        z = 1.96
        denominator = 1 + z*z/count
        center = (p + z*z/(2*count)) / denominator
        half = z*sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
        result.update(accepted_accuracy=p, accepted_accuracy_low95=center-half, accepted_accuracy_high95=center+half)
    return result
