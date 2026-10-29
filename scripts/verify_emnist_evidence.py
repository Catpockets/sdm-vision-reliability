"""Verify saved EMNIST evidence without model inference, training or SDM fitting."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from shift_metrics import fingerprint, metric_functions, threshold_for_target, admission_mask, selective_statistics


def verify(campaign):
    campaign = Path(campaign)
    state = json.loads((campaign / 'status.json').read_text())
    assert state['status'] == 'paused_by_user'
    assert (campaign / 'STOP_AFTER_CURRENT').exists()
    job = next(j for j in state['jobs'] if j['id'] == 'emnist_balanced')
    assert job['status'] == 'completed'
    run = campaign / 'runs/datasets/emnist_balanced/convnext_tiny_imagenet1k_v1/seed_42/calibration_25_vs_100/benchmark'
    meta = json.loads((run / 'experiment_metadata_full.json').read_text())
    assert meta['status'] == 'complete_benchmark'
    assert meta['config']['number_of_classes'] == 47 and meta['config']['run_mode'] == 'benchmark'
    assert len(meta['class_names']) == len(set(meta['class_names'])) == 47
    assert meta['config']['seed'] == 42 and meta['config']['epochs'] == 20
    assert meta['split_sizes'] == {'train_reference': 106925, 'validation': 1175,
                                 'standard_calibration': 1175, 'enlarged_calibration': 4700, 'test': 18800}
    with np.load(run / 'split_indices.npz') as saved:
        splits = {k: saved[k] for k in saved.files}
    for key, size in [('train', 106925), ('validation', 1175), ('standard', 1175), ('enlarged', 4700)]:
        assert len(splits[key]) == size and len(np.unique(splits[key])) == size
    assert set(splits['standard']) <= set(splits['enlarged'])
    for left, right in [('train', 'validation'), ('train', 'enlarged'), ('validation', 'enlarged')]:
        assert not np.intersect1d(splits[left], splits[right]).size
    assert len(set(splits['train']) | set(splits['validation']) | set(splits['enlarged'])) == 112800
    np.testing.assert_array_equal(np.sort(np.r_[splits['train'], splits['validation'], splits['enlarged']]), np.arange(112800))
    with np.load(run / 'test_outputs.npz') as saved:
        logits, labels, predictions, indices = [saved[k] for k in ['logits', 'targets', 'predictions', 'dataset_indices']]
        raw_probabilities = saved['probabilities']
    with np.load(run / 'calibration_outputs.npz') as saved:
        cal_logits, cal_labels, cal_indices, cal_probabilities = [saved[k] for k in ['logits', 'targets', 'dataset_indices', 'probabilities']]
    np.testing.assert_array_equal(cal_indices, splits['enlarged'])
    assert cal_logits.shape == (4700, 47)
    np.testing.assert_array_equal(np.bincount(cal_labels, minlength=47), np.full(47, 100))
    assert logits.shape == (18800, 47)
    np.testing.assert_array_equal(predictions, logits.argmax(1))
    np.testing.assert_array_equal(np.bincount(labels, minlength=47), np.full(47, 400))
    np.testing.assert_array_equal(np.sort(indices), np.arange(18800))
    individual = pd.read_csv(run / 'test_individual_results_full.csv.gz')
    np.testing.assert_array_equal(individual['test_index'], indices)
    np.testing.assert_array_equal(individual['true_class_id'], labels)
    np.testing.assert_array_equal(individual['predicted_class_id'], predictions)
    np.testing.assert_array_equal(individual['correct'], predictions == labels)
    native = pd.read_csv(run / 'test_reliability_metrics_full.csv').set_index('method')
    hr = pd.read_csv(run / 'test_hr_metrics_full.csv').set_index('track')
    ns = metric_functions()
    correct = predictions == labels
    def ranking_metrics(confidence):
        order = np.argsort(-confidence, kind='stable'); sorted_scores = confidence[order]
        ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(order) - 1]
        coverage = (ends + 1) / len(order)
        accuracy = np.cumsum(correct[order])[ends] / (ends + 1)
        error_auroc, error_auprc = ns['binary_ranking_metrics'](~correct, 1 - confidence)
        meets = coverage[accuracy >= .95]
        result = {'aurc': float(np.sum(np.diff(np.r_[0., coverage]) * (1 - accuracy))),
                  'error_auroc': float(error_auroc), 'error_auprc': float(error_auprc),
                  'coverage_at_95pct_accuracy': float(meets.max()) if len(meets) else 0.}
        for budget in [10, 25, 50, 75, 100]:
            eligible = np.flatnonzero(coverage <= budget / 100)
            result[f'accuracy_at_{budget}pct_coverage'] = float(accuracy[eligible[-1]]) if len(eligible) else None
            result[f'actual_coverage_at_{budget}pct_budget'] = float(coverage[eligible[-1]]) if len(eligible) else 0.
        return result
    recomputed, admissions = {}, []
    for method, temperature in [('softmax', 1.)] + [('temperature_' + fit['track'], fit['temperature']) for fit in meta['temperature_fits']]:
        probabilities = raw_probabilities if method == 'softmax' else ns['temperature_probabilities'](logits, temperature)
        result = ns['probability_metrics'](probabilities, labels, predictions)
        confidence_column = 'softmax_confidence' if method == 'softmax' else method.removeprefix('temperature_') + '_temperature_confidence'
        confidence = individual[confidence_column].to_numpy(dtype=probabilities.dtype)
        np.testing.assert_allclose(probabilities.max(1), confidence, rtol=1e-6, atol=1e-7)
        result.update(ranking_metrics(confidence))
        for key, value in result.items():
            if key not in {'curve_coverage', 'curve_risk'}:
                np.testing.assert_allclose(value, native.loc[method, key], rtol=2e-6, atol=2e-7, equal_nan=True, err_msg=method + ':' + key)
        recomputed[method] = {key: float(result[key]) for key in ['accuracy', 'nll', 'brier', 'ece_15', 'error_auroc', 'error_auprc', 'aurc']}
    for track in ['standard_25', 'enlarged_100']:
        confidence = individual[track + '_sdm_confidence'].to_numpy(dtype=np.float32)
        ranking = ranking_metrics(confidence)
        for key, value in ranking.items():
            np.testing.assert_allclose(np.nan if value is None else value, native.loc['sdm_' + track, key], rtol=1e-7, atol=1e-9, equal_nan=True)
        admit = individual[track + '_hr_admitted'].to_numpy(dtype=bool)
        count = int(admit.sum()); assert count == int(hr.loc[track, 'test_admitted']) == 0
        assert not bool(hr.loc[track, 'region_found_on_calibration'])
        assert hr.loc[track, 'test_coverage'] == 0 and pd.isna(hr.loc[track, 'test_accuracy_when_admitted'])
        admissions.append({'track': track, 'examples': len(labels), 'admitted': count,
                           'coverage': 0., 'accepted_accuracy': None, 'region_found_on_calibration': False})
        recomputed['sdm_' + track] = ranking
    frozen_baselines = []
    for fit in meta['temperature_fits']:
        track = fit['track']
        mask = np.isin(cal_indices, splits['standard']) if track == 'standard_25' else np.ones(len(cal_indices), dtype=bool)
        expected_per_class = 25 if track == 'standard_25' else 100
        np.testing.assert_array_equal(np.bincount(cal_labels[mask], minlength=47), np.full(47, expected_per_class))
        for method, temperature in [('softmax', 1.), ('temperature', fit['temperature'])]:
            cal_probs = cal_probabilities if method == 'softmax' else ns['temperature_probabilities'](cal_logits, temperature)
            cal_confidence = cal_probs.max(1)[mask]
            cal_correct = cal_logits[mask].argmax(1) == cal_labels[mask]
            threshold = threshold_for_target(cal_confidence, cal_correct, target=.95, min_admitted=100)
            cal_admit = admission_mask(cal_confidence, threshold)
            test_confidence = raw_probabilities.max(1) if method == 'softmax' else ns['temperature_probabilities'](logits, temperature).max(1)
            selection = selective_statistics(predictions, labels, admission_mask(test_confidence, threshold))
            frozen_baselines.append(dict(selection, track=track, method=method, calibration_threshold=threshold,
                                         calibration_admitted=int(cal_admit.sum()), calibration_examples=int(mask.sum()),
                                         calibration_accuracy=float(cal_correct[cal_admit].mean()) if cal_admit.any() else None))
    files = ['best_checkpoint.pt', 'experiment_metadata_full.json', 'split_indices.npz', 'test_outputs.npz',
             'calibration_outputs.npz', 'test_individual_results_full.csv.gz', 'test_reliability_metrics_full.csv',
             'test_hr_metrics_full.csv', 'temperature_fits_full.csv']
    evidence = [{'relative_to_campaign': str((run / name).relative_to(campaign)), 'sha256': fingerprint(run / name)} for name in files]
    pause = json.loads((campaign / 'pause.json').read_text())
    for record in pause['completed_artifact_checks']:
        if record['job'] == 'emnist_balanced':
            assert fingerprint(campaign / record['path']) == record['sha256']
    return {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'campaign_source_revision': state['source_revision'],
            'prior_snapshot_sha256': fingerprint(ROOT / 'docs/assets/clean-results-2026-10-20.json'),
            'verification_script_sha256': fingerprint(Path(__file__)), 'dataset': 'emnist_balanced',
            'scope': 'Saved clean EMNIST evaluation only; no training, model inference, SDM refit or shift evidence.',
            'verification_limits': 'Raw/temperature metrics and SDM confidence-ranking/admissions rechecked from per-image artifacts. Full SDM probability-matrix NLL/Brier/ECE are native exports, not independently recomputed. Split indices checked for sizes/nesting/disjointness; training and test use separate official populations.',
            'inventory_at_verification': [{'experiment': j['id'], 'status': j['status']} for j in state['jobs']],
            'split_sizes': meta['split_sizes'], 'class_count': 47, 'test_correct': int(correct.sum()),
            'temperature_fits': meta['temperature_fits'], 'fit_summary': meta['fit_summary'],
            'native_metrics': native.reset_index().to_dict('records'), 'recomputed_metrics': recomputed,
            'sdm_admissions': admissions, 'frozen_global_baselines': frozen_baselines, 'evidence': evidence}


def figure(snapshot, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    native = {row['method']: row for row in snapshot['native_metrics']}
    methods = ['softmax', 'temperature_standard_25', 'temperature_enlarged_100', 'sdm_standard_25', 'sdm_enlarged_100']
    labels = ['Softmax', 'Temp 25', 'Temp 100', 'SDM 25', 'SDM 100']
    colors = ['#555555', '#25769b', '#25769b', '#b56832', '#b56832']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, metric, title in [(axes[0], 'ece_15', 'Calibration error (lower is better)'), (axes[1], 'aurc', 'Risk–coverage area (lower is better)')]:
        values = [native[method][metric] for method in methods]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, fmt='%.3f', fontsize=9, padding=3)
        ax.set_title(title, fontsize=11); ax.set_ylim(0, max(values) * 1.2)
        ax.spines[['top', 'right']].set_visible(False); ax.tick_params(axis='x', labelrotation=25)
    fig.suptitle('EMNIST Balanced: 47 classes, 18,800 test images, seed 42', fontsize=12)
    fig.text(.5, -.07, 'Both frozen SDM regions admit 0 images; accepted accuracy is undefined. 25/100 = calibration images per class.', ha='center', fontsize=9)
    fig.savefig(path, dpi=180, bbox_inches='tight'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    result = verify(args.campaign)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if args.figure: figure(result, args.figure)
    print(f"Verified EMNIST: {result['test_correct']}/18800 correct; 0 SDM admissions on both tracks. Campaign remains paused.")


if __name__ == '__main__': main()
