"""Reconcile completed clean predictions and make failure figures; never run a model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from shift_metrics import fingerprint, metric_functions, threshold_for_target, admission_mask, selective_statistics

KEYS = ['cifar100', 'mnist', 'fashion_mnist', 'kmnist', 'emnist_balanced']
TITLES = ['CIFAR-100', 'MNIST', 'Fashion-MNIST', 'KMNIST', 'EMNIST Balanced']


def analyze(campaign, cache, output):
    campaign, cache, output = map(Path, (campaign, cache, output))
    output.mkdir(parents=True, exist_ok=True)
    state_path = campaign / 'status.json'
    state_before = state_path.read_bytes()
    state = json.loads(state_before)
    assert state['status'] == 'paused_by_user' and (campaign / 'STOP_AFTER_CURRENT').exists()
    old = json.loads((ROOT / 'docs/assets/clean-results-2026-10-20.json').read_text())
    emnist = json.loads((ROOT / 'docs/assets/clean-results-2026-10-29.json').read_text())
    ns = metric_functions()
    groups = []
    for key, title in zip(KEYS, TITLES):
        assert next(j for j in state['jobs'] if j['id'] == key)['status'] == 'completed'
        run = (campaign / 'runs/cifar100_convnext/seed_42/benchmark' if key == 'cifar100' else
               campaign / f'runs/datasets/{key}/convnext_tiny_imagenet1k_v1/seed_42/calibration_25_vs_100/benchmark')
        meta = json.loads((run / 'experiment_metadata_full.json').read_text())
        assert meta.get('status', 'complete_benchmark') == 'complete_benchmark'
        checkpoint_hash = fingerprint(run / 'best_checkpoint.pt')
        expected_files = old['evidence'][key]['files'] if key != 'emnist_balanced' else emnist['evidence']
        for evidence in expected_files:
            path = campaign / evidence['relative_to_campaign']
            assert fingerprint(path) == evidence['sha256'], path.name
        with np.load(run / 'test_outputs.npz') as saved:
            logits, probabilities, labels, predictions, ids = [saved[k] for k in ['logits', 'probabilities', 'targets', 'predictions', 'dataset_indices']]
        count, classes = probabilities.shape
        assert count == meta['split_sizes']['test']
        names = meta.get('class_names')
        if names is None:
            names = datasets.CIFAR100(cache / 'cifar100', train=False, download=False).classes
        assert len(names) == classes
        np.testing.assert_array_equal(predictions, logits.argmax(1))
        np.testing.assert_array_equal(np.sort(ids), np.arange(count))
        individual = pd.read_csv(run / 'test_individual_results_full.csv.gz')
        np.testing.assert_array_equal(individual['test_index'], ids)
        np.testing.assert_array_equal(individual['true_class_id'], labels)
        np.testing.assert_array_equal(individual['predicted_class_id'], predictions)
        correct = predictions == labels
        np.testing.assert_array_equal(individual['correct'], correct)
        native = pd.read_csv(run / 'test_reliability_metrics_full.csv').set_index('method')
        raw = ns['probability_metrics'](probabilities, labels, predictions)
        for metric in ['accuracy', 'nll', 'brier', 'ece_15']:
            np.testing.assert_allclose(raw[metric], native.loc['softmax', metric], rtol=2e-6, atol=2e-7)
        with np.load(run / 'split_indices.npz') as saved:
            split = {k: saved[k] for k in saved.files}
        assert set(split['standard']) <= set(split['enlarged'])
        for a, b in [('train', 'validation'), ('train', 'enlarged'), ('validation', 'enlarged')]:
            assert not np.intersect1d(split[a], split[b]).size
        policies = []
        for fit in meta['temperature_fits']:
            track = fit['track']
            if key == 'cifar100':
                cal_file = 'calibration_standard_outputs.npz' if track == 'standard_25' else 'calibration_enlarged_outputs.npz'
            else:
                cal_file = 'calibration_outputs.npz'
            with np.load(run / cal_file) as saved:
                cal_logits, cal_probs, cal_labels = [saved[k] for k in ['logits', 'probabilities', 'targets']]
                mask = np.ones(len(cal_labels), dtype=bool)
                if key != 'cifar100':
                    np.testing.assert_array_equal(saved['dataset_indices'], split['enlarged'])
                    if track == 'standard_25': mask = np.isin(saved['dataset_indices'], split['standard'])
            per_class = 25 if track == 'standard_25' else 100
            np.testing.assert_array_equal(np.bincount(cal_labels[mask], minlength=classes), np.full(classes, per_class))
            for method, temperature in [('softmax', 1.), ('temperature', fit['temperature'])]:
                cp = cal_probs if method == 'softmax' else ns['temperature_probabilities'](cal_logits, temperature)
                tp = probabilities if method == 'softmax' else ns['temperature_probabilities'](logits, temperature)
                if method == 'temperature':
                    check = ns['probability_metrics'](tp, labels, predictions)
                    for metric in ['accuracy', 'nll', 'brier', 'ece_15']:
                        np.testing.assert_allclose(check[metric], native.loc['temperature_' + track, metric], rtol=2e-6, atol=2e-7)
                    np.testing.assert_allclose(tp.max(1), individual[track + '_temperature_confidence'], rtol=1e-6, atol=1e-7)
                threshold = threshold_for_target(cp.max(1)[mask], cal_logits.argmax(1)[mask] == cal_labels[mask], .95, 100)
                admit = admission_mask(tp.max(1), threshold)
                selection = selective_statistics(predictions, labels, admit)
                selection = {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in selection.items()}
                policies.append(dict(selection, method=method, track=track, threshold=threshold,
                                     calibration_examples=int(mask.sum()), wrong_admitted=int((admit & ~correct).sum()),
                                     confidently_wrong_rate=float((admit & ~correct).mean())))
            admit = individual[track + '_hr_admitted'].to_numpy(dtype=bool)
            selection = selective_statistics(predictions, labels, admit)
            selection = {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in selection.items()}
            policies.append(dict(selection, method='sdm_identity', track=track, threshold=None,
                                 calibration_examples=int(mask.sum()), wrong_admitted=int((admit & ~correct).sum()),
                                 confidently_wrong_rate=float((admit & ~correct).mean())))
        confusion = np.zeros((classes, classes), dtype=int)
        np.add.at(confusion, (labels, predictions), 1)
        assert confusion.sum() == count and confusion.trace() == int(correct.sum())
        pd.DataFrame(confusion, index=pd.Index(names, name='true_class'), columns=names).to_csv(output / f'{key}-confusion.csv')
        fig, ax = plt.subplots(figsize=(12, 11), layout='constrained')
        image = ax.imshow(confusion / confusion.sum(1, keepdims=True), vmin=0, vmax=1, cmap='Blues')
        size = 6 if classes == 100 else 8 if classes == 47 else 10
        ax.set_xticks(np.arange(classes), names, rotation=90, fontsize=size)
        ax.set_yticks(np.arange(classes), names, fontsize=size)
        ax.set(xlabel='Predicted class', ylabel='True class', title=title + ': row-normalized test confusion')
        fig.colorbar(image, ax=ax, shrink=.75, label='Share of each true class')
        fig.savefig(output / f'{key}-confusion-matrix.png', dpi=160); plt.close(fig)
        directed = [(int(confusion[a, b]), a, b) for a in range(classes) for b in range(classes) if a != b and confusion[a, b]]
        directed.sort(key=lambda x: (-x[0], x[1], x[2]))
        pairs = [{'true_class': names[a], 'predicted_class': names[b], 'count': n, 'true_class_id': a, 'predicted_class_id': b}
                 for n, a, b in directed[:8]]
        recall = np.diag(confusion) / confusion.sum(1)
        error_order = np.lexsort((ids, -individual['standard_25_temperature_confidence'].to_numpy()))
        cases = [int(i) for i in error_order if not correct[i]][:3]
        loader = {'cifar100': datasets.CIFAR100, 'mnist': datasets.MNIST, 'fashion_mnist': datasets.FashionMNIST,
                  'kmnist': datasets.KMNIST, 'emnist_balanced': datasets.EMNIST}[key]
        source = cache / 'cifar100' if key == 'cifar100' else cache / 'datasets' / key
        data = loader(source, train=False, download=False, **({'split': 'balanced'} if key == 'emnist_balanced' else {}))
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), layout='constrained')
        examples = []
        for ax, row in zip(axes, cases):
            image, target = data[int(ids[row])]
            assert target == int(labels[row])
            if key == 'emnist_balanced': image = image.transpose(Image.Transpose.TRANSPOSE)
            confidence = float(individual['standard_25_temperature_confidence'].iloc[row])
            ax.imshow(image, cmap='gray' if key != 'cifar100' else None, interpolation='nearest')
            ax.set_title(f'ID {ids[row]} · confidence {confidence:.3f}\nTrue: {names[labels[row]]}\nPredicted: {names[predictions[row]]}', fontsize=9)
            ax.axis('off')
            examples.append({'test_id': int(ids[row]), 'true_class': names[labels[row]], 'predicted_class': names[predictions[row]], 'confidence': confidence})
        for ax in axes[len(cases):]: ax.axis('off')
        fig.suptitle(title + ': three highest-confidence errors (temperature 25/class)', fontsize=11)
        fig.savefig(output / f'{key}-errors.png', dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
        ax.barh([p['true_class'] + ' → ' + p['predicted_class'] for p in pairs][::-1], [p['count'] for p in pairs][::-1], color='#002676')
        ax.set(title=title + ': most frequent directed confusions', xlabel='Number of test errors')
        ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(output / f'{key}-confusions.png', dpi=160); plt.close(fig)
        evidence = [{'relative_to_campaign': str((run / name).relative_to(campaign)), 'sha256': fingerprint(run / name)}
                    for name in sorted(set(['test_outputs.npz', 'split_indices.npz', 'test_individual_results_full.csv.gz', 'experiment_metadata_full.json'] +
                                    (['calibration_standard_outputs.npz', 'calibration_enlarged_outputs.npz'] if key == 'cifar100' else ['calibration_outputs.npz'])))]
        groups.append({'dataset': key, 'title': title, 'classes': classes, 'examples': count, 'correct': int(correct.sum()),
                       'accuracy': float(correct.mean()), 'checkpoint_sha256': checkpoint_hash, 'evaluation_split': 'official_test',
                       'seed': 42, 'policies': policies, 'top_confusions': pairs, 'examples_of_errors': examples,
                       'class_recall': [{'class': names[i], 'support': int(confusion[i].sum()), 'recall': float(recall[i])} for i in range(classes)],
                       'native_metrics': native.reset_index().astype(object).where(pd.notnull(native.reset_index()), None).to_dict('records'), 'evidence': evidence})
        print(key, 'verified:', int(correct.sum()), '/', count, 'correct; policies', len(policies))
    result = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'source_revision': state['source_revision'],
              'analysis_script_sha256': fingerprint(Path(__file__)), 'scope': 'Saved-output-only clean ConvNeXt analysis; no model inference, training or SDM refit.',
              'selection': 'Confusions: largest directed counts, then class IDs. Examples: highest temperature-standard confidence among errors, then test ID. Descriptive selection after test exploration.',
              'limits': 'One seed; populations/checkpoints separate; global empirical thresholds differ from SDM region construction. No corruption/OOD or missing teammate artifacts. SDM probability metrics are native exports.',
              'groups': groups}
    (output / 'evidence.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    assert state_path.read_bytes() == state_before
    assert (campaign / 'STOP_AFTER_CURRENT').exists()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.campaign, args.cache, args.output)


if __name__ == '__main__': main()
