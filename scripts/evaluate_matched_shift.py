"""Freeze CIFAR calibration, verify its checkpoint, then score clean/corrupted/OOD inputs.

ConvNeXt consumes notebook-04 exports. ResNet consumes the existing notebook-02/03
checkpoint and SDM artifacts. Neither path trains or fits anything on shifted inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from shift_metrics import (admission_mask, fingerprint, metric_functions, notebook_definitions,
                           selective_statistics, threshold_for_target)

CORRUPTIONS = ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
               'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
               'elastic_transform', 'pixelate', 'jpeg_compression')


def read_arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {k: source[k] for k in source.files}


def validate_splits(splits):
    sets = {k: set(np.asarray(v).tolist()) for k, v in splits.items()}
    calibration = 'enlarged' if 'enlarged' in sets else 'calibration'
    for name in ['train', 'validation', calibration]:
        if len(sets[name]) != len(splits[name]):
            raise ValueError(f'Duplicate original indices in {name}')
    if any(sets[a] & sets[b] for a, b in [('train', 'validation'), ('train', calibration), ('validation', calibration)]):
        raise ValueError('Training, validation and calibration overlap')
    if 'standard' in sets and not sets['standard'] <= sets['enlarged']:
        raise ValueError('Standard calibration must be nested within enlarged calibration')


def verify_replay(actual, expected, labels, expected_labels):
    if not np.array_equal(labels, expected_labels) or not np.allclose(actual, expected, rtol=1e-3, atol=1e-3):
        raise ValueError('Checkpoint/preprocessing does not reproduce the cached calibration outputs')


def streamed_signals(namespace, embeddings, predictions, batch_size=128):
    nearest, q = [], []
    for start in range(0, len(embeddings), batch_size):
        distances, indices = namespace['exact_squared_l2_neighbors'](
            namespace['reference_embeddings'], embeddings[start:start+batch_size],
            neighbors=namespace['neighbors'], batch_size=batch_size)
        nearest.append(distances[:, 0])
        q.append(namespace['consecutive_correct_match_count'](indices, predictions[start:start+batch_size]))
    return np.concatenate(nearest), np.concatenate(q)


def resnet_transform(mean, std):
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def convnext_transform():
    return transforms.Compose([transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                               transforms.CenterCrop(224), transforms.ToTensor(),
                               transforms.Normalize((.485, .456, .406), (.229, .224, .225))])


@torch.inference_mode()
def infer(model, loader, device, kind):
    logits, embeddings, targets = [], [], []
    for images, labels in loader:
        images = images.to(device)
        if kind == 'convnext':
            vectors = model.classifier[1](model.classifier[0](model.avgpool(model.features(images))))
            output = model.classifier[2](vectors)
        else:
            vectors = model.embed(images)
            output = model.classifier(vectors)
        logits.append(output.cpu().numpy()); embeddings.append(vectors.cpu().numpy())
        targets.append(np.asarray(labels))
    return {'logits': np.concatenate(logits), 'embeddings': np.concatenate(embeddings),
            'targets': np.concatenate(targets)}


class CorruptionSlice(Dataset):
    def __init__(self, root, name, severity, transform):
        if name not in CORRUPTIONS or severity not in range(1, 6):
            raise ValueError('Unknown corruption or severity')
        self.images = np.load(Path(root) / f'{name}.npy', mmap_mode='r', allow_pickle=False)
        labels = np.load(Path(root) / 'labels.npy', allow_pickle=False)
        if self.images.shape != (50000, 32, 32, 3) or self.images.dtype != np.uint8:
            raise ValueError('Expected uint8 corruption array with shape (50000,32,32,3)')
        if labels.shape not in [(10000,), (50000,)] or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('Expected 10000 or 50000 integer corruption labels')
        if labels.min() < 0 or labels.max() > 99:
            raise ValueError('Corruption labels outside CIFAR-100 range')
        self.offset = (severity - 1) * 10000
        self.labels = labels if len(labels) == 10000 else labels[self.offset:self.offset + 10000]
        self.transform = transform

    def __len__(self): return 10000

    def __getitem__(self, index):
        if not 0 <= index < len(self): raise IndexError(index)
        return self.transform(Image.fromarray(self.images[self.offset + index])), int(self.labels[index])


class FrozenExperiment:
    def __init__(self, kind, run, data, device='cpu', mode='benchmark'):
        self.kind, self.run, self.data, self.mode = kind, Path(run), Path(data), mode
        self.device = torch.device(device)
        self.ns = metric_functions()
        self.ns['DEVICE'] = self.device
        self.tracks = {}
        self.frozen = False
        self.replayed_examples = 0
        self.fit_identity = {}
        self.clean_confidence = {}

    def freeze(self):
        if self.kind == 'convnext': self._freeze_convnext()
        else: self._freeze_resnet()
        self.frozen = True
        return self

    def _bind(self, path):
        self.fit_identity[str(Path(path).resolve())] = fingerprint(path)

    def _replay(self, arrays, indices):
        dataset = datasets.CIFAR100(self.data / 'cifar100', train=True, download=False, transform=self.transform)
        sample = np.linspace(0, len(indices) - 1, min(8, len(indices)), dtype=int)
        output = infer(self.model, DataLoader(Subset(dataset, indices[sample]), batch_size=8), self.device, self.kind)
        verify_replay(output['logits'], arrays['logits'][sample], output['targets'], arrays['targets'][sample])
        if not np.allclose(output['embeddings'], arrays['embeddings'][sample], rtol=1e-3, atol=1e-3):
            raise ValueError('Checkpoint does not reproduce cached embeddings')
        self.replayed_examples += len(sample)

    def _freeze_convnext(self):
        checkpoint_path = self.run / 'best_checkpoint.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.config = checkpoint['config']
        if not checkpoint.get('training_complete') or checkpoint.get('completed_epochs') != self.config['epochs'] or self.config['run_mode'] != self.mode:
            raise ValueError('Incomplete checkpoint or smoke/benchmark mismatch')
        self.model = convnext_tiny(weights=None)
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, 100)
        self.model.load_state_dict(checkpoint['model_state_dict']); self.model.to(self.device).eval()
        self.transform = convnext_transform()
        splits = read_arrays(self.run / 'split_indices.npz'); validate_splits(splits)
        if self.mode == 'benchmark' and [len(splits[k]) for k in ['train', 'validation', 'standard', 'enlarged']] != [37500, 2500, 2500, 10000]:
            raise ValueError('Expected notebook-04 full split')
        reference = read_arrays(self.run / 'reference_outputs.npz')
        calibration = read_arrays(self.run / 'calibration_enlarged_outputs.npz')
        for arrays, name in [(reference, 'train'), (calibration, 'enlarged')]:
            if self.mode == 'benchmark' and len(arrays['targets']) != len(splits[name]):
                raise ValueError('Incomplete reference/calibration export')
            if not np.array_equal(arrays['dataset_indices'], splits[name][:len(arrays['targets'])]):
                raise ValueError('Cached representation indices do not match frozen split')
        self._replay(calibration, splits['enlarged'][:len(calibration['targets'])])
        self._replay(reference, splits['train'][:len(reference['targets'])])
        ns = self.ns
        ns['reference'] = reference
        tensor = torch.as_tensor(reference['embeddings'], dtype=torch.float32)
        ns['embedding_mean'] = tensor.mean()
        ns['embedding_std'] = tensor.std(correction=1).clamp_min(torch.finfo(torch.float32).eps)
        ns['reference_embeddings'] = ns['normalize_embeddings'](reference['embeddings'])
        ns['neighbors'] = min(len(tensor), int(np.bincount(reference['targets'], minlength=100).max()) + 1)
        distances, q = streamed_signals(ns, ns['normalize_embeddings'](calibration['embeddings']), calibration['predictions'])
        ns['calibration_neighbor_distances'] = distances[:, None]
        ns['calibration_q_all'] = q
        standard_mask = np.isin(calibration['dataset_indices'], splits['standard'])
        if self.mode == 'smoke' and not standard_mask.any():
            raise ValueError('Smoke calibration exports have no standard-track members')
        for name, mask in [('standard_25', standard_mask), ('enlarged_100', np.ones(len(q), dtype=bool))]:
            arrays = {k: v[mask] for k, v in calibration.items()}
            track = ns['fit_track'](name, arrays, mask)
            fit = ns['fit_temperature'](arrays['logits'], arrays['targets'])
            track.update(temperature=fit['temperature'], calibration_examples=len(mask.nonzero()[0]))
            self._baseline_thresholds(track, arrays, min_admitted=100)
            self.tracks[name] = track
        self.clean_path = self.run / 'test_outputs.npz'
        self.eval_split = 'official_test'
        self.checkpoint = checkpoint_path
        for name in ['best_checkpoint.pt', 'split_indices.npz', 'reference_outputs.npz', 'calibration_enlarged_outputs.npz']:
            self._bind(self.run / name)

    def _baseline_thresholds(self, track, arrays, min_admitted):
        raw = self.ns['temperature_probabilities'](arrays['logits'], 1)
        scaled = self.ns['temperature_probabilities'](arrays['logits'], track['temperature'])
        correct = arrays['logits'].argmax(1) == arrays['targets']
        track['thresholds'] = {name: threshold_for_target(p.max(1), correct, .95, min_admitted)
                               for name, p in [('softmax', raw), ('temperature', scaled)]}

    def _freeze_resnet(self):
        from reexpress_sdm import SDMModel
        self.checkpoint = self.run / 'cifar100_resnet18.pt'
        checkpoint = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        self.config = checkpoint['settings']
        if self.mode == 'benchmark' and self.config.get('epochs') != 30:
            raise ValueError('Expected a complete 30-epoch ResNet benchmark')
        if self.mode == 'smoke' and self.config.get('epochs') != 1:
            raise ValueError('Smoke/benchmark checkpoint mismatch')
        splits = checkpoint['split_indices']; validate_splits(splits)
        if len(checkpoint['history']['train loss']) != self.config['epochs']:
            raise ValueError('Incomplete ResNet training history')
        if self.mode == 'benchmark' and [len(splits[k]) for k in ['train', 'validation', 'calibration']] != [45000, 2500, 2500]:
            raise ValueError('Expected full ResNet split')
        definitions = notebook_definitions(ROOT / 'notebooks/02_cifar100_cnn_resnet18.ipynb',
                                          ['ResidualBlock', 'CifarResNet'], {'torch': torch, 'nn': nn, 'F': F})
        self.model = definitions['CifarResNet']().to(self.device)
        self.model.load_state_dict(checkpoint['model_state']); self.model.eval()
        self.transform = resnet_transform(checkpoint['channel_mean'].flatten().tolist(), checkpoint['channel_std'].flatten().tolist())
        calibration = read_arrays(self.run / 'cifar100_resnet18_calibration_outputs.npz')
        if not np.array_equal(calibration['dataset_indices'], splits['calibration']):
            raise ValueError('Calibration/checkpoint split mismatch')
        self._replay(calibration, splits['calibration'])
        track = {'temperature': float(checkpoint['temperature']), 'calibration_examples': len(calibration['targets'])}
        self._baseline_thresholds(track, calibration, min_admitted=1)
        # Check artifacts against representations from this exact backbone, not a generic model name.
        exported = read_arrays(self.run / 'cifar100_resnet18_embeddings.npz')
        if not np.array_equal(exported['train_index'], splits['train']):
            raise ValueError('SDM support/checkpoint split mismatch')
        self._replay({'logits': exported['train_logits'], 'embeddings': exported['train_embeddings'], 'targets': exported['train_labels']}, splits['train'])
        probes = np.linspace(0, len(exported['train_index'])-1, min(8, len(exported['train_index'])), dtype=int)
        for classes, key in [(100, 'fine100'), (20, 'coarse20')]:
            path = self.run / f'sdm_resnet18_{key}.sdmkitmodel'
            sdm = SDMModel.load(path, device=str(self.device))
            if sdm.artifact.manifest['configuration']['numberOfClasses'] != classes:
                raise ValueError('SDM class-count mismatch')
            _, exemplars = sdm.transform(exported['train_embeddings'][probes])
            if not np.allclose(exemplars, sdm.artifact.support_vectors[probes], rtol=1e-3, atol=1e-3):
                raise ValueError('SDM artifact does not match this backbone support')
            if [record.id for record in sdm.artifact.support_records] != [f'train-{i}' for i in splits['train']]:
                raise ValueError('SDM support identities do not match checkpoint')
            expected_labels = exported['train_labels' if classes == 100 else 'train_groups']
            if not np.array_equal([record.label for record in sdm.artifact.support_records], expected_labels):
                raise ValueError('SDM support label mapping mismatch')
            if [row['id'] for row in sdm.artifact.calibration_rows] != [f'train-{i}' for i in splits['calibration']]:
                raise ValueError('SDM calibration identities do not match checkpoint')
            self.tracks[key] = dict(track, sdm=sdm, classes=classes)
            self._bind(path)
        with (self.data / 'cifar100/cifar-100-python/train').open('rb') as stream:
            training = pickle.load(stream, encoding='latin1')
        self.fine_to_group = np.zeros(100, dtype=int)
        self.fine_to_group[np.asarray(training['fine_labels'])] = training['coarse_labels']
        coarse = self.tracks['coarse20']
        cal_group = self._groups(self.ns['temperature_probabilities'](calibration['logits'], coarse['temperature']))
        raw_group = self._groups(self.ns['temperature_probabilities'](calibration['logits'], 1))
        coarse['thresholds'] = {name: threshold_for_target(p.max(1), p.argmax(1) == self.fine_to_group[calibration['targets']], .95, 1)
                                for name, p in [('softmax', raw_group), ('temperature', cal_group)]}
        self.clean_path = self.run / 'cifar100_resnet18_validation_outputs.npz'
        self.eval_split = 'validation'
        for name in ['cifar100_resnet18.pt', 'cifar100_resnet18_calibration_outputs.npz', 'cifar100_resnet18_embeddings.npz']:
            self._bind(self.run / name)

    def _groups(self, probabilities):
        grouped = np.zeros((len(probabilities), 20))
        np.add.at(grouped.T, self.fine_to_group, probabilities.T)
        return grouped

    def describe(self):
        if not self.frozen: raise RuntimeError('Fit must be frozen first')
        return {'backbone': self.kind, 'checkpoint_sha256': fingerprint(self.checkpoint),
                'fit_identity': self.fit_identity, 'config': self.config, 'mode': self.mode,
                'alpha': .95, 'replayed_calibration_examples': self.replayed_examples,
                'eval_split': self.eval_split, 'frozen_at': datetime.now(timezone.utc).isoformat(),
                'evaluation_driver_sha256': fingerprint(__file__),
                'metric_source_sha256': fingerprint(ROOT / 'notebooks/04_cifar100_sdm_calibration_comparison.ipynb'),
                'sdm_variant': 'frozen_identity_adaptor' if self.kind == 'convnext' else 'learned_reexpress_adaptor_0.4.9',
                'tracks': {k: {'temperature': t['temperature'], 'thresholds': t['thresholds'],
                                'calibration_examples': t['calibration_examples'],
                                'hr_region': ({'found': t['region']['found'],
                                               'q_prime_min': t['region']['q_prime_min'] if t['region']['found'] else None,
                                               'thresholds': t['region']['thresholds'].tolist() if t['region']['found'] else None}
                                              if 'region' in t else 'see hashed SDM artifact manifest')} for k, t in self.tracks.items()}}

    def methods(self, arrays):
        if not self.frozen: raise RuntimeError('Freeze calibration before reading evaluation outputs')
        result = {}
        logits = arrays['logits']
        raw = self.ns['temperature_probabilities'](logits, 1)
        if self.kind == 'convnext':
            predictions = logits.argmax(1)
            embeddings = self.ns['normalize_embeddings'](arrays['embeddings'])
            distances, q = streamed_signals(self.ns, embeddings, predictions)
            for name, track in self.tracks.items():
                d = self.ns['conservative_distance_quantile'](distances, track['cdfs'])
                sdm = self.ns['sdm_activation'](logits, q, d)
                q_prime = self.ns['rescaled_similarity'](q, sdm[np.arange(len(q)), predictions])
                admitted = self.ns['apply_high_reliability_region'](sdm, q_prime, predictions, track['region'])
                scaled = self.ns['temperature_probabilities'](logits, track['temperature'])
                result[(name, 'sdm_identity')] = (sdm, predictions, admitted, 100)
                for method, probabilities in [('softmax', raw), ('temperature', scaled)]:
                    result[(name, method)] = (probabilities, predictions,
                        admission_mask(probabilities.max(1), track['thresholds'][method]), 100)
        else:
            for name, track in self.tracks.items():
                scores = track['sdm'].score(arrays['embeddings'], nearest_exemplars=0)
                probabilities = np.array([s.sdm for s in scores])
                prediction = np.array([s.prediction for s in scores])
                result[(name, 'sdm_adaptor_centroid')] = (probabilities, prediction,
                    np.array([s.centroid_region_alpha >= .95 for s in scores]), track['classes'])
                result[(name, 'sdm_adaptor_lower')] = (probabilities, prediction,
                    np.array([s.lower_region_alpha >= .95 for s in scores]), track['classes'])
                for method, temp in [('softmax', 1), ('temperature', track['temperature'])]:
                    p = self.ns['temperature_probabilities'](logits, temp)
                    if track['classes'] == 20: p = self._groups(p)
                    result[(name, method)] = (p, p.argmax(1), admission_mask(p.max(1), track['thresholds'][method]), track['classes'])
        return result


def summarize(experiment, methods, labels, condition, base):
    rows, individuals = [], {}
    for (track, method), (probabilities, predictions, admitted, classes) in methods.items():
        confidence = probabilities[np.arange(len(predictions)), predictions]
        key = track + '__' + method
        target = labels if classes == 100 or labels is None else experiment.fine_to_group[labels]
        row = dict(base, track=track, method=method, classes=classes,
                   calibration_examples=experiment.tracks[track]['calibration_examples'],
                   admission_rule='upstream_HR_alpha95' if method.startswith('sdm') else 'global_empirical_calibration_accuracy95',
                   condition=condition, **selective_statistics(predictions, target, admitted))
        row['id_reference_split'] = experiment.eval_split
        row['eval_split'] = experiment.eval_split if condition == 'clean' else 'svhn_test' if condition == 'svhn_ood' else 'cifar100c_official_test_derived'
        if labels is not None:
            metric = experiment.ns['probability_metrics'](probabilities, target, predictions)
            metric.pop('curve_coverage'); metric.pop('curve_risk')
            # Selective coverage above refers to the calibration-frozen rule, not retrospective ranking.
            metric['full_coverage_accuracy'] = metric.pop('accuracy'); metric.pop('coverage')
            row.update(metric)
            if condition == 'clean': experiment.clean_confidence[key] = confidence
        else:
            row['ood_false_admission_rate'] = float(admitted.mean())
            id_confidence = experiment.clean_confidence[key]
            combined = np.r_[id_confidence, confidence]
            is_id = np.r_[np.ones(len(id_confidence), dtype=bool), np.zeros(len(confidence), dtype=bool)]
            row['id_vs_ood_auroc'] = experiment.ns['binary_ranking_metrics'](is_id, combined)[0]
        individuals[key + '__confidence'] = confidence
        individuals[key + '__prediction'] = predictions
        individuals[key + '__admitted'] = admitted
        rows.append(row)
    return rows, individuals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', choices=['convnext', 'resnet18'], required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'benchmark'], default='benchmark')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--corruptions', nargs='+', choices=CORRUPTIONS, default=list(CORRUPTIONS))
    parser.add_argument('--severities', nargs='+', type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5])
    parser.add_argument('--skip-corruptions', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1: parser.error('Batch size must be positive')
    device = args.device
    if device == 'auto': device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'frozen_fit.json').exists():
        raise FileExistsError('Output directory already contains a frozen run; use a new directory.')
    experiment = FrozenExperiment(args.backbone, args.run_dir, args.data_root, device, args.mode).freeze()
    frozen = experiment.describe()
    frozen.update(corruption_test_caveat='CIFAR-100-C derives from official CIFAR-100 test images; never use shift outcomes to tune the frozen choices',
                  corruptions=[] if args.skip_corruptions else args.corruptions, severities=args.severities,
                  metric_definition='grouped-threshold-step-aurc-v1', status='pipeline_check' if args.mode == 'smoke' else 'running')
    if not args.skip_corruptions:
        source_path = args.data_root / 'CIFAR-100-C/SOURCE.json'
        if not source_path.exists():
            raise FileNotFoundError('Prepare and verify the official corruption data before evaluation')
        frozen['corruption_source'] = json.loads(source_path.read_text())
    (args.output_dir / 'frozen_fit.json').write_text(json.dumps(frozen, indent=2) + '\n')
    base = {'backbone': args.backbone, 'checkpoint_sha256': frozen['checkpoint_sha256'],
            'seed': experiment.config['seed'], 'eval_split': experiment.eval_split, 'mode': args.mode,
            'metric_definition': frozen['metric_definition']}
    all_rows = []

    def evaluate(name, arrays, labels):
        rows, individual = summarize(experiment, experiment.methods(arrays), labels, name, base)
        all_rows.extend(rows)
        np.savez_compressed(args.output_dir / f'{name}_outputs.npz', **arrays, **individual)
        pd.DataFrame(all_rows).to_csv(args.output_dir / 'matched_shift_metrics.csv', index=False)
        print('Completed', name, 'images', len(arrays['logits']), flush=True)

    clean = read_arrays(experiment.clean_path)  # First evaluation labels loaded AFTER the frozen manifest.
    evaluate('clean', clean, clean['targets'])
    svhn = datasets.SVHN(args.data_root / 'datasets/svhn', split='test', download=False, transform=experiment.transform)
    svhn_indices = np.arange(min(20, len(svhn))) if args.mode == 'smoke' else np.arange(len(svhn))
    arrays = infer(experiment.model, DataLoader(Subset(svhn, svhn_indices), batch_size=args.batch_size), device, args.backbone)
    arrays['dataset_indices'] = svhn_indices
    # SVHN labels are audit-only digits; never score them as CIFAR classes.
    arrays['svhn_digit_labels'] = arrays.pop('targets')
    evaluate('svhn_ood', arrays, None)
    if not args.skip_corruptions:
        for name in args.corruptions:
            for severity in args.severities:
                dataset = CorruptionSlice(args.data_root / 'CIFAR-100-C', name, severity, experiment.transform)
                indices = np.arange(20) if args.mode == 'smoke' else np.arange(len(dataset))
                arrays = infer(experiment.model, DataLoader(Subset(dataset, indices), batch_size=args.batch_size), device, args.backbone)
                arrays['dataset_indices'] = indices
                evaluate(f'{name}_s{severity}', arrays, arrays['targets'])
    frozen['status'] = 'pipeline_check' if args.mode == 'smoke' else 'complete_benchmark'
    (args.output_dir / 'experiment_metadata.json').write_text(json.dumps(frozen, indent=2) + '\n')


if __name__ == '__main__': main()
