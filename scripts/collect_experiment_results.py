"""Collect completed benchmark evidence, excluding smoke runs and pending outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
from shift_metrics import fingerprint, metric_functions, selective_statistics


KEYS = ['dataset', 'condition', 'classes', 'eval_split', 'checkpoint_sha256', 'seed', 'track', 'metric_definition']


def comparison_group(row):
    identity = {key: row.get(key) for key in KEYS}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def collect(campaign, shift_dirs=()):
    import torch
    campaign = Path(campaign)
    state = json.loads((campaign / 'status.json').read_text())
    rows, inventory = [], []
    ns = metric_functions()
    for job in state['jobs']:
        status = job['status']
        live = False
        if status == 'running':
            try: os.kill(job['pid'], 0); live = True
            except (ProcessLookupError, KeyError): status = 'interrupted_or_stale'
        inventory.append({'experiment': job['id'], 'status': status, 'live_process': live,
                          'source_revision': state['source_revision'], 'log': job['log']})
        if status != 'completed': continue
        if job['id'] == 'resnet18':
            run = campaign / 'runs/cifar100_resnet18/seed_42/benchmark'
            metadata = json.loads((run / 'resnet18_execution.json').read_text())
            if metadata['status'] != 'completed_full_run': continue
            checkpoint_path = run / 'outputs/cifar100_resnet18.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            with np.load(run / 'outputs/cifar100_resnet18_validation_outputs.npz') as arrays:
                logits, labels = arrays['logits'], arrays['targets']
            for method, temperature in [('softmax', 1.), ('temperature', checkpoint['temperature'])]:
                probabilities = ns['temperature_probabilities'](logits, temperature)
                metrics = ns['probability_metrics'](probabilities, labels, logits.argmax(1))
                metrics.pop('curve_coverage'); metrics.pop('curve_risk')
                metrics['full_coverage_accuracy'] = metrics.pop('accuracy'); metrics.pop('coverage')
                row = dict(metrics, dataset='cifar100', condition='clean', backbone='resnet18',
                           classes=100, eval_split='validation', checkpoint_sha256=fingerprint(checkpoint_path),
                           seed=checkpoint['settings']['seed'], track='standard_25', method=method,
                           calibration_examples=len(checkpoint['split_indices']['calibration']), metric_definition='grouped-threshold-step-aurc-v1',
                           source_revision=metadata['source_revision'], source_file=str(checkpoint_path))
                if method == 'temperature':
                    threshold = checkpoint['thresholds']['plain']
                    admitted = np.zeros(len(labels), dtype=bool) if threshold is None else probabilities.max(1) >= threshold
                    selection = selective_statistics(logits.argmax(1), labels, admitted)
                    row.update(frozen_rule_coverage=selection['coverage'], frozen_rule_accuracy=selection['accepted_accuracy'],
                               frozen_rule_accuracy_low95=selection['accepted_accuracy_low95'],
                               frozen_rule_accuracy_high95=selection['accepted_accuracy_high95'])
                rows.append(row)
        elif job['id'] == 'resnet18_sdm':
            # SDM aggregates are ingested from the matched evaluator, avoiding a new fit/re-score here.
            continue
        else:
            if job['id'] == 'cifar100':
                run = campaign / 'runs/cifar100_convnext/seed_42/benchmark'
                execution = json.loads((run / 'cifar100_execution.json').read_text())
                if execution['status'] != 'completed_full_run': continue
            else:
                run = campaign / 'runs/datasets' / job['id'] / 'convnext_tiny_imagenet1k_v1/seed_42/calibration_25_vs_100/benchmark'
            metadata = json.loads((run / 'experiment_metadata_full.json').read_text())
            if metadata.get('status', 'complete_benchmark') != 'complete_benchmark': continue
            metrics = pd.read_csv(run / 'test_reliability_metrics_full.csv').rename(columns={'accuracy': 'full_coverage_accuracy'})
            checksum = fingerprint(run / 'best_checkpoint.pt')
            for record in metrics.to_dict('records'):
                record.pop('coverage', None)
                method = record['method']
                tracks = ['standard_25', 'enlarged_100'] if method == 'softmax' else ['standard_25' if 'standard_25' in method else 'enlarged_100']
                hr = pd.read_csv(run / 'test_hr_metrics_full.csv')
                for track in tracks:
                    family = 'softmax' if method == 'softmax' else 'temperature' if method.startswith('temperature_') else 'sdm_identity'
                    row = dict(record, method=family, dataset=job['id'], condition='clean', backbone='convnext',
                               classes=metadata['config'].get('number_of_classes', 100), eval_split='official_test',
                               checkpoint_sha256=checksum, seed=metadata['config']['seed'], track=track,
                               calibration_examples=metadata['split_sizes']['standard_calibration' if track == 'standard_25' else 'enlarged_calibration'],
                               source_revision=state['source_revision'], metric_definition='grouped-threshold-step-aurc-v1',
                               source_file=str(run / 'test_reliability_metrics_full.csv'))
                    if family == 'sdm_identity':
                        selection = hr.loc[hr['track'] == track].iloc[0]
                        row.update(frozen_rule_coverage=selection['test_coverage'], frozen_rule_accuracy=selection['test_accuracy_when_admitted'])
                    rows.append(row)
    for directory in shift_dirs:
        directory = Path(directory)
        path = directory / 'experiment_metadata.json'
        if not path.exists():
            inventory.append({'experiment': str(directory), 'status': 'shift_pending'}); continue
        metadata = json.loads(path.read_text())
        if metadata['status'] != 'complete_benchmark' or metadata['mode'] != 'benchmark':
            inventory.append({'experiment': str(directory), 'status': 'excluded_pipeline_check'}); continue
        frame = pd.read_csv(directory / 'matched_shift_metrics.csv')
        if set(frame['checkpoint_sha256']) != {metadata['checkpoint_sha256']}:
            raise ValueError('Shift table/checkpoint provenance mismatch')
        for record in frame.to_dict('records'):
            if record['backbone'] == 'resnet18' and record['track'] == 'fine100':
                record['track'] = 'standard_25'
            record['frozen_rule_coverage'] = record.pop('coverage')
            record['frozen_rule_accuracy'] = record.pop('accepted_accuracy')
            record['frozen_rule_accuracy_low95'] = record.pop('accepted_accuracy_low95')
            record['frozen_rule_accuracy_high95'] = record.pop('accepted_accuracy_high95')
            rows.append(dict(record, dataset='cifar100', source_file=str(directory / 'matched_shift_metrics.csv')))
    for row in rows: row['comparison_group'] = comparison_group(row)
    if rows:
        frame = pd.DataFrame(rows).drop_duplicates(subset=KEYS + ['method'], keep='last')
    return frame if rows else pd.DataFrame(columns=['dataset','condition','backbone','classes','eval_split','checkpoint_sha256','seed','track','method','full_coverage_accuracy','frozen_rule_coverage','frozen_rule_accuracy','comparison_group']), pd.DataFrame(inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--shift-dirs', type=Path, nargs='*', default=[])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    table, inventory = collect(args.campaign, args.shift_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / 'experiment_results.csv', index=False)
    inventory.to_csv(args.output_dir / 'experiment_inventory.csv', index=False)
    print(f'Completed benchmark metric rows: {len(table)}; tracked experiments: {len(inventory)}')
    print(inventory.to_string(index=False))


if __name__ == '__main__': main()
