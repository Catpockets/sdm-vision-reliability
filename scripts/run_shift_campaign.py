"""Prepare verified corruption data, validate pipelines, then evaluate completed models.

Runs as a one-off dependency job. It never overlaps GPU work with the training
campaign, never modifies its source snapshot, and records pending/failed states.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from collect_experiment_results import collect

URL = 'https://zenodo.org/api/records/3555552/files/CIFAR-100-C.tar/content'
MD5 = '11f0ed0f1191edbf9fa23466ae6021d3'
SIZE = 2918473216
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.md5()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): result.update(chunk)
    return result.hexdigest()


def prepare(data_root):
    data_root.mkdir(parents=True, exist_ok=True)
    archive = data_root / 'CIFAR-100-C.tar'
    partial = data_root / 'CIFAR-100-C.tar.download'
    if not archive.exists():
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                        '--connect-timeout', '15', '--max-time', '1800', '--continue-at', '-',
                        URL, '--output', str(partial)], check=True)
        if partial.stat().st_size != SIZE or digest(partial) != MD5:
            raise ValueError('Official corruption archive integrity check failed; not extracting')
        partial.replace(archive)
    if archive.stat().st_size != SIZE or digest(archive) != MD5:
        raise ValueError('Corruption archive checksum mismatch')
    from evaluate_matched_shift import CORRUPTIONS
    selected = {f'CIFAR-100-C/{name}.npy' for name in CORRUPTIONS} | {'CIFAR-100-C/labels.npy'}
    with tarfile.open(archive) as source:
        members = [m for m in source.getmembers() if m.name.rstrip('/') in selected and m.isfile()]
        if {m.name for m in members} != selected: raise ValueError('Archive lacks standard corruption files')
        source.extractall(data_root, members=members, filter='data')
    metadata = {'source': URL, 'archive_bytes': SIZE, 'archive_md5': MD5,
                'verified_at': datetime.now(timezone.utc).isoformat()}
    (data_root / 'CIFAR-100-C/SOURCE.json').write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / 'status.json'
    if state_path.exists(): raise FileExistsError('Shift campaign already initialized; inspect its state before restarting')
    state = {'status': 'preparing_data', 'pid': os.getpid(), 'source_root': str(ROOT),
             'corruptions': 15, 'severities': [1,2,3,4,5], 'jobs': []}
    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n'); temporary.replace(state_path)
    save()
    try:
        prepare(args.data_root)
        state['status'] = 'checking_real_corruption_pipelines'; save()
        runs = {'resnet18': 'runs/cifar100_resnet18/seed_42/{mode}/outputs',
                'convnext': 'runs/cifar100_convnext/seed_42/{mode}'}
        for backbone in ['resnet18', 'convnext']:
            output = args.output_root / f'{backbone}-pipeline-check'
            command = [sys.executable, '-u', str(ROOT/'scripts/evaluate_matched_shift.py'),
                       '--backbone', backbone, '--run-dir', str(args.campaign/runs[backbone].format(mode='smoke')),
                       '--data-root', str(args.data_root), '--output-dir', str(output),
                       '--mode', 'smoke', '--device', 'cpu', '--corruptions', 'gaussian_noise', '--severities', '1']
            with (args.output_root/f'{backbone}-pipeline-check.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            state['jobs'].append({'id': backbone+'-pipeline-check', 'status':'completed', 'benchmark':False}); save()
        state['status'] = 'waiting_for_training'; save()
        while True:
            training = json.loads((args.campaign/'status.json').read_text())
            table, inventory = collect(args.campaign)
            results = args.campaign/'results'; results.mkdir(exist_ok=True)
            table.to_csv(results/'experiment_results.csv',index=False)
            inventory.to_csv(results/'experiment_inventory.csv',index=False)
            if training['status'] != 'running': break
            try: os.kill(training['pid'],0)
            except ProcessLookupError: raise RuntimeError('Training supervisor is missing; refusing to overlap GPU jobs')
            time.sleep(30)
        for required in ['resnet18_sdm','cifar100']:
            if next(j for j in training['jobs'] if j['id']==required)['status']!='completed':
                raise RuntimeError(f'Required trained model incomplete: {required}')
        state['status']='evaluating_frozen_checkpoints'; save()
        shifts=[]
        for backbone in ['resnet18','convnext']:
            output = args.output_root/backbone
            job={'id':backbone,'status':'running','benchmark':True}; state['jobs'].append(job); save()
            command=[sys.executable,'-u',str(ROOT/'scripts/evaluate_matched_shift.py'),
                     '--backbone',backbone,'--run-dir',str(args.campaign/runs[backbone].format(mode='benchmark')),
                     '--data-root',str(args.data_root),'--output-dir',str(output)]
            with (args.output_root/f'{backbone}-benchmark.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            job['status']='completed'; shifts.append(output); save()
            table, inventory = collect(args.campaign,shifts)
            table.to_csv(args.campaign/'results/experiment_results.csv',index=False)
            inventory.to_csv(args.campaign/'results/experiment_inventory.csv',index=False)
        state['status']='completed'; save()
    except Exception as error:
        state['status']='failed'; state['error']=f'{type(error).__name__}: {error}'; save()
        print(state['error'],flush=True)
        # Use a one-off process supervisor (KeepAlive=false); failures remain failures.
        return 1


if __name__=='__main__':raise SystemExit(main() or 0)
