"""Execute pinned CIFAR-100/ResNet source notebooks outside the checkout."""
import argparse
import hashlib
from checkpoint_training import run_resnet_epochs
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('kind', choices=['cifar100', 'resnet18', 'resnet18_sdm'])
parser.add_argument('--mode', choices=['smoke', 'benchmark'], required=True)
parser.add_argument('--data-root', type=Path, required=True)
parser.add_argument('--output-root', type=Path, required=True)
parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--source-revision')
parser.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='auto')
args = parser.parse_args()
SOURCE = args.source_root.resolve()
DATA = args.data_root.resolve() / 'cifar100'
smoke = args.mode == 'smoke'
family = 'cifar100_convnext' if args.kind == 'cifar100' else 'cifar100_resnet18'
work = args.output_root.resolve() / family / 'seed_42' / args.mode
work.mkdir(parents=True, exist_ok=True)
if args.kind == 'cifar100':
    filename = '04_cifar100_sdm_calibration_comparison.ipynb'
    os.environ['SDM_DATA_DIR'] = str(DATA)
    os.environ['SDM_RUN_DIR'] = str(work)
    os.environ['SDM_RUN_MODE'] = args.mode
    os.environ['SDM_FORCE_RETRAIN'] = '0'
    os.chdir(SOURCE)
else:
    filename = '02_cifar100_cnn_resnet18.ipynb' if args.kind == 'resnet18' else '03_sdm_resnet18.ipynb'
    (work / 'notebooks').mkdir(exist_ok=True)
    (work / 'outputs').mkdir(exist_ok=True)
    if not (work / 'data').exists():
        (work / 'data').symlink_to(DATA, target_is_directory=True)
    os.chdir(work / 'notebooks')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda: plt.close('all')
import numpy as np
namespace = {'display': lambda *a: None, '__name__': '__main__', '_source_digest': hashlib.sha256((SOURCE / 'notebooks' / filename).read_bytes()).hexdigest()}
notebook = json.loads((SOURCE / 'notebooks' / filename).read_text())
print(f'{args.kind}: mode={args.mode}; source={SOURCE}; output={work}', flush=True)
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    code = ''.join(cell['source'])
    if code.lstrip().startswith('%pip'):
        print('Installation cell skipped: dependencies are installed and checked before execution.', flush=True)
        continue
    print('Executing', filename, cell['id'], flush=True)
    if args.kind == 'resnet18' and cell['id'] == 'cell-17':
        run_resnet_epochs(namespace, work / 'epoch_checkpoint.pt')
    else:
        exec(compile(code, f"{filename}:{cell['id']}", 'exec'), namespace)
    if args.device != 'auto' and cell['id'] in {'7c87267e', 'cell-02'}:
        namespace['DEVICE' if args.kind == 'cifar100' else 'device'] = namespace['torch'].device(args.device)
        if args.kind == 'cifar100':
            namespace['USE_AMP'] = args.device == 'cuda'
            namespace['CONFIG']['device'] = args.device
        print('Execution device override:', args.device, flush=True)
    if args.kind == 'resnet18' and cell['id'] == 'cell-02' and smoke:
        namespace['EPOCHS'] = 1
    if args.kind == 'resnet18' and cell['id'] == 'cell-07' and smoke:
        # Keep every class in a disjoint, small real-data pipeline check.
        for prefix, idx, count in [('train', 'train_idx', 2), ('val', 'val_idx', 2), ('cal', 'cal_idx', 2)]:
            labels = namespace['y_' + prefix]
            chosen = np.concatenate([np.flatnonzero(labels == i)[:count] for i in range(100)])
            namespace['x_' + prefix] = namespace['x_' + prefix][chosen]
            namespace['y_' + prefix] = labels[chosen]
            namespace[idx] = namespace[idx][chosen]
            if prefix == 'val': namespace['g_val'] = namespace['g_val'][chosen]
    if args.kind == 'resnet18_sdm' and cell['id'] == 'cell-02':
        namespace['SAVE_SDM_MODELS'] = True
        if smoke: namespace['SDM_EPOCHS'] = 1
if args.kind == 'cifar100':
    for label, arrays, indices in [
        ('reference', namespace['reference'], namespace['train_indices']),
        ('calibration_enlarged', namespace['calibration_all'], namespace['calibration_100_indices']),
        ('test', namespace['test'], np.arange(len(namespace['test']['targets']))),
    ]:
        np.savez_compressed(work / f'{label}_outputs.npz', **arrays, dataset_indices=indices[:len(arrays['targets'])])
    np.savez_compressed(work / 'calibration_standard_outputs.npz', **namespace['calibration_sets']['standard_25'])
    np.savez_compressed(work / 'split_indices.npz', train=namespace['train_indices'], validation=namespace['validation_indices'], standard=namespace['calibration_25_indices'], enlarged=namespace['calibration_100_indices'])
    print(namespace['temperature_table'].to_string(index=False), flush=True)
    print(namespace['metrics_table'].to_string(), flush=True)
metadata = {
    'kind': args.kind, 'mode': args.mode,
    'status': 'pipeline_check' if smoke else 'completed_full_run',
    'source_revision': args.source_revision,
    'source_notebook_sha256': namespace['_source_digest'],
    'execution_driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_notebook': str(SOURCE / 'notebooks' / filename),
    'settings': {'backbone_seed': 42, 'sdm_seed': namespace.get('SDM_SEED'), 'epochs': namespace.get('EPOCHS', namespace.get('SDM_EPOCHS'))},
    'resnet_final_test': 'untouched; existing ResNet notebooks evaluate validation only',
    'numpy': np.__version__,
}
(work / f'{args.kind}_execution.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('Completed', args.kind, args.mode, flush=True)
