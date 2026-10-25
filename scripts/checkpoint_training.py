"""Resume the existing ResNet notebook's final-epoch recipe at epoch boundaries."""
import hashlib
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch


def run_resnet_epochs(namespace, path):
    n = namespace
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    settings = {key: n[key] for key in ['SEED', 'EPOCHS', 'BATCH_SIZE', 'MAX_LR', 'WEIGHT_DECAY']}
    settings['source_digest'] = n['_source_digest']
    settings['split_digest'] = hashlib.sha256(b''.join(np.asarray(n[k], dtype=np.int64).tobytes()
                                             for k in ['train_idx', 'val_idx', 'cal_idx'])).hexdigest()
    history = []
    if path.exists():
        checkpoint = torch.load(path, map_location=n['device'], weights_only=False)
        if checkpoint['settings'] != settings:
            raise ValueError('Resume checkpoint settings/source/splits changed; use a distinct run directory')
        n['model'].load_state_dict(checkpoint['model'])
        n['optimizer'].load_state_dict(checkpoint['optimizer'])
        n['scheduler'].load_state_dict(checkpoint['scheduler'])
        history = checkpoint['history']
        random.setstate(checkpoint['python_rng']); np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if 'mps_rng' in checkpoint: torch.mps.set_rng_state(checkpoint['mps_rng'].cpu())
        if 'cuda_rng' in checkpoint: torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
        print(f'Resuming after durable epoch {len(history)}/{n["EPOCHS"]}', flush=True)
    for epoch in range(len(history) + 1, n['EPOCHS'] + 1):
        started = time.monotonic()
        train_loss, train_accuracy = n['train_one_epoch']()
        logits, _ = n['predict'](n['val_images'])
        labels = n['val_labels'].cpu()
        val_loss = torch.nn.functional.cross_entropy(logits, labels).item()
        val_accuracy = (logits.argmax(1) == labels).float().mean().item()
        history.append({'epoch': epoch, 'train loss': train_loss, 'val loss': val_loss,
                        'train acc': train_accuracy, 'val acc': val_accuracy})
        checkpoint = {'settings': settings, 'model': n['model'].state_dict(),
                      'optimizer': n['optimizer'].state_dict(), 'scheduler': n['scheduler'].state_dict(),
                      'history': history, 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                      'torch_rng': torch.get_rng_state()}
        if n['device'].type == 'mps': checkpoint['mps_rng'] = torch.mps.get_rng_state()
        if n['device'].type == 'cuda': checkpoint['cuda_rng'] = torch.cuda.get_rng_state_all()
        temporary = path.with_suffix('.tmp')
        torch.save(checkpoint, temporary); temporary.replace(path)
        print(f'epoch {epoch:2d}/{n["EPOCHS"]} train={train_accuracy:.1%} validation={val_accuracy:.1%} '
              f'checkpoint saved ({time.monotonic()-started:.0f}s)', flush=True)
    n['history'] = pd.DataFrame(history).set_index('epoch')
