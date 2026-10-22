"""Exercise notebook 04's actual training controller with a small CPU model."""
import json
import random
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class ConvNeXtCheckpointTest(unittest.TestCase):
    def run_training(self, root, *, interrupt=False, config_change=False, workers=0):
        notebook = json.loads((ROOT / 'notebooks/04_cifar100_sdm_calibration_comparison.ipynb').read_text())
        source = ''.join(next(c for c in notebook['cells'] if c['id'] == '1e00ebcf')['source'])
        controller = source[source.index('checkpoint_path ='):]
        torch.manual_seed(7); np.random.seed(7); random.seed(7)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.3), torch.nn.Linear(4, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(24.).reshape(8, 3) / 24,
            torch.arange(8) % 2), batch_size=2, shuffle=True,
            generator=torch.Generator().manual_seed(11), num_workers=workers, persistent_workers=False)
        calls = 0
        def epoch(loader_arg, *, training, max_batches):
            nonlocal calls
            if training:
                calls += 1
                if interrupt and calls == 2:
                    raise InterruptedError('simulated process interruption after epoch one')
                model.train()
                for x, y in loader_arg:
                    optimizer.zero_grad()
                    jitter = float(np.random.random() + random.random()) * .001
                    loss = torch.nn.functional.cross_entropy(model(x + jitter), y)
                    loss.backward(); optimizer.step()
                return {'loss': float(loss.detach()), 'top1_accuracy': .5}
            # Best epoch deliberately precedes final epoch.
            return {'loss': 1., 'top1_accuracy': .9 - (scheduler.last_epoch + 1) * .1}
        ns = dict(torch=torch, np=np, random=random, time=time, Path=Path, model=model,
            optimizer=optimizer, scheduler=scheduler, scaler=torch.amp.GradScaler('cuda', enabled=False),
            train_loader=loader, validation_loader=loader, classification_epoch=epoch,
            RUN_DIR=Path(root), DEVICE=torch.device('cpu'), FORCE_RETRAIN=False,
            EPOCHS=3, MAX_TRAIN_BATCHES=None, MAX_EXPORT_BATCHES=None,
            CONFIG={'epochs': 3, 'seed': 7, 'batch_size': 4 if config_change else 2},
            train_aug_base=SimpleNamespace(classes=['a', 'b'], transform='fixed-augmentation-v1'),
            train_indices=np.arange(8), validation_indices=np.arange(8, 10),
            calibration_25_indices=np.arange(10, 12), calibration_100_indices=np.arange(10, 14))
        exec(compile(controller, 'notebook04-training-controller', 'exec'), ns)
        return ns

    def test_interrupted_training_matches_uninterrupted_and_loads_best_epoch(self):
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as resumed:
            expected = self.run_training(full)
            with self.assertRaises(InterruptedError): self.run_training(resumed, interrupt=True)
            actual = self.run_training(resumed)
            self.assertEqual([row['epoch'] for row in actual['history']], [1, 2, 3])
            for key, value in expected['model'].state_dict().items():
                torch.testing.assert_close(value, actual['model'].state_dict()[key], rtol=0, atol=0)
            left = torch.load(Path(full) / 'epoch_checkpoint.pt', weights_only=False)
            right = torch.load(Path(resumed) / 'epoch_checkpoint.pt', weights_only=False)
            for key, value in left['model_state_dict'].items():
                torch.testing.assert_close(value, right['model_state_dict'][key], rtol=0, atol=0)
            self.assertEqual(left['history'], right['history'])
            best = torch.load(Path(resumed) / 'best_checkpoint.pt', weights_only=False)
            self.assertTrue(best['training_complete']); self.assertEqual(best['completed_epochs'], 3)
            self.assertEqual(best['best_epoch'], 1)
            self.assertEqual(len(best['history']), 3)
            self.assertNotEqual(float(next(iter(right['model_state_dict'].values())).sum()),
                                float(next(iter(best['model_state_dict'].values())).sum()))
            # Completed re-use neither trains nor changes recorded history.
            self.assertEqual(self.run_training(resumed)['history'], actual['history'])

    def test_resume_rejects_changed_config_and_legacy_incomplete_checkpoints(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(InterruptedError): self.run_training(root, interrupt=True)
            with self.assertRaisesRegex(ValueError, 'identity|configuration'):
                self.run_training(root, config_change=True)
        with tempfile.TemporaryDirectory() as root:
            torch.save({'training_complete': False, 'history': [{'epoch': 1}], 'config': {'epochs': 3, 'seed': 7, 'batch_size': 2}, 'class_names': ['a', 'b']}, Path(root) / 'best_checkpoint.pt')
            with self.assertRaisesRegex(ValueError, 'incomplete|Incomplete'):
                self.run_training(root)


if __name__ == '__main__': unittest.main()
