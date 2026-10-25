"""Prove interrupted training resumes to the same final weights and RNG state."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from checkpoint_training import run_resnet_epochs


def fixture(interrupt_after=None):
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    calls = 0
    def train():
        nonlocal calls
        if interrupt_after is not None and calls == interrupt_after:
            raise RuntimeError('simulated interruption')
        calls += 1
        inputs = torch.rand(8, 3)
        labels = torch.randint(2, (8,))
        loss = torch.nn.functional.cross_entropy(model(inputs), labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        return loss.item(), .5
    def predict(images):
        return model(images).detach(), None
    return dict(SEED=42, EPOCHS=5, BATCH_SIZE=8, MAX_LR=.1, WEIGHT_DECAY=0,
                _source_digest='same-verified-source', train_idx=np.arange(8),
                val_idx=np.arange(8, 16), cal_idx=np.arange(16, 24),
                device=torch.device('cpu'), model=model, optimizer=optimizer, scheduler=scheduler,
                train_one_epoch=train, predict=predict, val_images=torch.ones(4, 3),
                val_labels=torch.tensor([0, 1, 0, 1]))


class DurableTrainingTest(unittest.TestCase):
    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as folder:
            full = fixture(); run_resnet_epochs(full, Path(folder) / 'full.pt')
            interrupted = fixture(interrupt_after=2)
            path = Path(folder) / 'resumed.pt'
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                run_resnet_epochs(interrupted, path)
            self.assertEqual(len(torch.load(path, weights_only=False)['history']), 2)
            resumed = fixture(); run_resnet_epochs(resumed, path)
            for actual, expected in zip(resumed['model'].parameters(), full['model'].parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            np.testing.assert_array_equal(resumed['history'].values, full['history'].values)
            self.assertEqual(resumed['scheduler'].state_dict(), full['scheduler'].state_dict())

    def test_incompatible_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'resume.pt'
            original = fixture(); run_resnet_epochs(original, path)
            changed = fixture(); changed['MAX_LR'] = .2
            with self.assertRaisesRegex(ValueError, 'settings/source/splits'):
                run_resnet_epochs(changed, path)


if __name__ == '__main__': unittest.main()
