"""CIFAR-100 training/calibration splits and held-out vision benchmarks."""

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR100, SVHN
from torchvision.transforms import ToTensor


# The 15 benchmark corruptions; the archive's extra corruptions are not pooled in.
CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)


def load_cifar100(root="data", *, calibration_per_class=50, seed=42, download=False):
    """Return train/calibration/test datasets of float32 RGB tensors in [0, 1].

    Reserve calibration examples from each training class using a local RNG.
    The official test set is never used to choose the split. No augmentation or
    normalization is applied; experiments can add model-specific preprocessing.
    """
    if not isinstance(calibration_per_class, int) or calibration_per_class < 1:
        raise ValueError("calibration_per_class must be a positive integer")
    training = CIFAR100(root, train=True, transform=ToTensor(), download=download)
    targets = np.asarray(training.targets)
    rng = np.random.default_rng(seed)
    train_indices, calibration_indices = [], []
    for label in np.unique(targets):
        indices = rng.permutation(np.flatnonzero(targets == label))
        if calibration_per_class >= len(indices):
            raise ValueError("calibration_per_class must leave training samples in every class")
        calibration_indices.extend(indices[:calibration_per_class].tolist())
        train_indices.extend(indices[calibration_per_class:].tolist())

    return {
        "train": Subset(training, sorted(train_indices)),
        "calibration": Subset(training, sorted(calibration_indices)),
        "test": CIFAR100(root, train=False, transform=ToTensor(), download=download),
    }


class CIFAR100C(Dataset):
    """One corruption and severity of the official CIFAR-100-C test benchmark.

    ``root`` contains ``CIFAR-100-C/<corruption>.npy`` and ``labels.npy``.
    Memory mapping avoids loading the 50,000-image file into RAM. Only the
    requested severity's 10,000 samples are exposed.
    """

    def __init__(self, root="data", *, corruption, severity):
        if corruption not in CORRUPTIONS:
            raise ValueError(f"Unknown corruption {corruption!r}; choose from {CORRUPTIONS}")
        if not isinstance(severity, int) or severity not in range(1, 6):
            raise ValueError("severity must be an integer from 1 to 5")
        directory = Path(root) / "CIFAR-100-C"
        self.images = np.load(directory / f"{corruption}.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(directory / "labels.npy", allow_pickle=False)
        if self.images.shape != (50000, 32, 32, 3) or self.images.dtype != np.uint8:
            raise ValueError("Corruption images must be uint8 with shape (50000, 32, 32, 3)")
        if labels.shape not in ((10000,), (50000,)) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels.npy must contain 10000 or 50000 integer labels")
        if labels.min() < 0 or labels.max() > 99:
            raise ValueError("CIFAR-100-C labels must be between 0 and 99")
        self.offset = (severity - 1) * 10000
        self.targets = labels if len(labels) == 10000 else labels[self.offset:self.offset + 10000]
        self.transform = ToTensor()

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        image = Image.fromarray(self.images[self.offset + index])
        return self.transform(image), int(self.targets[index])


def load_svhn(root="data", *, download=False):
    """Load only the official SVHN test split for far-OOD evaluation.

    Images use the same [0, 1], float32, 3 x 32 x 32 representation as CIFAR.
    Labels are digit IDs 0–9, not CIFAR classes; torchvision maps source label 10
    to 0. Do not score these labels against a CIFAR classifier's class predictions.
    """
    return SVHN(root, split="test", transform=ToTensor(), download=download)
