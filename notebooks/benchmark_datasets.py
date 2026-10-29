"""Dataset loading and leakage-free split helpers for notebooks 05–10.

Metrics and the SDM derivation remain visible in each generated notebook.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    loader: str
    classes: int
    grayscale: bool = False
    transpose: bool = False
    split: str | None = None

    @property
    def source_url(self):
        return f'https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.{self.loader}.html'


DATASETS = {
    'cifar10': DatasetSpec('cifar10', 'CIFAR-10', 'CIFAR10', 10),
    'mnist': DatasetSpec('mnist', 'MNIST', 'MNIST', 10, grayscale=True),
    'fashion_mnist': DatasetSpec('fashion_mnist', 'Fashion-MNIST', 'FashionMNIST', 10, grayscale=True),
    'kmnist': DatasetSpec('kmnist', 'KMNIST', 'KMNIST', 10, grayscale=True),
    'emnist_balanced': DatasetSpec('emnist_balanced', 'EMNIST Balanced', 'EMNIST', 47, grayscale=True, transpose=True, split='balanced'),
    'svhn': DatasetSpec('svhn', 'SVHN', 'SVHN', 10),
}


def transpose_emnist(image):
    """Display EMNIST's stored rows/columns in upright orientation."""
    return image.transpose(Image.Transpose.TRANSPOSE)


def dataset_transforms(spec, image_size=224):
    prefix = [transforms.Lambda(transpose_emnist)] if spec.transpose else []
    if spec.grayscale:
        prefix.append(transforms.Grayscale(num_output_channels=3))
    normalizing = [
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ]
    resize = transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC)
    if spec.key == 'cifar10':
        augmentation = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    elif spec.grayscale:
        augmentation = [transforms.RandomAffine(degrees=10, translate=(.05, .05))]
    else:
        # Flips can change a digit's meaning. SVHN uses translation only.
        augmentation = [transforms.RandomCrop(32, padding=4)]
    return (
        transforms.Compose(prefix + augmentation + [resize] + normalizing),
        transforms.Compose(prefix + [resize] + normalizing),
    )


def load_dataset(spec, root, *, train, transform, download=True):
    """Expose .targets and .classes consistently without touching test data early."""
    root = Path(root) / spec.key
    loader = getattr(datasets, spec.loader)
    options = {'root': root, 'transform': transform, 'download': download}
    if spec.key == 'svhn':
        options['split'] = 'train' if train else 'test'
    else:
        options['train'] = train
        if spec.split:
            options['split'] = spec.split
    dataset = loader(**options)
    if spec.key == 'svhn':
        # torchvision already maps original SVHN label 10 to digit 0.
        dataset.targets = np.asarray(dataset.labels, dtype=np.int64)
        dataset.classes = [str(i) for i in range(spec.classes)]
    else:
        dataset.targets = np.asarray(dataset.targets, dtype=np.int64)
    if len(dataset.classes) != spec.classes:
        raise ValueError(f'{spec.title}: expected {spec.classes} classes, got {len(dataset.classes)}')
    if len(dataset.targets) and (dataset.targets.min() < 0 or dataset.targets.max() >= spec.classes):
        raise ValueError(f'{spec.title}: targets must be zero-based class indices')
    return dataset


def stratified_splits(targets, classes, *, seed=42, smoke=False):
    """Use fixed per-class calibration sizes; preserve all remaining training data.

    Full runs reserve 25 validation and 100 enlarged calibration images/class.
    Smoke runs use 1 validation, 2 calibration, and 2 training images/class.
    The standard set is always a subset of the enlarged set.
    """
    targets = np.asarray(targets)
    validation_count, standard_count, enlarged_count = (1, 1, 2) if smoke else (25, 25, 100)
    rng = np.random.default_rng(seed)
    parts = {key: [] for key in ('train', 'validation', 'standard', 'enlarged')}
    for class_id in range(classes):
        indices = rng.permutation(np.flatnonzero(targets == class_id))
        if len(indices) < validation_count + enlarged_count + 2:
            raise ValueError(f'Class {class_id} has insufficient images for disjoint splits')
        offset = validation_count
        parts['validation'].extend(indices[:offset])
        parts['standard'].extend(indices[offset:offset + standard_count])
        parts['enlarged'].extend(indices[offset:offset + enlarged_count])
        rest = indices[offset + enlarged_count:]
        parts['train'].extend(rest[:2] if smoke else rest)
    return {key: np.asarray(value, dtype=np.int64) for key, value in parts.items()}


def smoke_test_indices(targets, classes, per_class=2):
    """A deterministic partial test slice; never used for benchmark conclusions."""
    targets = np.asarray(targets)
    return np.concatenate([np.flatnonzero(targets == i)[:per_class] for i in range(classes)])
