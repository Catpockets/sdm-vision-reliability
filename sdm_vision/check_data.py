"""Report dataset sizes, full label ranges, and a small batch from each split."""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import CIFAR100C, CORRUPTIONS, load_cifar100, load_svhn


def summarize(dataset, samples):
    images, labels = next(iter(DataLoader(dataset, batch_size=samples, shuffle=False)))
    if isinstance(dataset, Subset):
        targets = np.asarray(dataset.dataset.targets)[dataset.indices]
    elif hasattr(dataset, "targets"):
        targets = np.asarray(dataset.targets)
    else:
        targets = np.asarray(dataset.labels)
    if images.shape[1:] != (3, 32, 32) or images.dtype != torch.float32:
        raise ValueError(f"Unexpected image batch: {images.shape}, {images.dtype}")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Images must contain finite values in [0, 1]")
    return {
        "size": len(dataset),
        "label_range": [int(targets.min()), int(targets.max())],
        "samples_checked": len(labels),
        "batch_shape": list(images.shape),
        "dtype": str(images.dtype),
        "pixel_range": [images.min().item(), images.max().item()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    parser.add_argument("--download", action="store_true", help="Download CIFAR-100 and SVHN only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-per-class", type=int, default=50)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--corruptions", nargs="+", choices=CORRUPTIONS, default=["gaussian_noise"])
    parser.add_argument("--severities", nargs="+", type=int, choices=range(1, 6), default=[1])
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    # Fail before downloading if the manually prepared corruption files are absent.
    for corruption in args.corruptions:
        CIFAR100C(args.root, corruption=corruption, severity=args.severities[0])
    datasets = load_cifar100(args.root, calibration_per_class=args.calibration_per_class,
                             seed=args.seed, download=args.download)
    datasets["svhn_ood"] = load_svhn(args.root, download=args.download)
    report = {name: summarize(dataset, args.samples) for name, dataset in datasets.items()}
    for corruption in args.corruptions:
        for severity in args.severities:
            dataset = CIFAR100C(args.root, corruption=corruption, severity=severity)
            report[f"cifar100c/{corruption}/{severity}"] = summarize(dataset, args.samples)
    print(json.dumps({"seed": args.seed, "datasets": report}, indent=2))


if __name__ == "__main__":
    main()
