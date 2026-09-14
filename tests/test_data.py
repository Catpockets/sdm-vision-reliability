"""Offline checks for split leakage, corruption indexing, and input compatibility."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.io import savemat
import torch
from torch.utils.data import Dataset
from torchvision.datasets import SVHN

from sdm_vision import check_data
from sdm_vision.data import CIFAR100C, CORRUPTIONS, load_cifar100, load_svhn


class TinyCIFAR(Dataset):
    """Replace only CIFAR's file I/O; retain the loader's real transform."""

    def __init__(self, root, *, train, transform, download):
        self.train = train
        self.transform = transform
        self.targets = np.repeat(np.arange(100), 5 if train else 1).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.fromarray(np.full((32, 32, 3), index % 256, dtype=np.uint8))
        return self.transform(image), self.targets[index]


class SplitTests(unittest.TestCase):
    @patch("sdm_vision.data.CIFAR100", TinyCIFAR)
    def test_stratified_split_is_disjoint_exhaustive_and_keeps_test_held_out(self):
        splits = load_cifar100(calibration_per_class=2)
        train, calibration, test = (splits[key] for key in ("train", "calibration", "test"))
        self.assertEqual((len(train), len(calibration), len(test)), (300, 200, 100))
        self.assertFalse(set(train.indices) & set(calibration.indices))
        self.assertEqual(set(train.indices) | set(calibration.indices), set(range(500)))
        self.assertIs(train.dataset, calibration.dataset)
        self.assertTrue(train.dataset.train)
        self.assertFalse(test.train)
        labels = np.asarray(calibration.dataset.targets)[calibration.indices]
        np.testing.assert_array_equal(np.bincount(labels), np.full(100, 2))
        image, label = calibration[0]
        self.assertEqual(image.shape, (3, 32, 32))
        self.assertEqual(image.dtype, torch.float32)
        self.assertIsInstance(label, int)

    @patch("sdm_vision.data.CIFAR100", TinyCIFAR)
    def test_seed_repeats_membership_without_changing_global_rng(self):
        before = np.random.get_state()
        first = load_cifar100(calibration_per_class=2, seed=42)
        repeated = load_cifar100(calibration_per_class=2, seed=42)
        other = load_cifar100(calibration_per_class=2, seed=43)
        self.assertEqual(first["calibration"].indices, repeated["calibration"].indices)
        self.assertNotEqual(first["calibration"].indices, other["calibration"].indices)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    @patch("sdm_vision.data.CIFAR100", TinyCIFAR)
    def test_invalid_calibration_sizes(self):
        for size in (0, -1, 1.5, 5, 6):
            with self.subTest(size=size), self.assertRaises(ValueError):
                load_cifar100(calibration_per_class=size)


class FileDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "CIFAR-100-C"
        self.directory.mkdir()
        # A sparse on-disk array with the real archive shape; only boundary
        # images are populated, so tests need no downloaded dataset.
        images = np.lib.format.open_memmap(
            self.directory / "gaussian_noise.npy", mode="w+", dtype=np.uint8,
            shape=(50000, 32, 32, 3),
        )
        for block in range(5):
            images[block * 10000] = block * 50
            images[(block + 1) * 10000 - 1] = block * 50 + 1
        images.flush()
        del images
        self.labels = np.tile(np.arange(100, dtype=np.int64), 500)
        np.save(self.directory / "labels.npy", self.labels)

    def make_svhn(self):
        path = self.root / "test_32x32.mat"
        savemat(path, {"X": np.full((32, 32, 3, 2), 255, dtype=np.uint8),
                       "y": np.array([[10], [9]], dtype=np.uint8)})
        checksum = hashlib.md5(path.read_bytes()).hexdigest()
        return patch.dict(SVHN.split_list, {"test": ("https://unused.invalid/test", path.name, checksum)})

    def test_all_severity_boundaries_and_labels(self):
        for severity in range(1, 6):
            with self.subTest(severity=severity):
                dataset = CIFAR100C(self.root, corruption="gaussian_noise", severity=severity)
                self.assertEqual(len(dataset), 10000)
                self.assertIsInstance(dataset.images, np.memmap)
                first, first_label = dataset[0]
                last, last_label = dataset[9999]
                torch.testing.assert_close(first, torch.full((3, 32, 32), (severity - 1) * 50 / 255))
                torch.testing.assert_close(last, torch.full((3, 32, 32), ((severity - 1) * 50 + 1) / 255))
                self.assertEqual((first_label, last_label), (0, 99))
                for index in (-1, 10000):
                    with self.assertRaises(IndexError):
                        dataset[index]

    def test_repeated_label_blocks_are_sliced_at_selected_severity(self):
        np.save(self.directory / "labels.npy", np.repeat(np.arange(5), 10000))
        dataset = CIFAR100C(self.root, corruption="gaussian_noise", severity=4)
        self.assertEqual(dataset[0][1], 3)
        self.assertEqual(dataset[9999][1], 3)

    def test_single_label_block(self):
        np.save(self.directory / "labels.npy", self.labels[:10000])
        dataset = CIFAR100C(self.root, corruption="gaussian_noise", severity=5)
        self.assertEqual(dataset[9999][1], 99)

    def test_invalid_corruption_severity_and_missing_files(self):
        for corruption, severity in (("../outside", 1), ("gaussian_noise", 0),
                                     ("gaussian_noise", 6), ("gaussian_noise", 1.5)):
            with self.subTest(corruption=corruption, severity=severity), self.assertRaises(ValueError):
                CIFAR100C(self.root, corruption=corruption, severity=severity)
        self.assertEqual(len(CORRUPTIONS), 15)
        with self.assertRaises(FileNotFoundError):
            CIFAR100C(self.root, corruption="fog", severity=1)

    def test_invalid_labels(self):
        for labels in (np.zeros(10, dtype=int), np.zeros(10000),
                       np.full(10000, -1), np.full(10000, 100)):
            with self.subTest(shape=labels.shape, dtype=labels.dtype):
                np.save(self.directory / "labels.npy", labels)
                with self.assertRaises(ValueError):
                    CIFAR100C(self.root, corruption="gaussian_noise", severity=1)

    def test_invalid_image_shape_and_dtype(self):
        for images in (np.zeros((10, 32, 32, 3), dtype=np.uint8),
                       np.zeros((50000, 32, 32, 3), dtype=bool)):
            np.save(self.directory / "gaussian_noise.npy", images)
            with self.assertRaises(ValueError):
                CIFAR100C(self.root, corruption="gaussian_noise", severity=1)

    def test_real_svhn_reader_maps_zero_and_uses_only_test_file(self):
        with self.make_svhn():
            dataset = load_svhn(self.root)
        self.assertEqual(dataset.split, "test")
        self.assertEqual(len(dataset), 2)
        image, label = dataset[0]
        self.assertEqual(label, 0)
        self.assertEqual(dataset[1][1], 9)
        torch.testing.assert_close(image, torch.ones((3, 32, 32)))

    @patch("sdm_vision.data.CIFAR100", TinyCIFAR)
    def test_smoke_report_includes_every_split_and_requested_severity(self):
        output = io.StringIO()
        args = ["check_data", "--root", str(self.root), "--samples", "2",
                "--calibration-per-class", "2", "--severities", "1", "5"]
        with self.make_svhn(), patch("sys.argv", args), contextlib.redirect_stdout(output):
            check_data.main()
        report = json.loads(output.getvalue())["datasets"]
        self.assertEqual(len(report), 6)
        self.assertEqual(report["calibration"]["size"], 200)
        self.assertEqual(report["svhn_ood"]["label_range"], [0, 9])
        self.assertEqual(report["train"]["label_range"], [0, 99])
        self.assertEqual(report["cifar100c/gaussian_noise/5"]["size"], 10000)
        for row in report.values():
            self.assertEqual(row["batch_shape"], [2, 3, 32, 32])
            self.assertEqual(row["samples_checked"], 2)


if __name__ == "__main__":
    unittest.main()
