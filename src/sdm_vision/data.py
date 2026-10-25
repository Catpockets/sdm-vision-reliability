"""CIFAR-10 and Imagenette, both stored as uint8 images of 36 x 36 pixels (the network sees 32 x 32 crops)."""

from pathlib import Path
import hashlib
import pickle
import tarfile
import urllib.request

import numpy as np
from PIL import Image

STORE_SIZE = 36  # 32 x 32 crops of a 36 x 36 image, as in notebook 08 with RESOLUTION = 32

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_SHA256 = "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"
IMAGENETTE_CLASSES = {  # ImageNet identifiers, in the folder (sorted) order used as labels
    "n01440764": "tench", "n02102040": "English_springer", "n02979186": "cassette_player",
    "n03000684": "chain_saw", "n03028079": "church", "n03394916": "French_horn",
    "n03417042": "garbage_truck", "n03425413": "gas_pump", "n03445777": "golf_ball", "n03888257": "parachute",
}


def _download(url, archive, check):
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    assert check(archive.read_bytes()), f"{archive} is not the expected file; delete it and rerun"
    with tarfile.open(archive) as file:
        file.extractall(archive.parent, filter="data")


def load_cifar10(data_dir):
    """The 50,000 CIFAR-10 training images (the official test batch stays untouched), reflect-padded to 36 x 36."""
    data_dir = Path(data_dir)
    folder = data_dir / "cifar-10-batches-py"
    if not folder.exists():
        _download(CIFAR10_URL, data_dir / "cifar-10-python.tar.gz", lambda raw: hashlib.md5(raw).hexdigest() == CIFAR10_MD5)
    images, labels = [], []
    for i in range(1, 6):
        with (folder / f"data_batch_{i}").open("rb") as file:
            batch = pickle.load(file, encoding="latin1")
        images.append(batch["data"].reshape(-1, 3, 32, 32))
        labels.extend(batch["labels"])
    with (folder / "batches.meta").open("rb") as file:
        class_names = pickle.load(file, encoding="latin1")["label_names"]
    pad = (STORE_SIZE - 32) // 2
    images = np.pad(np.concatenate(images), ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")
    assert len(images) == 50000
    return images, np.array(labels), class_names


def _load_image(path, size):
    # Shorter side to `size` pixels, then the central square (notebook 08).
    image = Image.open(path).convert("RGB")
    scale = size / min(image.size)
    image = image.resize((max(size, round(image.width * scale)), max(size, round(image.height * scale))), Image.BICUBIC)
    left, top = (image.width - size) // 2, (image.height - size) // 2
    return np.asarray(image.crop((left, top, left + size, top + size))).transpose(2, 0, 1)


def load_imagenette(data_dir):
    """The 9,469 images of Imagenette's train folder at 36 x 36 (its val folder stays untouched).

    Uses the same cache file as notebook 08 with RESOLUTION = 32.
    """
    data_dir = Path(data_dir)
    wnids = sorted(IMAGENETTE_CLASSES)
    class_names = [IMAGENETTE_CLASSES[wnid] for wnid in wnids]
    cache = data_dir / f"imagenette2-160_{STORE_SIZE}px.npz"
    if cache.exists():
        with np.load(cache) as saved:
            images, labels = saved["train_images"], saved["train_labels"]
    else:
        folder = data_dir / "imagenette2-160"
        if not folder.exists():
            _download(IMAGENETTE_URL, data_dir / "imagenette2-160.tgz", lambda raw: hashlib.sha256(raw).hexdigest() == IMAGENETTE_SHA256)
        splits = {}
        for name in ["train", "val"]:
            files = sorted(folder.glob(f"{name}/*/*.JPEG"))
            splits[name] = (np.stack([_load_image(file, STORE_SIZE) for file in files]),
                            np.array([wnids.index(file.parent.name) for file in files]),
                            np.array([str(file.relative_to(folder)) for file in files]))
        np.savez(cache, train_images=splits["train"][0], train_labels=splits["train"][1], train_files=splits["train"][2],
                 test_images=splits["val"][0], test_labels=splits["val"][1], test_files=splits["val"][2])
        images, labels = splits["train"][0], splits["train"][1]
    assert len(images) == 9469 and images.shape[1:] == (3, STORE_SIZE, STORE_SIZE)
    return images, labels, class_names


LOADERS = {"cifar10": load_cifar10, "imagenette": load_imagenette}


def fixed_split(labels, train_per_class=500, cal_per_class=200, val_per_class=150, seed=42):
    """One split per dataset, the same for every class subset: within each class, shuffle with `seed`,
    then the first images go to validation, the next to calibration, the next to training."""
    rng = np.random.default_rng(seed)
    split = {"train": [], "calibration": [], "validation": []}
    for class_id in np.unique(labels):
        shuffled = rng.permutation(np.flatnonzero(labels == class_id))
        assert len(shuffled) >= val_per_class + cal_per_class + train_per_class, f"class {class_id} has only {len(shuffled)} images"
        split["validation"].extend(shuffled[:val_per_class])
        split["calibration"].extend(shuffled[val_per_class : val_per_class + cal_per_class])
        split["train"].extend(shuffled[val_per_class + cal_per_class : val_per_class + cal_per_class + train_per_class])
    split = {name: np.array(index) for name, index in split.items()}
    assert not (set(split["train"]) & set(split["calibration"]) or set(split["train"]) & set(split["validation"])
                or set(split["calibration"]) & set(split["validation"]))
    return split


def class_order(chain, number_of_classes=10):
    """The order in which chain `chain` adds classes: the first K entries are the classes of the K-class task."""
    return np.random.default_rng(chain).permutation(number_of_classes)
