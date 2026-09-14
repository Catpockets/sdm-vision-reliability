# Dataset loading

Use the pinned Python environment from [issue #4](https://github.com/Catpockets/sdm-vision-reliability/issues/4)
(Python 3.12.12, torch 2.10.0, torchvision 0.25.0, NumPy 2.3.5, SciPy 1.17.1,
Pillow 12.1.1). Merge that environment PR first. The loader PR adds no dependencies.
Before it merges, tests can run with an existing environment prepared from its
branch. Run the commands below from the repository root; no package installation
or `PYTHONPATH` adjustment is needed.

## Prepare local data

| Dataset | Source and preparation | Local path under `--root` |
| --- | --- | --- |
| CIFAR-100 | [Original dataset](https://www.cs.toronto.edu/~kriz/cifar.html); torchvision downloads and checks its archive with `download=True`. | `cifar-100-python/` |
| CIFAR-100-C | [Official Zenodo record](https://zenodo.org/records/3555552); download and extract manually as below. | `CIFAR-100-C/*.npy` |
| SVHN | [Official cropped digits](http://ufldl.stanford.edu/housenumbers/); torchvision downloads and checks the test file with `download=True`. SciPy reads the `.mat` format. | `test_32x32.mat` |

Downloads are opt-in. With `download=False` (the default), missing data raises an
error instead of using the network. Never commit datasets. Issue #4 ignores the
default root `data/`; use an external directory if testing before that PR merges.

CIFAR-100-C is roughly 2.9 GB to download and needs additional space to extract.
On Linux, for example:

```bash
mkdir -p data
curl --fail --location --retry 3 'https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1' -o data/CIFAR-100-C.tar
echo '11f0ed0f1191edbf9fa23466ae6021d3  data/CIFAR-100-C.tar' | md5sum --check
tar -xf data/CIFAR-100-C.tar -C data
```

The checksum is published in the Zenodo record. On macOS use `md5`; on Windows
use `Get-FileHash -Algorithm MD5` and an archive extractor. Verify the downloaded
checksum before extraction. `--root` must point to the parent of `CIFAR-100-C/`,
not the corruption directory itself. For manual CIFAR/SVHN preparation, put the
official extracted archive/test file at the paths in the table; torchvision
still checks integrity.

## Split and transform contract

- CIFAR-100's official 50,000-image training set is split **within each class**:
  450 training and 50 calibration examples per class by default. The result is
  45,000 training and 5,000 calibration images. A local NumPy generator with
  seed 42 selects membership; it does not change global RNG state. Indices are
  sorted back into source order, and the returned `Subset.indices` can be saved
  with experiment metadata. Change `calibration_per_class` or `seed` explicitly
  and record both. The same versioned data, seed, and settings reproduce the split.
- The official 10,000-image CIFAR-100 test set remains intact. Training and
  calibration are disjoint and exhaust only the official training set. Fit
  calibration and select abstention thresholds on calibration data only.
- CIFAR-100-C exposes 10,000 test images for one corruption/severity pair.
  Severity 1–5 selects successive blocks from each 50,000-image file. Both the
  repeated 50,000-label file and a single 10,000-label block are accepted. The
  module's `CORRUPTIONS` lists the 15 standard benchmark corruptions; extra
  archive corruptions are excluded. Report each corruption/severity separately.
- SVHN exposes only its 26,032-image **test** split, for far-OOD evaluation.
  Its 0–9 digit labels are a separate label space: label 0 is not CIFAR class 0.
  Measure rejection/false acceptance, not CIFAR classification accuracy on SVHN.
- All loaders return `(image, label)`: `float32` RGB tensors of shape
  `(3, 32, 32)` with pixels in `[0, 1]`, and integer labels. There is no resizing,
  normalization, or random augmentation. Apply the chosen backbone's fixed
  preprocessing consistently to all evaluation sets; any learned normalization
  must use training data only. Add training augmentation separately so it cannot
  change calibration/test inputs. Use a seeded `DataLoader` to shuffle training.

CIFAR-100-C is derived from the clean test images, so it is not independent new
test data. Keep both corruption data and SVHN out of training, calibration, model
selection, and threshold tuning. Do not pool their labels or treat a corruption
accuracy target as a guarantee under arbitrary distribution shifts.

## Use the loaders

```python
from torch.utils.data import DataLoader
import torch
from sdm_vision.data import CIFAR100C, load_cifar100, load_svhn

splits = load_cifar100("data", seed=42, download=True)
train_loader = DataLoader(splits["train"], batch_size=64, shuffle=True,
                          generator=torch.Generator().manual_seed(42))
corrupted = CIFAR100C("data", corruption="gaussian_noise", severity=3)
ood = load_svhn("data", download=True)
```

## Check loading

After preparing CIFAR-100-C, this downloads missing CIFAR-100/SVHN data and checks
a small batch from every clean split, SVHN, and the requested corruption blocks:

```bash
python -m sdm_vision.check_data --root data --download --samples 8 --severities 1 2 3 4 5
```

It prints JSON with the split size, full label range, checked sample count, batch
shape, dtype, and sampled pixel range. Expected sizes are 45,000 / 5,000 / 10,000
for CIFAR-100, 26,032 for SVHN, and 10,000 per corruption/severity. Label ranges
are 0–99 for CIFAR and 0–9 for SVHN. Remove `--download` to check offline.
Select multiple corruption types with `--corruptions gaussian_noise fog`.
This is a loading smoke check, not a full dataset audit or model evaluation.

Run the focused offline tests (temporary synthetic fixtures, no downloads):

```bash
python -m unittest discover -s tests -v
```
