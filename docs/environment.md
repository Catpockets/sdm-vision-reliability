# Research environment

Use **Python 3.12.12**, recorded in `.python-version`. The small initial stack is
pinned in `requirements.txt`: PyTorch 2.10.0, torchvision 0.25.0, NumPy 2.3.5,
SciPy 1.17.1 (for SVHN `.mat` files), and Pillow 12.1.1. Add notebook, plotting,
or modeling dependencies when an experiment actually needs them.

## Install

Run from the repository root using Python 3.12.12 with `venv` and `pip` available:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

On Windows, use `py -3.12 -m venv .venv` and activate with
`.venv\Scripts\Activate.ps1` in PowerShell. Verify the patch version too.
If you manage Python with [uv](https://docs.astral.sh/uv/guides/install-python/),
`uv python install 3.12.12` and `uv venv --python 3.12.12 --seed .venv` provide
the same starting environment.

Choose **one** PyTorch backend before installing the full requirements.
These use the official [matching PyTorch/torchvision wheel versions](https://pytorch.org/get-started/previous-versions/#v2100).

CPU on Linux or Windows:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
```

NVIDIA GPU on Linux or Windows (CUDA 12.8 wheels, compatible driver required):

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

Apple Silicon macOS (CPU and MPS):

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0
```

Then, for all backends:

```bash
python -m pip install -r requirements.txt
python -m pip check
python scripts/check_environment.py
```

The smoke command imports all five dependencies, converts a small RGB image to a
`float32` tensor, and runs a tensor operation on CPU and every detected CUDA/MPS
device. It prints versions, GPU names, tested devices, and `status: passed` as
JSON; import or device errors produce a nonzero exit. If you expect a GPU, verify
it appears in `tested_devices`; a CPU-only pass does not validate GPU support.
No model weights or datasets are downloaded.

Use a new virtual environment when changing backends. CPU is the portable
baseline; macOS MPS and Windows require validation on those platforms. A system
CUDA toolkit is not needed for these prebuilt wheels; a compatible NVIDIA driver
is. Use the official selector for other accelerators and record deviations from
this baseline.

## Reproducibility

Direct dependency versions are pinned; transitive dependencies and platform
wheels can differ. Save the exact resolved environment with each experiment:

```bash
mkdir -p outputs
python -m pip freeze > outputs/environment.txt
python scripts/check_environment.py > outputs/device.json
```

Record the Git commit, dataset split seed, model seed, transforms, and command
alongside results. Start experiment code with explicit seeds, for example:

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
```

Pass an explicitly seeded `torch.Generator` to shuffled data loaders. Seed
NumPy `default_rng` instances separately. With worker processes, seed Python and
NumPy in `worker_init_fn` using `torch.initial_seed() % 2**32`.
For stricter runs, enable `torch.use_deterministic_algorithms(True)` and follow
the [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.10/notes/randomness.html)
for backend requirements; unsupported operations may raise errors or run slower.
Seeds do not guarantee identical results across releases, devices, or platforms.

## Local ignore rules

Keep ignore rules local; do not commit a `.gitignore` for this setup. Find this
clone's exclude file with `git rev-parse --git-path info/exclude` and add:

```gitignore
.venv/
venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.env
.env.*
!.env.example
/data/
/datasets/
/checkpoints/
/outputs/
/runs/
/logs/
/wandb/
/mlruns/
```

Configure these exclusions before creating the environment or downloading data.
They apply only to this clone and are not committed. Alternatively, store data
and generated outputs outside the repository. Keep source notebooks,
configuration, small reviewed test fixtures, and dependency manifests in Git.
Review staged files before committing; ignore rules do not remove tracked files.
