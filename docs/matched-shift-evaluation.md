# Matched checkpoint evaluation and results collection

PR #23's baseline notebooks are numbered **11/12/13** so PR #29 can use 05–10.
Their legacy artifact filenames (`metrics_05_baselines.json`, etc.) are retained.
Notebook 11 freezes temperature and attainable calibration thresholds before
loading test labels, and records hashes of the checkpoint and output arrays.
Notebooks 12/13 reject changed inputs and assert identical frozen thresholds.
A missing 95% region means no admissions, never threshold zero or a crash.
Older tables need regeneration after the tied-score correction.

Install the shared training environment first: `python -m pip install -r requirements-training.txt`. Use Python 3.12 and a compatible PyTorch/torchvision pair.

## Same model for every method

`python scripts/evaluate_matched_shift.py --help` lists the available options.
The evaluator reuses notebook 04's reviewed metric/SDM functions. ConvNeXt runs
raw softmax, temperature scaling and identity-adaptor SDM on nested 25/100-per-class
calibration tracks. ResNet uses its original temperature fit and the saved
reexpress-sdm fine-100/coarse-20 artifacts; its SDM adaptor may change predictions.
These are distinct method variants and class counts, not interchangeable runs.
ResNet centroid/lower rows select different admission policies on the same fitted
model; both use its central SDM probabilities for probability/ranking metrics.

Before evaluating any condition, the evaluator checks disjoint splits, completion,
original sample identities, and class counts. It replays eight calibration and
eight training examples through the checkpoint and checks logits/embeddings.
ResNet also checks SDM support vectors/labels/IDs and calibration IDs against the
backbone's exports. It records input hashes and frozen calibration rules before
loading clean evaluation labels. Nothing is fitted on SVHN or corruptions.

Data root layout: `cifar100/cifar-100-python/`, `datasets/svhn/`, and `CIFAR-100-C/`.
The corruption directory contains canonical `.npy` files and `labels.npy`; downloads
and extraction are preparation steps, never silently substituted fixtures.
The official archive is [Zenodo record 3555552](https://zenodo.org/records/3555552),
2,918,473,216 bytes, MD5 `11f0ed0f1191edbf9fa23466ae6021d3` (checked October 19, 2026).
Verify it before extracting with Python's `filter="data"`.

```bash
python scripts/evaluate_matched_shift.py --backbone convnext \
  --run-dir /path/to/cifar100_convnext/seed_42/benchmark \
  --data-root /path/to/data-cache --output-dir /path/to/new/convnext-shift
python scripts/evaluate_matched_shift.py --backbone resnet18 \
  --run-dir /path/to/cifar100_resnet18/seed_42/benchmark/outputs \
  --data-root /path/to/data-cache --output-dir /path/to/new/resnet-shift
```

The fixed full scope is all 15 standard corruptions at all five severities, plus
SVHN test as far-OOD. `--mode smoke` uses smoke checkpoints and only 20 shifted
images per condition; `--corruptions gaussian_noise --severities 1` bounds a real
data check. `--skip-corruptions` checks clean/SVHN only. Smoke metadata is explicitly
`pipeline_check`, and the collector excludes it.

ResNet's clean reference is **validation**; ConvNeXt's is **official test**.
CIFAR-100-C derives from official CIFAR-100 test images and labels. Corruption
results therefore constitute held-out test-derived evidence even when ResNet's
clean official test has not been opened; never tune choices from those results.
Corruption/severity slices reuse the same underlying test images; do not pool them as independent observations.
SVHN rows report false admission and ID-versus-OOD AUROC, with no fabricated CIFAR
accuracy. Coarse-20 classification and fine-100 classification remain separate.

Each condition saves raw logits/embeddings, original indices, confidences,
predictions, admission masks, and metrics. Frozen rule coverage and Wilson 95%
intervals accompany accepted accuracy. AUPRC/AUROC and risk/coverage metrics group
ties; AURC uses the documented step integral. No-region accuracy is undefined.

## Durable training and aggregate evidence

`run_cifar_notebook.py` runs existing notebooks 02/03/04 outside the checkout.
Notebook source and driver hashes are exported. It leaves ResNet's clean official
test sealed, saves SDM artifacts, and exports ConvNeXt raw representations.
ResNet training saves an atomic epoch checkpoint including optimizer, scheduler,
random states, history and source/split/configuration identities. Restarting with
the same arguments resumes the existing final-epoch recipe; incompatible identities
fail. Tests prove interrupted and uninterrupted training produce identical weights.

```bash
python scripts/run_cifar_notebook.py resnet18 --mode benchmark \
  --data-root /path/to/data-cache --output-root /path/to/runs
python scripts/run_cifar_notebook.py resnet18_sdm --mode benchmark \
  --data-root /path/to/data-cache --output-root /path/to/runs
```

For the campaign's existing `status.json` and registered run layout:

```bash
python scripts/collect_experiment_results.py --campaign /path/to/campaign \
  --shift-dirs /path/to/convnext-shift /path/to/resnet-shift \
  --output-dir /path/to/results
```

`experiment_results.csv` includes completed benchmark rows only, with checkpoint
hash, seed, class count, calibration context, evaluation population, metric version,
and source file. Raw softmax is repeated per calibration context for paired
comparisons, not as an independent run. Matching clean rows from the shift evaluator
replace native duplicates. Different checkpoints, class counts, calibration tracks,
or populations have different comparison-group IDs. `experiment_inventory.csv`
records queued, running, failed, stale and excluded pipeline states separately.
An empty benchmark table is honest while training is unfinished.

Prerequisites: PR #27's notebook 04 updates and PR #30's ResNet runtime fixes.
No lead approval/merge or full research result is implied by pipeline checks.

The one-off dependency runner `scripts/run_shift_campaign.py` resumes the canonical
archive download, verifies/extracts it, executes real corruption smoke checks on
CPU, and waits for the training supervisor to finish before starting GPU shift
runs. It refreshes the result/inventory CSVs from completed jobs and records any
failure in its own `status.json`. Use a one-off process supervisor with automatic
restarts disabled; this is a batch dependency chain, not an agent schedule.
