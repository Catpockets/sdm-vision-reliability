# Dataset reliability notebooks 05–10

Each notebook trains its own ImageNet-pretrained ConvNeXt-Tiny classifier and reproduces the updated notebook 04 experiment. The notebooks are independent; none needs outputs from a previous dataset notebook.

| Notebook | Dataset | Classes | Dataset-specific handling |
|---|---|---:|---|
| [05](../notebooks/05_cifar10_sdm_calibration_comparison.ipynb) | CIFAR-10 | 10 | RGB natural images; training crops and horizontal flips |
| [06](../notebooks/06_mnist_sdm_calibration_comparison.ipynb) | MNIST | 10 | Grayscale replicated to RGB; mild affine augmentation, no flips |
| [07](../notebooks/07_fashion_mnist_sdm_calibration_comparison.ipynb) | Fashion-MNIST | 10 | Grayscale clothing images; mild affine augmentation |
| [08](../notebooks/08_kmnist_sdm_calibration_comparison.ipynb) | KMNIST | 10 | Kuzushiji characters; preserve class names, no flips |
| [09](../notebooks/09_emnist_balanced_sdm_calibration_comparison.ipynb) | EMNIST Balanced | 47 | Explicit `balanced` split, upright transpose, zero-based labels |
| [10](../notebooks/10_svhn_sdm_calibration_comparison.ipynb) | SVHN | 10 | Official train/test splits, no extra split, digit 0 mapped by torchvision, no flips |

SVHN here is an in-distribution classification experiment. These notebooks do not replace separate SVHN-as-OOD or corrupted-CIFAR evaluations. The MNIST-derived datasets are distinct benchmarks, not alternative splits of one training run. Cross-dataset metric differences also reflect different class counts and task difficulty.

## Start a notebook

Use Python 3.12. Activate the training environment containing a compatible PyTorch/torchvision pair, then from the repository root:

```bash
python -m pip install -r requirements-training.txt
python -m jupyterlab notebooks/05_cifar10_sdm_calibration_comparison.ipynb
```

For NVIDIA CUDA wheels, follow the [official installation selector](https://pytorch.org/get-started/locally/) for the installed driver and hardware. CPU and MPS fallbacks work but full ConvNeXt runs are expensive. SciPy is required for the SVHN `.mat` loader.

Use a separate kernel per notebook, inspect the class gallery, and choose **Restart Kernel and Run All Cells**. The default is a full 20-epoch experiment, not a smoke run. Completed compatible checkpoints are reused; incomplete training restarts from the beginning. `SDM_FORCE_RETRAIN=1` explicitly replaces the previous model in that run directory. No accuracy target is promised and weak runs are retained.

The environment variables `SDM_EPOCHS` and `SDM_BATCH_SIZE` can set training resources before launch. Changing them requires a new output root or explicit retraining. Fix these choices before viewing final-test results.

## Smoke checks and headless execution

Start Jupyter in smoke mode:

```bash
SDM_RUN_MODE=smoke python -m jupyterlab notebooks/06_mnist_sdm_calibration_comparison.ipynb
```

Or execute all cells using the same Python interpreter without a Jupyter kernel:

```bash
python scripts/run_dataset_notebook.py --dataset mnist --mode smoke
```

The headless runner saves the same tables and plots externally and leaves notebook outputs cleared. `--dataset` accepts `cifar10`, `mnist`, `fashion_mnist`, `kmnist`, `emnist_balanced`, or `svhn`. It does not launch the other five datasets. Use `--mode benchmark` for a full run.

Smoke mode keeps every class but uses only 2 training images, 1 validation image, and nested 1/2 calibration images per class, plus 2 official test examples per class. Training is capped at two batches for a pipeline check. The full-run track names `standard_25` and `enlarged_100` are retained, while the configuration, split manifest and fit diagnostics report the actual smoke sizes. Smoke metrics must not be interpreted as research results.

The first run downloads the official data and pretrained weights; smoke mode still requires the full dataset download. EMNIST's upstream archive includes all splits and is particularly large. To use prepared data and pretrained weights without downloading:

```bash
python scripts/run_dataset_notebook.py --dataset mnist --mode smoke --offline
```

Missing offline data or weights produce an error; the code does not silently substitute synthetic data or a randomly initialized backbone.

## Evaluation protocol and metrics

Every full run uses seed 42 to reserve **25 validation images/class**, **100 enlarged calibration images/class**, and all remaining official training examples as training/reference support. The standard calibration track uses the first 25 examples/class from the enlarged set. Splits are disjoint and preserve natural training imbalance. Validation selects model weights. All SDM CDFs, HR thresholds, class cohorts and positive scalar temperatures are fitted/frozen before the official test split is loaded.

Five methods share the same raw-logit predictions: raw softmax, temperature softmax fitted on each calibration track, and identity-adaptor SDM fitted on each track. Thus full-coverage classification precision/recall and accuracy are identical; confidence ordering, calibrated probabilities, and selective results may differ.

| Metric family | Output |
|---|---|
| Classification | Top-1 accuracy, macro/micro precision and recall, class support and per-class precision/recall |
| Temperature | Positive temperature fitted on calibration NLL only, before/after calibration NLL, bounded-search flag |
| Calibration | NLL, multiclass Brier score, 15-bin ECE |
| Error detection | Error AUROC and AUPRC using low confidence |
| Selective prediction | Risk–coverage curves, AURC, retrospective coverage at 95% accuracy, accuracy at 10/25/50/75/100% coverage |
| High-Reliability | Region found, admitted count, coverage, selective accuracy, macro/micro precision and recall |
| Cohorts | Calibration-frozen below/at-or-above-target cohorts using true-class and predicted-class assignments |
| Class confusions | Labelled normalized confusion matrix, error zoom, frequent directed mistakes |
| Recall–coverage | Macro recall on y-axis versus admitted coverage on x-axis, for all five methods; tied confidence groups enter together |

Macro metrics average over classes observed in the true evaluation labels. Undefined class precision is set to zero; recall is undefined for unobserved true classes. Abstentions remain in recall's denominator. With no admissions, micro precision is undefined and recall is zero. Temperature search is limited to `[0.01, 100]` and bound fits are flagged. The 95% HR target and retrospective coverage curves are empirical diagnostics, not guarantees under distribution shift.

Equal-confidence groups enter together in every ranking and selective metric. Error AUPRC is average precision at complete score groups. AURC uses the right-endpoint step integral across attainable coverage intervals, including the interval from zero to the first group. Fixed-coverage accuracy uses the largest attainable threshold coverage within the requested budget; the table also exports `actual_coverage_at_*pct_budget`. When no group fits, accuracy is undefined and actual coverage is zero. These definitions correct the previous row-order-dependent metrics; regenerate older tables before comparing them with these runs.

SDM matches notebook 04's frozen identity-adaptor variant and pinned upstream implementation. Exact nearest-neighbor search is batched, retaining only each query's similarity count and closest exemplar rather than every neighbor array. This limits memory use as training support grows.

## Saved results and provenance

Data defaults to `~/.cache/sdm-vision-reliability/datasets/<dataset>/`. Results default to:

```text
~/.cache/sdm-vision-reliability/runs/<dataset>/convnext_tiny_imagenet1k_v1/
  seed_42/calibration_25_vs_100/<benchmark-or-smoke>/
```

`SDM_DATA_DIR` overrides the cache root; `SDM_RUN_DIR` overrides the **output root**, with dataset/model/seed/experiment/mode always appended to prevent checkpoint collisions. The runner also accepts `--data-root` and `--output-root`.

Each run saves its split indices, compatible best checkpoint, reference/calibration/test `.npz` outputs with original dataset indices, per-image CSV, comparison/cohort/HR/per-class/selective metric CSVs, fitted-temperature CSV, recall-coverage CSV, PNGs, and metadata with class names, actual split sizes, configuration, upstream commit and package versions. Smoke metadata is labelled `pipeline_check`. Keep all data, weights, outputs and executed notebook copies outside Git.

## Maintaining parity

The generated notebooks keep notebook 04's metric and SDM code visible for study. Dataset loading/splitting lives in `notebooks/benchmark_datasets.py`. The generator copies source cells only and never edits notebook 04, including its local executed outputs:

```bash
python scripts/build_dataset_notebooks.py
python -m unittest discover -s tests
```

Regeneration requires notebook 04's precision/recall/temperature updates from PR #27. Tests verify exact metric-function parity, all six schemas, label adapters, preprocessing, disjoint splits with class imbalance, streamed neighbor signals, and six complete synthetic smoke pipelines. Synthetic fixtures use a tiny backbone and do not establish benchmark accuracy. A real MNIST CUDA smoke run additionally checks the actual loader and ConvNeXt path; full dataset downloads and full training still require lead validation.
