"""Generate notebooks 05–10 from the reviewed notebook 04 protocol.

Run from any directory: python scripts/build_dataset_notebooks.py
Only source cells are copied. Notebook 04 and its local outputs are never changed.
"""
import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'notebooks/04_cifar100_sdm_calibration_comparison.ipynb'
CATALOG = [
    ('05', 'cifar10', 'CIFAR-10', 10, 'CIFAR10', 'Natural-image classification; random crops and horizontal flips are used only for training.'),
    ('06', 'mnist', 'MNIST', 10, 'MNIST', 'Handwritten digit classification; grayscale is replicated into three channels. No digit flips are used.'),
    ('07', 'fashion_mnist', 'Fashion-MNIST', 10, 'FashionMNIST', 'Clothing classification; grayscale is replicated into three channels and mild affine augmentation is used for training.'),
    ('08', 'kmnist', 'KMNIST', 10, 'KMNIST', 'Kuzushiji character classification; retain the dataset\'s ten class names and use no character flips.'),
    ('09', 'emnist_balanced', 'EMNIST Balanced', 47, 'EMNIST', 'Use the **balanced** split with 47 zero-based classes, not the separate letters/digits variants. Apply the same upright transpose to training and evaluation images. Inspect the gallery before interpreting results.'),
    ('10', 'svhn', 'SVHN', 10, 'SVHN', 'Train on the official **train** split and evaluate on **test**. The optional extra split is unused. Torchvision maps original label 10 to digit 0. This is an in-distribution SVHN classifier, separate from using SVHN as OOD for a CIFAR classifier. No digit flips are used.'),
]


def build(template, number, key, title, classes, loader, note):
    notebook = copy.deepcopy(template)
    cells = {cell['id']: cell for cell in notebook['cells']}
    def put(id, text): cells[id]['source'] = text.splitlines(keepends=True)
    def replace(id, old, new):
        text = ''.join(cells[id]['source'])
        if old not in text: raise ValueError(f'Template changed: {id}: {old}')
        put(id, text.replace(old, new))

    # Class-count literals are generalized only in code; track names retain the
    # nominal full-run 25/class and 100/class experiment labels.
    for cell in notebook['cells']:
        text = ''.join(cell['source'])
        if cell['cell_type'] == 'code':
            text = re.sub(r'\brange\(100\)', 'range(NUM_CLASSES)', text).replace('minlength=100', 'minlength=NUM_CLASSES')
            text = text.replace('number_of_classes=100', 'number_of_classes=NUM_CLASSES')
            text = text.replace('np.full(100,', 'np.full(NUM_CLASSES,')
            text = text.replace('out=np.full(100,', 'out=np.full(NUM_CLASSES,')
            cell.update(execution_count=None, outputs=[], metadata={})
        else:
            text = text.replace('CIFAR-100', title).replace('100-class', f'{classes}-class')
        cell['source'] = text.splitlines(keepends=True)

    put('c38adac3', f'''# {number} — {title}: SDM and temperature calibration comparison

{note}

This notebook reproduces **notebook 04's full reliability protocol** on {title}: one seed-42 ImageNet-pretrained ConvNeXt-Tiny model, two nested calibration sets, raw softmax, two temperature-scaled softmax baselines, and two frozen identity-adaptor SDM tracks. The classifier has **{classes} output classes**.

| Track | Full-run images/class | Total calibration images |
|---|---:|---:|
| `standard_25` | 25 | {25 * classes:,} |
| `enlarged_100` | 100 | {100 * classes:,} |

Each class also reserves 25 validation images. All remaining official training examples provide training/reference support; the official test split is evaluated after every fit is frozen. Unlike CIFAR-100, some datasets have unequal class sizes: report support and macro/micro metrics together.

Metrics match notebook 04: precision/recall (macro, micro, per-class, HR), fitted temperatures, NLL, Brier, ECE, error AUROC/AUPRC, risk–coverage/AURC, coverage at 95% selective accuracy, fixed-coverage accuracy, HR results, cohorts, confusion plots, and **macro recall versus coverage**. CSVs, PNGs, split manifests, logits, and embeddings are saved outside Git.

This is a new dataset experiment, not a claimed accuracy result. Keep rejected/weak runs; do not select checkpoints or datasets by final-test performance. The 95% target is an observed selective metric, not a distribution-shift guarantee.

Dataset API: [{loader}](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.{loader}.html).
''')
    put('fecc0724', '''## Setup and experimental rules

Use Python 3.12 and a matching PyTorch/torchvision installation. From the repository root, activate your training environment, then run:

```bash
python -m pip install -r requirements-training.txt
python -m jupyterlab
```

For an NVIDIA installation, choose the CUDA wheel pair appropriate to your machine using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). CPU and Apple MPS are supported fallbacks, but full ConvNeXt runs are expensive. SVHN needs SciPy for its `.mat` files. Initial dataset/model downloads may be large; EMNIST downloads the archive containing all splits. `SDM_DOWNLOAD=0` uses prepared local data and weights only.

Choose **Restart Kernel and Run All Cells**. For a small pipeline check, start Jupyter with `SDM_RUN_MODE=smoke`; it uses 2 training, 1 validation, and nested 1/2 calibration examples per class, plus 2 test examples per class. The track names retain the nominal full-run 25/100 labels. Smoke results are incomplete and must not be interpreted. Smoke mode does not avoid the initial full dataset download.

1. Train with cross-entropy; select checkpoints using validation accuracy only.
2. Fit both SDM tracks, cohort maps, and softmax temperatures on calibration data only.
3. Freeze all fits before loading the official test split.
4. Evaluate all methods on identical test images; never tune from test outcomes.

`SDM_DATA_DIR` selects the shared download/cache root. `SDM_RUN_DIR` selects an output **root**; this notebook always appends dataset/model/seed/experiment/mode directories to prevent cross-dataset checkpoint collisions. Changing training settings requires a new output root or explicit `SDM_FORCE_RETRAIN=1`.
''')
    text = ''.join(cells['7c87267e']['source'])
    text = text.replace('import time\n', 'import time\nimport sys\n')
    text = text.replace('RUN_MODE =', f'''REPO_ROOT = next(
    path for path in (Path.cwd(), *Path.cwd().parents)
    if (path / "notebooks" / "benchmark_datasets.py").is_file()
)
sys.path.insert(0, str(REPO_ROOT / "notebooks"))
from benchmark_datasets import DATASETS, dataset_transforms, load_dataset, stratified_splits, smoke_test_indices

DATASET_KEY = "{key}"
SPEC = DATASETS[DATASET_KEY]
NUM_CLASSES = SPEC.classes
DOWNLOAD = os.environ.get("SDM_DOWNLOAD", "1") == "1"

RUN_MODE =''', 1)
    start = text.index('RUN_DIR = Path(')
    end = text.index('DATA_DIR.mkdir', start)
    text = text[:start] + '''RUN_ROOT = Path(os.environ.get("SDM_RUN_DIR", DATA_DIR / "runs")).expanduser()
RUN_DIR = RUN_ROOT / DATASET_KEY / MODEL_NAME / f"seed_{SEED}" / EXPERIMENT_NAME / RUN_MODE
''' + text[end:]
    text = text.replace('MAX_EXPORT_BATCHES = 2 if SMOKE else None', 'MAX_EXPORT_BATCHES = None  # Smoke uses explicit balanced subsets, not an ordered prefix.')
    text = text.replace('BATCH_SIZE = 8 if SMOKE else 128', 'BATCH_SIZE = 8 if SMOKE else int(os.environ.get("SDM_BATCH_SIZE", "128"))')
    text = text.replace('EPOCHS = 1 if SMOKE else 20', 'EPOCHS = 1 if SMOKE else int(os.environ.get("SDM_EPOCHS", "20"))\nif BATCH_SIZE < 1 or EPOCHS < 1:\n    raise ValueError("Batch size and epochs must be positive.")')
    text = text.replace('NUM_WORKERS = 0 if platform.system()', 'NUM_WORKERS = 0 if SMOKE or platform.system()')
    text = text.replace('    "run_mode": RUN_MODE,', '    "dataset": DATASET_KEY,\n    "number_of_classes": NUM_CLASSES,\n    "source_notebook": "04_cifar100_sdm_calibration_comparison.ipynb",\n    "protocol_issue": 28,\n    "run_mode": RUN_MODE,')
    text = text.replace('    "calibration_images_per_class": [25, 100],', '    "calibration_images_per_class": [1, 2] if SMOKE else [25, 100],\n    "validation_images_per_class": 1 if SMOKE else 25,\n    "preprocessing_version": "dataset_specific_v1",')
    put('7c87267e', text)
    put('a44d9346', '''## Build disjoint training, validation, and nested calibration splits

Use a seed-42 permutation **within every true class**. Full runs reserve 25 validation and 100 enlarged calibration images/class; the standard set takes the first 25 of those calibration images. All remaining examples are training/reference support. This preserves natural class imbalance rather than silently truncating to a CIFAR-100-sized support.

Only training images receive stochastic augmentation. Evaluation is deterministic, retains the complete input field, and replicates grayscale channels for the pretrained RGB backbone. Handwriting and digits are never flipped. The class-labelled gallery lets you verify label mapping and EMNIST orientation before training.
''')
    put('6797cc1c', '''train_transform, eval_transform = dataset_transforms(SPEC, IMAGE_SIZE)
train_aug_base = load_dataset(SPEC, DATA_DIR / "datasets", train=True, transform=train_transform, download=DOWNLOAD)
train_eval_base = load_dataset(SPEC, DATA_DIR / "datasets", train=True, transform=eval_transform, download=False)
CLASS_NAMES = list(train_aug_base.classes)
targets = train_aug_base.targets
splits = stratified_splits(targets, NUM_CLASSES, seed=SEED, smoke=SMOKE)
train_indices = splits["train"]
validation_indices = splits["validation"]
calibration_25_indices = splits["standard"]
calibration_100_indices = splits["enlarged"]

assert set(calibration_25_indices) <= set(calibration_100_indices)
assert not set(train_indices) & set(validation_indices)
assert not set(train_indices) & set(calibration_100_indices)
assert not set(validation_indices) & set(calibration_100_indices)
assert len(set(train_indices) | set(validation_indices) | set(calibration_100_indices)) == (
    len(train_indices) + len(validation_indices) + len(calibration_100_indices)
)
if not SMOKE:
    assert len(train_indices) + len(validation_indices) + len(calibration_100_indices) == len(targets)
np.savez_compressed(RUN_DIR / "split_indices.npz", **splits)

loader_options = {
    "batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS,
    "pin_memory": DEVICE.type == "cuda", "persistent_workers": False,  # Recreate seeded workers at epoch boundaries.
}
train_loader = DataLoader(
    Subset(train_aug_base, train_indices), shuffle=True,
    generator=torch.Generator().manual_seed(SEED), **loader_options
)
reference_loader = DataLoader(Subset(train_eval_base, train_indices), shuffle=False, **loader_options)
validation_loader = DataLoader(Subset(train_eval_base, validation_indices), shuffle=False, **loader_options)
calibration_loader = DataLoader(Subset(train_eval_base, calibration_100_indices), shuffle=False, **loader_options)

split_table = pd.DataFrame({
    "class_name": CLASS_NAMES,
    **{name: np.bincount(targets[indices], minlength=NUM_CLASSES) for name, indices in splits.items()},
})
display(split_table)
print({"training/reference": len(train_indices), "validation": len(validation_indices),
       "standard calibration": len(calibration_25_indices), "enlarged calibration": len(calibration_100_indices),
       "official test loaded": False})

fig, axes = plt.subplots(2, 5, figsize=(12, 5), layout="constrained")
for class_id, axis in zip(np.linspace(0, NUM_CLASSES - 1, 10, dtype=int), axes.flat):
    image_index = np.flatnonzero(targets == class_id)[0]
    # Show the deterministic orientation/channel conversion used for evaluation.
    image, label = train_eval_base[image_index]
    image = image.permute(1, 2, 0).numpy() * np.asarray([.229, .224, .225]) + np.asarray([.485, .456, .406])
    axis.imshow(np.clip(image, 0, 1))
    axis.set_title(CLASS_NAMES[label])
    axis.axis("off")
fig.suptitle(f"{SPEC.title}: class-labelled evaluation inputs (ten sampled classes)")
fig.savefig(RUN_DIR / "dataset_gallery.png", dpi=160)
plt.show()
''')
    replace('1e00ebcf', 'convnext_tiny(weights=weights)', 'convnext_tiny(weights=weights if DOWNLOAD else None)')
    # Download-disabled runs must still start from the exact pretrained weights.
    replace('1e00ebcf', 'model.classifier[2] =', '''if not DOWNLOAD:
    cached_weights = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not cached_weights.is_file():
        raise FileNotFoundError(f"Pretrained weights unavailable offline: {cached_weights}")
    model.load_state_dict(torch.load(cached_weights, map_location="cpu", weights_only=True))
model.classifier[2] =''')
    replace('1e00ebcf', 'in_features, 100)', 'in_features, NUM_CLASSES)')
    replace('1e00ebcf', 'logits.topk(5, dim=1)', 'logits.topk(min(5, NUM_CLASSES), dim=1)')
    replace('463f20b7', '''# The enlarged loader is ordered class-by-class, 100 examples per class in a full run.
if SMOKE:
    standard_mask = np.arange(len(calibration_all["targets"])) < min(len(calibration_all["targets"]), BATCH_SIZE)
else:
    standard_mask = np.tile(np.arange(100) < 25, 100)''', '''# Match by original dataset indices, so membership survives unequal class sizes.
standard_mask = np.isin(calibration_100_indices, calibration_25_indices)''')
    replace('463f20b7', 'print({name:', '''np.savez_compressed(RUN_DIR / "reference_outputs.npz", **reference, dataset_indices=train_indices)
np.savez_compressed(RUN_DIR / "calibration_outputs.npz", **calibration_all, dataset_indices=calibration_100_indices)
print({name:''')
    replace('545aa824', 'One more neighbor than the 375/class support maximum', 'One more neighbor than the largest observed true-class support')
    neighbor_text = ''.join(cells['14d4ec9e']['source'])
    neighbor_start = neighbor_text.index('calibration_neighbor_distances, calibration_neighbor_indices =')
    neighbor_text = neighbor_text[:neighbor_start] + """print({"calibration_queries": len(calibration_embeddings), "neighbors_per_query": neighbors})


def similarity_distance_signals(query_embeddings, query_predictions, batch_size=192):
    # Keep only q and the nearest exemplar. Full top-k groups are temporary;
    # support can contain thousands of examples/class in these datasets.
    nearest_distances, nearest_indices, similarities = [], [], []
    for start in range(0, len(query_embeddings), batch_size):
        stop = start + batch_size
        distances, indices = exact_squared_l2_neighbors(
            reference_embeddings, query_embeddings[start:stop], neighbors=neighbors, batch_size=batch_size
        )
        similarities.append(consecutive_correct_match_count(indices, query_predictions[start:stop]))
        nearest_distances.append(distances[:, :1])
        nearest_indices.append(indices[:, :1])
    return np.concatenate(nearest_distances), np.concatenate(nearest_indices), np.concatenate(similarities)
"""
    put('14d4ec9e', neighbor_text)
    similarity_cell = next(cell for cell in notebook['cells'] if cell['cell_type'] == 'code' and ''.join(cell['source']).startswith('def consecutive_correct_match_count'))
    similarity_text = ''.join(similarity_cell['source'])
    start = similarity_text.index('calibration_q_all = consecutive_correct_match_count(')
    stop = similarity_text.index('print(pd.Series', start)
    similarity_text = similarity_text[:start] + """calibration_neighbor_distances, calibration_neighbor_indices, calibration_q_all = similarity_distance_signals(
    calibration_embeddings, calibration_all["predictions"]
)
""" + similarity_text[stop:]
    put(similarity_cell['id'], similarity_text)
    put('85ef06d2', '''# First access to the official test split: all model/calibration choices are frozen.
test_base = load_dataset(SPEC, DATA_DIR / "datasets", train=False, transform=eval_transform, download=DOWNLOAD)
test_indices = smoke_test_indices(test_base.targets, NUM_CLASSES) if SMOKE else np.arange(len(test_base))
test_loader = DataLoader(Subset(test_base, test_indices), shuffle=False, **loader_options)
test = collect_outputs(test_loader, max_batches=MAX_EXPORT_BATCHES)
np.savez_compressed(RUN_DIR / "test_outputs.npz", **test, dataset_indices=test_indices)
test_embeddings = normalize_embeddings(test["embeddings"])
test_neighbor_distances, test_neighbor_indices, test_q = similarity_distance_signals(
    test_embeddings, test["predictions"]
)
test_d_nearest = test_neighbor_distances[:, 0]
print({"official_test_examples": len(test["targets"]), "loaded_after_freeze": True, "smoke": SMOKE})
''')
    replace('confusion-matrix', 'Held-out CIFAR-100 confusion matrix', f'Held-out {title} confusion matrix')
    replace('recall-coverage-explanation', 'for the balanced full ' + title + ' test set it equals top-1 accuracy.', 'for balanced test classes it equals top-1 accuracy. For imbalanced test data, macro recall can differ from accuracy.')
    replace('ee85a9a4', '"test_index": np.arange(len(test["targets"]))', '"test_index": test_indices')
    replace('ee85a9a4', '"split_sizes": {"train_reference": 37_500, "validation": 2_500, "standard_calibration": 2_500, "enlarged_calibration": 10_000, "test": 10_000}', '''"split_sizes": {"train_reference": len(train_indices), "validation": len(validation_indices), "standard_calibration": len(calibration_25_indices), "enlarged_calibration": len(calibration_100_indices), "test": len(test_indices)}''')
    replace('ee85a9a4', '    "config": CONFIG,', '    "config": CONFIG,\n    "dataset_source": SPEC.source_url,\n    "class_names": CLASS_NAMES,\n    "status": "pipeline_check" if SMOKE else "complete_benchmark",\n    "split_manifest": str(RUN_DIR / "split_indices.npz"),')
    # Save the same plot families in addition to displaying them.
    replace('b54c5f75', 'plt.tight_layout()\nplt.show()', 'plt.tight_layout()\nfig.savefig(RUN_DIR / "risk_coverage_and_hr.png", dpi=160)\nplt.show()')
    for cell in notebook['cells']:
        # Strip execution/IDE metadata from the source template.
        cell['metadata'] = {}
    notebook['metadata'] = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
        'sdm_protocol': {'dataset': key, 'source_notebook': TEMPLATE.name, 'issue': 28},
    }
    return notebook


def main():
    template = json.loads(TEMPLATE.read_text())
    required = {'temperature-scaling-fit', 'precision-recall-tables', 'recall-coverage-plot'}
    if not required <= {cell['id'] for cell in template['cells']}:
        raise ValueError('Notebook 04 requires the metric updates from PR #27 first.')
    for spec in CATALOG:
        number, key, *_ = spec
        notebook = build(template, *spec)
        destination = ROOT / f'notebooks/{number}_{key}_sdm_calibration_comparison.ipynb'
        destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
        print(destination.name)


if __name__ == '__main__':
    main()
