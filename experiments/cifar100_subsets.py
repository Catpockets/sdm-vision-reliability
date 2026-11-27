"""Train and evaluate five prespecified ten-class draws from CIFAR-100 (issue #35)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "notebooks"), str(ROOT / "scripts")]

from shift_metrics import admission_mask, metric_functions, selective_statistics, threshold_for_target

SEEDS = (0, 1, 2, 3, 4)
SPLIT_SEED = 42
TRACKS = ("standard", "enlarged")
METHODS = ("softmax", "temperature", "sdm_identity")
TARGET = 0.95
OUTPUT = ROOT / "outputs" / "cifar100_subsets"


def experiment_plan(class_names, smoke):
    from sdm_vision.data import class_order

    if len(class_names) != 100 or len(set(class_names)) != 100:
        raise ValueError("Expected the 100 distinct CIFAR-100 fine labels")
    draws = []
    for seed in SEEDS:
        class_ids = sorted(class_order(seed, len(class_names))[:10].tolist())
        draws.append({"seed": seed, "class_ids": class_ids, "class_names": [class_names[i] for i in class_ids]})
    return {
        "protocol_version": 2,
        "mode": "smoke" if smoke else "benchmark",
        "dataset": "CIFAR-100 fine labels",
        "split_seed": SPLIT_SEED,
        "training": {"model": "ResNet-18 CIFAR stem", "epochs": 1 if smoke else 40, "batch_size": 128,
                     "selection": "fixed final epoch", "optimizer": "SGD Nesterov, one-cycle",
                     "max_lr": 0.1, "weight_decay": 0.0005},
        "per_class": {"train": 16 if smoke else 375, "standard": 25, "enlarged": 100,
                      "test": 4 if smoke else 100},
        "global_baseline_target": TARGET,
        "global_baseline_min_admitted": 100,
        "draws": draws,
    }


def atomic_json(path, content):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def split_training(labels, class_ids, train_per_class):
    """Keep the same per-class populations across draws, including overlapping classes."""
    from sdm_vision.data import fixed_split

    labels = np.asarray(labels)
    full_split = fixed_split(labels, train_per_class=train_per_class, cal_per_class=100,
                             val_per_class=25, seed=SPLIT_SEED)
    train = full_split["train"]
    calibration = full_split["calibration"].reshape(len(np.unique(labels)), 100)
    standard = calibration[:, :25].flatten()
    enlarged = calibration.flatten()
    return {name: ids[np.isin(labels[ids], class_ids)]
            for name, ids in (("train", train), ("standard", standard), ("enlarged", enlarged))}


def remap_labels(labels, class_ids):
    mapping = np.full(100, -1, dtype=int)
    mapping[class_ids] = np.arange(len(class_ids))
    labels = np.asarray(labels)
    if np.any((labels < 0) | (labels >= len(mapping))) or np.any(mapping[labels] < 0):
        raise ValueError("Cannot remap a label outside this draw")
    return mapping[labels]


def pixels(dataset, indices):
    images = dataset.data[indices].transpose(0, 3, 1, 2)
    return np.pad(images, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="reflect")


def sdm_values(namespace, logits, distances, similarity, cdfs):
    distance_quantile = namespace["conservative_distance_quantile"](distances, cdfs)
    probabilities = namespace["sdm_activation"](logits, similarity, distance_quantile)
    predictions = logits.argmax(axis=1)
    confidence = probabilities[np.arange(len(predictions)), predictions]
    return probabilities, namespace["rescaled_similarity"](similarity, confidence)


def fit_policies(namespace, reference, calibration, standard_mask, device):
    import torch
    from evaluate_matched_shift import streamed_signals

    namespace.update(DEVICE=device, reference=reference)
    tensor = torch.as_tensor(reference["embeddings"], dtype=torch.float32)
    namespace.update(embedding_mean=tensor.mean(), embedding_std=tensor.std().clamp_min(torch.finfo(tensor.dtype).eps))
    namespace["reference_embeddings"] = namespace["normalize_embeddings"](reference["embeddings"])
    namespace["neighbors"] = min(len(tensor), int(np.bincount(reference["targets"]).max()) + 1)
    logits = calibration["logits"]
    distance, similarity = streamed_signals(namespace, namespace["normalize_embeddings"](calibration["embeddings"]),
                                            logits.argmax(axis=1))
    tracks = {}
    for name, mask in zip(TRACKS, (standard_mask, np.ones(len(logits), dtype=bool))):
        track_logits, labels = logits[mask], calibration["targets"][mask]
        temperature = namespace["fit_temperature"](track_logits, labels)["temperature"]
        correct = track_logits.argmax(axis=1) == labels
        thresholds = {}
        for method, scale in (("softmax", 1.0), ("temperature", temperature)):
            confidence = namespace["temperature_probabilities"](track_logits, scale).max(axis=1)
            thresholds[method] = threshold_for_target(confidence, correct, target=TARGET, min_admitted=100)
        cdfs = namespace["build_class_distance_cdfs"](distance[mask], similarity[mask], labels,
                                                    number_of_classes=track_logits.shape[1])
        probabilities, q_prime = sdm_values(namespace, track_logits, distance[mask], similarity[mask], cdfs)
        region = namespace["fit_high_reliability_region"](probabilities, q_prime, labels, alpha=TARGET)
        tracks[name] = {"temperature": temperature, "thresholds": thresholds, "cdfs": cdfs, "region": region}
    return tracks


def seen_test_pixels(training_and_calibration_images, test_images):
    """Find exact pixel matches regardless of dataset ID or class label."""
    seen = {image.tobytes() for image in training_and_calibration_images}
    return np.array([image.tobytes() in seen for image in test_images], dtype=bool)


def summarize_admissions(predictions, labels, admitted, seen_pixels):
    eligible = ~seen_pixels
    hits = (predictions == labels) & admitted & eligible
    statistics = selective_statistics(predictions[eligible], labels[eligible], admitted[eligible])
    return {
        "official_examples": len(labels),
        "excluded_seen_pixels": int(seen_pixels.sum()),
        "correct_admitted": int(hits.sum()),
        **{key: value if np.isfinite(value) else None for key, value in statistics.items()},
    }


def score_policies(namespace, tracks, evaluation, seen_pixels):
    from evaluate_matched_shift import streamed_signals

    logits, labels = evaluation["logits"], evaluation["targets"]
    predictions = logits.argmax(axis=1)
    distance, similarity = streamed_signals(namespace, namespace["normalize_embeddings"](evaluation["embeddings"]),
                                            predictions)
    rows, arrays = [], {}
    for track, policy in tracks.items():
        sdm_probabilities, q_prime = sdm_values(namespace, logits, distance, similarity, policy["cdfs"])
        sdm_admitted = namespace["apply_high_reliability_region"](sdm_probabilities, q_prime, predictions, policy["region"])
        arrays[f"{track}_q_prime"] = q_prime
        for method, scale in (("softmax", 1.0), ("temperature", policy["temperature"]), ("sdm_identity", None)):
            probabilities = sdm_probabilities if scale is None else namespace["temperature_probabilities"](logits, scale)
            confidence = probabilities[np.arange(len(predictions)), predictions]
            admitted = sdm_admitted if scale is None else admission_mask(confidence, policy["thresholds"][method])
            rows.append({"track": track, "method": method,
                         **summarize_admissions(predictions, labels, admitted, seen_pixels)})
            arrays[f"{track}_{method}_confidence"] = confidence
            arrays[f"{track}_{method}_admitted"] = admitted
    return rows, arrays


def run_draw(draw_number, *, data_dir, output=OUTPUT, smoke=True, download=False):
    import torch
    from torchvision.datasets import CIFAR100
    from sdm_vision import train

    if not 1 <= draw_number <= len(SEEDS):
        raise ValueError(f"Draw must be between 1 and {len(SEEDS)}")
    training = CIFAR100(data_dir, train=True, download=download)
    plan = experiment_plan(training.classes, smoke)
    draw = plan["draws"][draw_number - 1]
    root = Path(output) / plan["mode"]
    folder = root / f"draw-{draw_number}"
    status_path = folder / "status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous["status"] == "complete":
            raise FileExistsError(f"Completed draw preserved at {folder}. Choose a fresh output root.")
    folder.mkdir(parents=True, exist_ok=True)
    # Every draw writes the same complete plan before any model is trained.
    plan_path = root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError(f"Existing experiment plan differs at {plan_path}")
    atomic_json(plan_path, plan)
    # A per-draw lock protects exports if two notebooks start the same draw.
    lock_path = folder / ".running"
    with lock_path.open("x"):
        pass
    try:
        if status_path.exists() and json.loads(status_path.read_text())["status"] == "complete":
            raise FileExistsError(f"Completed draw preserved at {folder}")
        atomic_json(status_path, {"status": "running"})
        try:
            rows = evaluate_draw(training, data_dir, folder, draw, plan, train, torch)
            atomic_json(status_path, {"status": "complete", "rows": rows})
        except BaseException as error:
            atomic_json(status_path, {"status": "failed", "reason": f"{type(error).__name__}: {error}"})
            raise
    finally:
        lock_path.unlink()
    return rows


def evaluate_draw(training, data_dir, folder, draw, plan, train, torch):
    from torchvision.datasets import CIFAR100

    class_ids = draw["class_ids"]
    labels = np.asarray(training.targets)
    split = split_training(labels, class_ids, plan["per_class"]["train"])
    np.savez_compressed(folder / "training_splits.npz", **split, fine_labels=labels)
    device = train.pick_device()
    train_labels = remap_labels(labels[split["train"]], class_ids)
    settings = plan["training"]
    model, normalization = train.train_resnet(
        pixels(training, split["train"]), train_labels, len(class_ids), device,
        epochs=settings["epochs"], seed=draw["seed"], batch_size=settings["batch_size"],
        max_lr=settings["max_lr"], weight_decay=settings["weight_decay"],
    )
    torch.save({"model": model.state_dict(), "mean": normalization.mean.cpu(), "std": normalization.std.cpu(),
                "draw": draw, "settings": settings, "torch_version": str(torch.__version__)}, folder / "checkpoint.pt")

    def predict(dataset, indices):
        logits, embeddings = train.predict(model, normalization, pixels(dataset, indices))
        return {"logits": logits.numpy(), "embeddings": embeddings,
                "predictions": logits.argmax(1).numpy(),
                "targets": remap_labels(np.asarray(dataset.targets)[indices], class_ids)}

    reference = predict(training, split["train"])
    np.savez_compressed(folder / "reference.npz", **reference, dataset_indices=split["train"])
    calibration = predict(training, split["enlarged"])
    standard_mask = np.isin(split["enlarged"], split["standard"])
    namespace = metric_functions()
    tracks = fit_policies(namespace, reference, calibration, standard_mask, device)
    np.savez_compressed(folder / "calibration.npz", **calibration, dataset_indices=split["enlarged"],
                        standard_mask=standard_mask)
    atomic_json(folder / "frozen_policies.json", {
        name: {"temperature": policy["temperature"], "thresholds": policy["thresholds"],
               "sdm_region": {"found": policy["region"]["found"],
                              "q_prime_min": policy["region"]["q_prime_min"] if policy["region"]["found"] else None,
                              "thresholds": policy["region"]["thresholds"].tolist() if policy["region"]["found"] else None}}
        for name, policy in tracks.items()
    })

    # The prediction policies are frozen before opening the held-out test dataset.
    test = CIFAR100(data_dir, train=False, download=False)
    if test.classes != training.classes:
        raise ValueError("Train and test class maps disagree")
    test_labels = np.asarray(test.targets)
    test_ids = np.sort(np.concatenate([np.flatnonzero(test_labels == idx)[:plan["per_class"]["test"]]
                                       for idx in class_ids]))
    if len(test_ids) != len(class_ids) * plan["per_class"]["test"]:
        raise ValueError("Incomplete official test population")
    evaluation = predict(test, test_ids)
    seen_pixels = seen_test_pixels(
        training.data[np.concatenate((split["train"], split["enlarged"]))], test.data[test_ids]
    )
    rows, scores = score_policies(namespace, tracks, evaluation, seen_pixels)
    np.savez_compressed(folder / "test_predictions.npz", dataset_indices=test_ids,
                        fine_labels=test_labels[test_ids], logits=evaluation["logits"],
                        labels=evaluation["targets"], predictions=evaluation["predictions"],
                        seen_train_or_calibration_pixels=seen_pixels, **scores)
    return rows


def summary_rows(output=OUTPUT, smoke=True):
    root = Path(output) / ("smoke" if smoke else "benchmark")
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    rows = []
    for draw, seed in enumerate(SEEDS, start=1):
        path = root / f"draw-{draw}" / "status.json"
        saved = json.loads(path.read_text()) if path.exists() else {"status": "missing"}
        classes = ", ".join(plan["draws"][draw - 1]["class_names"]) if plan else None
        base = {"draw": draw, "seed": seed, "status": saved["status"], "classes": classes}
        if saved["status"] == "running" and not path.with_name(".running").exists():
            base["status"] = "interrupted"
        if saved["status"] == "complete":
            rows.extend(base | row for row in saved["rows"])
        else:
            rows.append(base | {"reason": saved.get("reason")})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw", type=int, choices=range(1, len(SEEDS) + 1), required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "cifar100")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--benchmark", action="store_true", help="Run all 40 epochs for the chosen draw.")
    parser.add_argument("--download", action="store_true", help="Allow torchvision to prepare CIFAR-100.")
    args = parser.parse_args()
    print(json.dumps(run_draw(args.draw, data_dir=args.data_dir, output=args.out,
                              smoke=not args.benchmark, download=args.download), indent=2))


if __name__ == "__main__":
    main()
