"""Replay two-stage routing from verified CIFAR-100 outputs without training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "notebooks"), str(ROOT / "experiments")]

from shift_metrics import admission_mask, fingerprint, metric_functions

TRACKS = ("standard", "enlarged")


def indexes_for_ids(source_ids, wanted_ids):
    """Resolve dataset IDs even when an export was saved out of order."""
    source_ids, wanted_ids = np.asarray(source_ids), np.asarray(wanted_ids)
    if (source_ids.ndim != 1 or wanted_ids.ndim != 1
            or not np.issubdtype(source_ids.dtype, np.integer)
            or not np.issubdtype(wanted_ids.dtype, np.integer)
            or len(np.unique(source_ids)) != len(source_ids)
            or len(np.unique(wanted_ids)) != len(wanted_ids)):
        raise ValueError("Expected unique integer IDs in both exports")
    order = np.argsort(source_ids)
    keys = source_ids[order]
    positions = np.searchsorted(keys, wanted_ids)
    if np.any(positions == len(keys)) or not np.array_equal(keys[positions], wanted_ids):
        raise ValueError("Fallback export is missing requested IDs")
    return order[positions]


def threshold_at_budget(confidence, target_count):
    """Closest attainable count, keeping ties whole and preferring fewer first-stage uses on ties."""
    confidence = np.asarray(confidence)
    if confidence.ndim != 1 or not np.isfinite(confidence).all() or not 0 <= target_count <= len(confidence):
        raise ValueError("Expected finite scores and an attainable count interval")
    ordered = np.sort(confidence)[::-1]
    if not len(ordered):
        return None
    counts = np.r_[0, np.flatnonzero(ordered[:-1] != ordered[1:]) + 1, len(ordered)]
    count = counts[np.argmin(np.abs(counts - target_count))]
    return float(ordered[count - 1]) if count else None


def confidence_at_budget(confidence, target_count):
    return admission_mask(confidence, threshold_at_budget(confidence, target_count))


def summarize(primary, fallback, labels, use_primary):
    primary_correct, fallback_correct = primary == labels, fallback == labels
    accepted = np.where(use_primary, primary, fallback)
    gains = int((use_primary & primary_correct & ~fallback_correct).sum())
    losses = int((use_primary & ~primary_correct & fallback_correct).sum())
    return {
        "examples": len(labels),
        "primary_calls": int(use_primary.sum()),
        "fallback_calls": int((~use_primary).sum()),
        "fallback_share": float((~use_primary).mean()),
        "correct": int((accepted == labels).sum()),
        "accuracy": float((accepted == labels).mean()),
        "primary_correct_on_routed": int((use_primary & primary_correct).sum()),
        "gains_vs_fallback_only": gains,
        "losses_vs_fallback_only": losses,
    }


def verify_file(path, expected):
    actual = fingerprint(path)
    if actual != expected:
        raise ValueError(f"Artifact digest mismatch: {path.name}")
    return actual


def freeze_calibration_budgets(subsets, provenance, plan, training, device_name):
    """Recover calibration admissions and freeze confidence cutoffs without reading test rows."""
    import torch
    from cifar100_subsets import fit_policies, score_policies

    device = torch.device(device_name)
    training_targets = np.asarray(training.targets)
    budgets = {}
    for number, draw in enumerate(plan["draws"], 1):
        folder = subsets / f"draw-{number}"
        classes = np.asarray(draw["class_ids"])
        hashes = {}
        for name in ("training_splits.npz", "calibration.npz", "frozen_policies.json"):
            path = folder / name
            hashes[name] = verify_file(path, provenance["sha256"][str(path.relative_to(subsets))])
        hashes["reference.npz"] = fingerprint(folder / "reference.npz")
        original = json.loads((folder / "frozen_policies.json").read_text())

        with np.load(folder / "training_splits.npz") as split, np.load(folder / "reference.npz") as reference, \
             np.load(folder / "calibration.npz") as calibration:
            if (not np.array_equal(reference["dataset_indices"], split["train"])
                    or not np.array_equal(calibration["dataset_indices"], split["enlarged"])
                    or not np.array_equal(calibration["dataset_indices"][calibration["standard_mask"]], split["standard"])
                    or np.intersect1d(split["train"], split["enlarged"]).size):
                raise ValueError(f"Development split mismatch in draw {number}")
            reference = {key: reference[key] for key in reference.files}
            calibration = {key: calibration[key] for key in calibration.files}
        for arrays in (reference, calibration):
            if (not np.array_equal(classes[arrays["targets"]], training_targets[arrays["dataset_indices"]])
                    or not np.array_equal(arrays["predictions"], arrays["logits"].argmax(1))):
                raise ValueError(f"Development labels or predictions mismatch in draw {number}")

        namespace = metric_functions()
        replayed = fit_policies(namespace, reference, calibration, calibration["standard_mask"], device)
        for track, gate in replayed.items():
            expected = original[track]
            recorded_region = expected["sdm_region"]
            region = gate["region"]
            if (gate["temperature"] != expected["temperature"] or gate["thresholds"] != expected["thresholds"]
                    or region["found"] != recorded_region["found"]):
                raise ValueError(f"Calibration replay mismatch in draw {number}, {track}")
            if region["found"]:
                np.testing.assert_allclose(region["thresholds"], recorded_region["thresholds"], rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(region["q_prime_min"], recorded_region["q_prime_min"], rtol=1e-5, atol=1e-7)
                gate["region"] = {**recorded_region, "thresholds": np.asarray(recorded_region["thresholds"], dtype=np.float32)}
        _, signals = score_policies(namespace, replayed, calibration, np.zeros(len(calibration["targets"]), dtype=bool))

        tracks = {}
        for track, members in zip(TRACKS, (calibration["standard_mask"], np.ones(len(calibration["targets"]), dtype=bool))):
            target = int(signals[f"{track}_sdm_identity_admitted"][members].sum())
            methods = {}
            for name in ("softmax", "temperature"):
                confidence = signals[f"{track}_{name}_confidence"][members]
                threshold = threshold_at_budget(confidence, target)
                methods[name] = {"threshold": threshold, "primary_calls": int(admission_mask(confidence, threshold).sum())}
            tracks[track] = {"examples": int(members.sum()), "sdm_primary_calls": target, "methods": methods}
        budgets[number] = {"artifacts_sha256": hashes, "tracks": tracks}
    return budgets


def evaluate(subsets, campaign, fallback_evidence, data_root, device_name="cpu"):
    from torchvision.datasets import CIFAR100

    training = CIFAR100(data_root, train=True, download=False)
    provenance = json.loads((subsets / "verified-results.json").read_text())
    plan_digest = verify_file(subsets / "plan.json", provenance["sha256"]["plan.json"])
    plan = json.loads((subsets / "plan.json").read_text())
    if plan["mode"] != "benchmark" or len(plan["draws"]) != 5 or plan["draws"] != provenance["draws"]:
        raise ValueError("Expected five verified benchmark draws")
    budgets = freeze_calibration_budgets(subsets, provenance, plan, training, device_name)
    test = CIFAR100(data_root, train=False, download=False)
    test_targets = np.asarray(test.targets)
    expected = json.loads(fallback_evidence.read_text())
    evidence = next(group for group in expected["groups"] if group["dataset"] == "cifar100")
    if evidence["evaluation_split"] != "official_test" or evidence["classes"] != len(test.classes):
        raise ValueError("Expected official 100-class ConvNeXt test evidence")
    files = {Path(entry["relative_to_campaign"]).name: entry for entry in evidence["evidence"]}
    verified = {}
    for name in ("split_indices.npz", "test_outputs.npz"):
        entry = files[name]
        path = campaign / entry["relative_to_campaign"]
        verified[name] = verify_file(path, entry["sha256"])

    with np.load(campaign / files["split_indices.npz"]["relative_to_campaign"]) as split:
        used_ids = np.concatenate([split[key] for key in ("train", "validation", "enlarged")])
    if not np.array_equal(np.sort(used_ids), np.arange(len(training))):
        raise ValueError("Expected the complete, disjoint development split of the fallback")
    # All primary draws also draw their development populations from this official train set.
    seen = {image.tobytes() for image in training.data[used_ids]}
    eligible = np.array([image.tobytes() not in seen for image in test.data], dtype=bool)
    with np.load(campaign / files["test_outputs.npz"]["relative_to_campaign"]) as fallback:
        fallback_order = indexes_for_ids(fallback["dataset_indices"], np.arange(len(test)))
        fallback_logits = fallback["logits"][fallback_order]
        if (len(fallback["dataset_indices"]) != len(test) or fallback_logits.shape != (len(test), len(test.classes))
                or not np.isfinite(fallback_logits).all()
                or not np.array_equal(fallback["targets"][fallback_order], test_targets)
                or not np.array_equal(fallback["predictions"][fallback_order], fallback_logits.argmax(1))
                or int((fallback_logits.argmax(1) == test_targets).sum()) != evidence["correct"]):
            raise ValueError("Fallback predictions do not match official CIFAR-100 evidence")

    rows, per_image, draws = [], {}, []
    for number, draw in enumerate(plan["draws"], 1):
        classes = np.asarray(draw["class_ids"])
        if (classes.shape != (10,) or not np.issubdtype(classes.dtype, np.integer)
                or len(np.unique(classes)) != 10 or np.any((classes < 0) | (classes >= len(test.classes)))
                or [test.classes[i] for i in classes] != draw["class_names"]):
            raise ValueError(f"Unexpected class map in draw {number}")
        path = subsets / f"draw-{number}" / "test_predictions.npz"
        source_hash = verify_file(path, provenance["sha256"][str(path.relative_to(subsets))])
        with np.load(path) as first:
            original_ids = first["dataset_indices"]
            ids = np.flatnonzero(np.isin(test_targets, classes))
            first_order = indexes_for_ids(original_ids, ids)
            if len(original_ids) != len(ids):
                raise ValueError(f"Incomplete or extra primary predictions in draw {number}")
            primary = first["predictions"][first_order]
            labels = first["labels"][first_order]
            if (np.any((labels < 0) | (labels >= len(classes)))
                    or not np.array_equal(classes[labels], test_targets[ids])
                    or not np.array_equal(first["fine_labels"][first_order], test_targets[ids])
                    or not np.array_equal(primary, first["logits"][first_order].argmax(1))):
                raise ValueError(f"Primary predictions do not match official labels in draw {number}")

            keep = eligible[ids]
            if not keep.any():
                raise ValueError(f"No eligible test images in draw {number}")
            fallback = fallback_logits[np.ix_(ids[keep], classes)].argmax(1)
            labels, primary = labels[keep], primary[keep]
            masks = {
                ("all", "fallback_only"): np.zeros(len(labels), dtype=bool),
                ("all", "primary_only"): np.ones(len(labels), dtype=bool),
            }
            for track in TRACKS:
                sdm = first[f"{track}_sdm_identity_admitted"][first_order][keep]
                if sdm.dtype != bool:
                    raise ValueError("Expected saved boolean SDM admission masks")
                masks[track, "sdm"] = sdm
                for name in ("softmax", "temperature"):
                    scores = first[f"{track}_{name}_confidence"][first_order][keep]
                    masks[track, f"{name}_matched_budget"] = confidence_at_budget(scores, int(sdm.sum()))
                    frozen = budgets[number]["tracks"][track]["methods"][name]["threshold"]
                    masks[track, f"{name}_frozen_budget"] = admission_mask(scores, frozen)
        draws.append({"draw": number, "class_ids": classes.tolist(), "official_examples": len(ids),
                      "excluded_exact_pixel_ids": ids[~keep].tolist(), "primary_export_sha256": source_hash})
        image_rows = {"dataset_indices": ids[keep], "fine_labels": test_targets[ids[keep]],
                      "primary_fine_prediction": classes[primary], "fallback_fine_prediction": classes[fallback]}
        for (track, policy), mask in masks.items():
            rows.append({"draw": number, "calibration_track": track, "policy": policy,
                         **summarize(primary, fallback, labels, mask)})
            image_rows[f"{track}_{policy}_primary"] = mask
        per_image[number] = image_rows

    return {
        "scope": "Retrospective clean CIFAR-100 routing, with known ten-class membership per draw.",
        "matching": {
            "matched_budget": "Nearest attainable whole-score-group call count in the unlabeled test cohort.",
            "frozen_budget": "Nearest attainable count on calibration, with the resulting cutoff applied unchanged on test.",
        },
        "calibration_device": device_name,
        "calibration_budgets": budgets,
        "primary_revision": provenance.get("candidate", provenance.get("training_candidate")),
        "primary_plan_sha256": plan_digest,
        "fallback_checkpoint_sha256": evidence["checkpoint_sha256"],
        "fallback_artifact_sha256": verified,
        "draws": draws,
        "rows": rows,
    }, per_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", required=True, type=Path, help="Verified random-subset benchmark directory.")
    parser.add_argument("--campaign", required=True, type=Path, help="Saved experiment campaign directory.")
    parser.add_argument("--fallback-evidence", required=True, type=Path, help="Verified clean evidence JSON.")
    parser.add_argument("--data", required=True, type=Path, help="Prepared CIFAR-100 data directory.")
    parser.add_argument("--output", required=True, type=Path, help="New directory for the audit exports.")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"), help="Calibration replay device.")
    args = parser.parse_args()
    report, images = evaluate(args.subsets, args.campaign, args.fallback_evidence, args.data, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "routing.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output / "routing.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    for number, records in images.items():
        np.savez_compressed(args.output / f"draw-{number}.npz", **records)
    print(f"Wrote {len(report['rows'])} policy rows, five per-image exports and provenance to {args.output}")


if __name__ == "__main__":
    main()
