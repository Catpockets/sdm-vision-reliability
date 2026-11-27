"""Measure warm, synchronized inference from decoded CIFAR pixels to CPU predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns

import numpy as np
from PIL import Image
import torch
from torchvision.datasets import CIFAR100
from torchvision.models import convnext_tiny

from evaluate_subset_cascade import TRACKS, indexes_for_ids, verify_file
from evaluate_matched_shift import convnext_transform, read_arrays, streamed_signals
from shift_metrics import admission_mask, metric_functions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cifar100_subsets import fit_policies, sdm_values
from sdm_vision.model import ImageNetResNet

POLICIES = ("convnext", "temperature", "sdm")


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_fallback(path, expected_hash, device):
    verify_file(path, expected_hash)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = convnext_tiny(weights=None, num_classes=100)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval().to(device)


def prepare_routes(folder, draw, budget, track, receipt, fallback, device):
    path = folder / "checkpoint.pt"
    verify_file(path, receipt["sha256"][f"draw-{draw}/checkpoint.pt"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    class_ids = torch.as_tensor(checkpoint["draw"]["class_ids"], dtype=torch.long, device=device)
    model = ImageNetResNet(num_classes=len(class_ids), stem="cifar")
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    mean, std = (checkpoint[key].to(device) for key in ("mean", "std"))
    transform = convnext_transform()

    for name in ("reference.npz", "calibration.npz", "frozen_policies.json"):
        verify_file(folder / name, budget["artifacts_sha256"][name])
    calibration = read_arrays(folder / "calibration.npz")
    functions = metric_functions()
    policies = fit_policies(functions, read_arrays(folder / "reference.npz"), calibration,
                           calibration["standard_mask"], device)
    frozen = json.loads((folder / "frozen_policies.json").read_text())[track]
    region = frozen["sdm_region"]
    policy = policies[track]
    if policy["temperature"] != frozen["temperature"] or policy["region"]["found"] != region["found"]:
        raise ValueError(f"Calibration replay mismatch in draw {draw}")
    if region["found"]:
        np.testing.assert_allclose(policy["region"]["thresholds"], region["thresholds"], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(policy["region"]["q_prime_min"], region["q_prime_min"], rtol=1e-5, atol=1e-7)
        region["thresholds"] = np.asarray(region["thresholds"], dtype=np.float32)
    functions["reference_embeddings"] = torch.as_tensor(functions["reference_embeddings"], device=device)
    cutoff = budget["tracks"][track]["methods"]["temperature"]["threshold"]

    def convnext(pixels):
        batch = torch.stack([transform(Image.fromarray(image)) for image in pixels]).to(device)
        return fallback(batch).index_select(1, class_ids).argmax(1)

    def route(pixels, method):
        if method == "convnext" or (method == "temperature" and cutoff is None) or (method == "sdm" and not region["found"]):
            return class_ids[convnext(pixels)].cpu().numpy(), np.ones(len(pixels), dtype=bool)
        inputs = torch.from_numpy(pixels).permute(0, 3, 1, 2).to(device)
        embeddings = model.embed((inputs.float() / 255 - mean) / std)
        logits = model.classifier(embeddings).cpu().numpy()
        predictions = logits.argmax(1)
        if method == "temperature":
            confidence = functions["temperature_probabilities"](logits, frozen["temperature"]).max(1)
            admitted = admission_mask(confidence, cutoff)
        else:
            normalized = functions["normalize_embeddings"](embeddings.cpu().numpy())
            distances, q = streamed_signals(functions, normalized, predictions)
            probabilities, q_prime = sdm_values(functions, logits, distances, q, policy["cdfs"])
            admitted = functions["apply_high_reliability_region"](probabilities, q_prime, predictions, region)
        delegated = ~admitted
        if delegated.any():
            predictions[delegated] = convnext(pixels[delegated]).cpu().numpy()
        return checkpoint_classes[predictions], delegated

    checkpoint_classes = np.asarray(checkpoint["draw"]["class_ids"])
    return route


def summarize_requests(rows):
    durations = np.array([row["ns"] for row in rows], dtype=float)
    images = sum(row["images"] for row in rows)
    fallback = sum(row["fallback"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    p50, p95 = np.percentile(durations, [50, 95]) / 1e6
    return {"requests": len(rows), "images": images, "fallback": fallback, "correct": correct,
            "accuracy": correct / images, "p50_ms": float(p50), "p95_ms": float(p95),
            "mean_ms": float(durations.mean() / 1e6), "images_per_second": float(images * 1e9 / durations.sum()),
            "routing_disagreements": sum(row["routing_disagreements"] for row in rows),
            "prediction_disagreements": sum(row["prediction_disagreements"] for row in rows)}


@torch.inference_mode()
def benchmark(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    receipt = json.loads((args.subsets / "verified-results.json").read_text())
    routing_path = args.routing / "routing.json"
    routing_hash = verify_file(routing_path, args.routing_sha256)
    evidence = json.loads(routing_path.read_text())
    verify_file(args.subsets / "plan.json", evidence["primary_plan_sha256"])
    fallback = load_fallback(args.convnext_checkpoint, evidence["fallback_checkpoint_sha256"], device)
    dataset = CIFAR100(args.data, train=False, download=False)
    targets = np.asarray(dataset.targets)
    records, samples = [], {}

    for draw in range(1, 6):
        budget = evidence["calibration_budgets"][str(draw)]
        route = prepare_routes(args.subsets / f"draw-{draw}", draw, budget, args.track, receipt, fallback, device)
        with np.load(args.routing / f"draw-{draw}.npz") as replay:
            if args.samples > len(replay["dataset_indices"]):
                raise ValueError("Sample count exceeds the eligible test cohort")
            order = np.random.default_rng(args.seed + draw).permutation(len(replay["dataset_indices"]))[:args.samples]
            ids = replay["dataset_indices"][order]
            labels = replay["fine_labels"][order]
            np.testing.assert_array_equal(targets[ids], labels)
            primary = replay["primary_fine_prediction"][order]
            alternate = replay["fallback_fine_prediction"][order]
            primary_masks = {"sdm": replay[f"{args.track}_sdm_primary"][order],
                             "temperature": replay[f"{args.track}_temperature_frozen_budget_primary"][order]}
        expected_masks = {"convnext": np.ones(len(ids), dtype=bool),
                          **{method: ~mask for method, mask in primary_masks.items()}}
        expected_labels = {method: np.where(mask, alternate, primary) for method, mask in expected_masks.items()}
        samples[draw] = ids.tolist()
        pixels = dataset.data[ids]

        for size in args.batch_sizes:
            batches = [slice(start, start + size) for start in range(0, len(ids), size)]
            for i in range(args.warmup):
                selection = batches[i % len(batches)]
                for method in POLICIES:
                    route(pixels[selection], method)
            synchronize(device)
            for repeat in range(args.repeats):
                for request, selection in enumerate(batches):
                    rotation = (draw + repeat + request) % len(POLICIES)
                    for method in POLICIES[rotation:] + POLICIES[:rotation]:
                        synchronize(device)
                        start = perf_counter_ns()
                        predictions, delegated = route(pixels[selection], method)
                        synchronize(device)
                        elapsed = perf_counter_ns() - start
                        records.append({"draw": draw, "batch_size": size, "repeat": repeat, "request": request,
                                        "policy": method, "ns": elapsed, "images": len(predictions),
                                        "fallback": int(delegated.sum()), "correct": int((predictions == labels[selection]).sum()),
                                        "routing_disagreements": int((delegated != expected_masks[method][selection]).sum()),
                                        "prediction_disagreements": int((predictions != expected_labels[method][selection]).sum())})
            print(f"Draw {draw}, batch {size} complete", flush=True)

    summaries = []
    for size in args.batch_sizes:
        for method in POLICIES:
            for draw in (0, 1, 2, 3, 4, 5):
                group = [row for row in records if row["batch_size"] == size and row["policy"] == method
                         and (draw == 0 or row["draw"] == draw)]
                summaries.append({"draw": draw, "batch_size": size, "policy": method, **summarize_requests(group)})
    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    report = {"device": str(device), "platform": platform.platform(), "torch": str(torch.__version__),
              "host_cpu": cpu, "threads": args.threads, "sample_seed": args.seed,
              "calibration_track": args.track, "repeats": args.repeats, "warmup": args.warmup,
              "scope": "Decoded in-memory uint8 pixels to CPU predictions. Model setup and dataset I/O are outside timing.",
              "input_routing_sha256": routing_hash,
              "sample_ids": samples, "summary": summaries}
    return report, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("subsets", "routing", "convnext-checkpoint", "data", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--routing-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--track", choices=TRACKS, default="enlarged")
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (min(args.samples, args.repeats, args.warmup, args.threads) < 1
            or any(size < 1 or args.samples % size for size in args.batch_sizes)
            or len(set(args.batch_sizes)) != len(args.batch_sizes)):
        parser.error("Use positive counts and distinct batch sizes that divide the sample count")
    args.output.mkdir(parents=True, exist_ok=False)
    report, records = benchmark(args)
    report["source_revision"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    (args.output / "latency.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output / "requests.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for row in report["summary"]:
        if row["draw"] == 0:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
