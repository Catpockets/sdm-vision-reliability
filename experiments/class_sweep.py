"""Class-count sweep: softmax vs SDM on CIFAR-10 and Imagenette (32 x 32) with K = 2..10 nested classes.

Each chain adds the classes in its own random order (sdm_vision.data.class_order), so the K-class task
contains the (K-1)-class task plus one new class. Every run trains a fresh ResNet-18, fits the temperature
and SDM on calibration, and evaluates on validation: the K classes' images (in-set) and, for K < 10,
the excluded classes' images (out-of-distribution). Results go to <out>/<dataset>/chain<c>_K<k>/;
a run with the current evaluation_version is skipped, so the sweep can be stopped and resumed.
Legacy metrics are preserved and rejected; use a new --out directory for corrected runs.

    python experiments/class_sweep.py --dataset cifar10 imagenette --chains 3 --k 2-10
    python experiments/class_sweep.py --k 2-4 --steps 1600 --out outputs/class_sweep_fixed_steps
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sdm_vision import data, metrics, sdm, train  # noqa: E402

TARGETS = [0.90, 0.95, 0.99]
BATCH_SIZE = 128


def parse_k(text):
    if "-" in text:
        low, high = map(int, text.split("-"))
        return list(range(low, high + 1))
    return [int(k) for k in text.split(",")]


def softmax(logits, temperature=1.0):
    probs = (logits / temperature).softmax(1).numpy()
    return probs.max(1), probs.argmax(1)


def run_one(dataset, images, labels, class_names, split, chain, k, args, device):
    order = data.class_order(chain)
    classes = order[:k]
    excluded = order[k:]
    remap = np.full(len(class_names), -1)
    remap[classes] = np.arange(k)
    pick = {name: index[np.isin(labels[index], classes)] for name, index in split.items()}
    ood_index = split["validation"][np.isin(labels[split["validation"]], excluded)]
    x = {name: images[index] for name, index in pick.items()}
    y = {name: remap[labels[index]] for name, index in pick.items()}
    assert all((y[name] >= 0).all() for name in y)

    # With --steps, every K gets about the same number of optimizer steps instead of the same number of epochs.
    steps_per_epoch = -(-len(y["train"]) // BATCH_SIZE)
    epochs = -(-args.steps // steps_per_epoch) if args.steps else args.epochs
    start = time.time()
    model, pixels = train.train_resnet(x["train"], y["train"], k, device, epochs=epochs, seed=chain, batch_size=BATCH_SIZE)
    train_seconds = time.time() - start
    logits, embeddings = {}, {}
    for name in ["train", "calibration", "validation"]:
        logits[name], embeddings[name] = train.predict(model, pixels, x[name])
    if len(ood_index):
        logits["ood"], embeddings["ood"] = train.predict(model, pixels, images[ood_index])
    temperature = train.fit_temperature(logits["calibration"], y["calibration"])

    names = [class_names[c] for c in classes]
    artifact = sdm.fit_sdm(embeddings["train"], y["train"], embeddings["calibration"], y["calibration"], names, device,
                           fingerprint=f"{dataset}_chain{chain}_K{k}", epochs=args.sdm_epochs, seed=chain)
    assert sdm.ladder_is_consistent(artifact, k), "region_ladder does not reproduce SDM's regions"
    sdm_cal = sdm.score(artifact, embeddings["calibration"], device)
    sdm_val = sdm.score(artifact, embeddings["validation"], device)
    sdm_ood = sdm.score(artifact, embeddings["ood"], device) if len(ood_index) else None

    # ---- in-set: validation images of the K classes
    y_val, y_cal = y["validation"], y["calibration"]
    val_raw, val_pred = softmax(logits["validation"])
    val_conf, _ = softmax(logits["validation"], temperature)
    cal_conf, cal_pred = softmax(logits["calibration"], temperature)
    val_correct, cal_correct = val_pred == y_val, cal_pred == y_cal
    sdm_correct = sdm_val["prediction"] == y_val
    train_correct = logits["train"].argmax(1).numpy() == y["train"]
    labels_t = torch.from_numpy(y_val).long()
    low, high = metrics.wilson_interval(val_correct.sum(), len(val_correct))
    result = {
        "evaluation_version": metrics.EVALUATION_VERSION,
        "dataset": dataset, "chain": chain, "k": k, "classes": names, "class_ids": classes.tolist(),
        "excluded": [class_names[c] for c in excluded], "epochs": epochs, "train_steps": epochs * steps_per_epoch, "seed": chain,
        "n_train": len(y["train"]), "n_calibration": len(y_cal), "n_validation": len(y_val), "n_ood": int(len(ood_index)),
        "train_seconds": train_seconds,
        "train_accuracy": metrics.accuracy(train_correct),
        "val_accuracy": metrics.accuracy(val_correct), "val_accuracy_interval": [float(low), float(high)],
        "chance_adjusted_accuracy": (metrics.accuracy(val_correct) - 1 / k) / (1 - 1 / k),
        "temperature": temperature,
        "ece_raw": metrics.expected_calibration_error(val_raw, val_correct),
        "ece_scaled": metrics.expected_calibration_error(val_conf, val_correct),
        "nll_raw": F.cross_entropy(logits["validation"], labels_t).item(),
        "nll_scaled": F.cross_entropy(logits["validation"] / temperature, labels_t).item(),
        "softmax_auroc": metrics.auroc(val_conf, val_correct), "softmax_aurc": metrics.aurc(val_conf, val_correct),
        "softmax_e_aurc": metrics.e_aurc(val_conf, val_correct),
        "sdm_accuracy": metrics.accuracy(sdm_correct), "sdm_agreement": float((sdm_val["prediction"] == val_pred).mean()),
        "sdm_auroc": metrics.auroc(sdm_val["probability"], sdm_correct), "sdm_aurc": metrics.aurc(sdm_val["probability"], sdm_correct),
        "sdm_e_aurc": metrics.e_aurc(sdm_val["probability"], sdm_correct),
        "sdm_ood_flag_rate_val": float(sdm_val["is_ood"].mean()),
        "sdm_regions": [round(region["alpha"], 2) for region in artifact.manifest["regions"]],
    }
    thresholds = {}
    for target in TARGETS:
        for rule, conservative in [("plain", False), ("conservative", True)]:
            threshold = metrics.choose_threshold(cal_conf, cal_correct, target, conservative)
            thresholds[(target, rule)] = threshold
            admit = metrics.admitted(val_conf, threshold)
            key = f"softmax_{rule}_{round(target * 100)}"
            result[f"{key}_threshold"] = threshold
            result[f"{key}_coverage"] = float(admit.mean())
            result[f"{key}_accuracy"] = metrics.accuracy(val_correct[admit])

    regions = sorted(result["sdm_regions"], reverse=True)
    result["sdm_top_alpha"] = regions[0] if regions else None
    sdm_admit = {"top": sdm_val["centroid_alpha"] >= (regions[0] - 1e-9 if regions else np.inf),
                 "095": sdm_val["centroid_alpha"] >= 0.95 - 1e-9,
                 "any": sdm_val["centroid_alpha"] > 0}
    for name, admit in sdm_admit.items():
        result[f"sdm_coverage_{name}"] = float(admit.mean())
        result[f"sdm_accuracy_{name}"] = metrics.accuracy(sdm_correct[admit])
    result["sdm_worst_class_precision_top"] = metrics.worst_class_precision(sdm_admit["top"], sdm_val["prediction"], sdm_correct)

    # Match SDM's top-region coverage on calibration only; validation coverages can differ.
    n = len(y_val)
    aurc_difference = lambda i: metrics.aurc(val_conf[i], val_correct[i]) - metrics.aurc(sdm_val["probability"][i], sdm_correct[i])
    result["aurc_softmax_minus_sdm"] = aurc_difference(np.arange(n))
    result["aurc_softmax_minus_sdm_interval"] = metrics.paired_bootstrap(aurc_difference, n)
    matched = np.zeros(n, bool)
    if regions:
        calibration_coverage = float((sdm_cal["centroid_alpha"] >= regions[0] - 1e-9).mean())
        matched_threshold = metrics.threshold_for_coverage(cal_conf, calibration_coverage)
        matched = metrics.admitted(val_conf, matched_threshold)
        result["matched_calibration_sdm_coverage"] = calibration_coverage
        result["matched_calibration_softmax_coverage"] = float(metrics.admitted(cal_conf, matched_threshold).mean())
        result["matched_softmax_threshold"] = matched_threshold
        result["matched_softmax_coverage"] = float(matched.mean())
        result["matched_softmax_accuracy"] = metrics.accuracy(val_correct[matched])
        accuracy_difference = lambda i: metrics.accuracy(sdm_correct[i][sdm_admit["top"][i]]) - metrics.accuracy(val_correct[i][matched[i]])
        result["matched_accuracy_sdm_minus_softmax"] = accuracy_difference(np.arange(n))
        result["matched_accuracy_sdm_minus_softmax_interval"] = metrics.paired_bootstrap(accuracy_difference, n)

    # ---- out-of-distribution: validation images of the excluded classes (every answer is wrong)
    if sdm_ood is not None:
        ood_conf, _ = softmax(logits["ood"], temperature)
        is_ood = np.r_[np.zeros(n, bool), np.ones(len(ood_conf), bool)]
        result |= {
            "ood_softmax_answered_plain_95": float(metrics.admitted(ood_conf, thresholds[(0.95, "plain")]).mean()),
            "ood_softmax_answered_conservative_95": float(metrics.admitted(ood_conf, thresholds[(0.95, "conservative")]).mean()),
            "ood_sdm_answered_095": float((sdm_ood["centroid_alpha"] >= 0.95 - 1e-9).mean()),
            "ood_sdm_answered_top": float((sdm_ood["centroid_alpha"] >= (regions[0] - 1e-9 if regions else np.inf)).mean()),
            "ood_sdm_answered_any": float((sdm_ood["centroid_alpha"] > 0).mean()),
            "ood_softmax_answered_matched": float(metrics.admitted(ood_conf, result["matched_softmax_threshold"]).mean()) if regions else None,
            "ood_sdm_flag_rate": float(sdm_ood["is_ood"].mean()),
            # How well each score separates in-set (high) from excluded-class (low) images.
            "ood_softmax_auroc": metrics.auroc(np.r_[val_conf, ood_conf], ~is_ood),
            "ood_sdm_auroc": metrics.auroc(np.r_[sdm_val["probability"], sdm_ood["probability"]], ~is_ood),
        }

    scores = {"temperature": temperature, "val_labels": y_val, "cal_labels": y_cal,
              "val_logits": logits["validation"].numpy(), "cal_logits": logits["calibration"].numpy(),
              **{f"sdm_cal_{key}": value for key, value in sdm_cal.items()},
              **{f"sdm_val_{key}": value for key, value in sdm_val.items()}}
    if sdm_ood is not None:
        scores |= {"ood_logits": logits["ood"].numpy(), "ood_labels": labels[ood_index],
                   **{f"sdm_ood_{key}": value for key, value in sdm_ood.items()}}
    return result, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", choices=sorted(data.LOADERS), default=sorted(data.LOADERS))
    parser.add_argument("--chains", type=int, default=3, help="number of random class orders (chain c uses seed c)")
    parser.add_argument("--k", type=parse_k, default=list(range(2, 11)), help="class counts, e.g. 2-10 or 2,5,10")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps", type=int, default=None,
                        help="train for about this many optimizer steps at every K (overrides --epochs); 1600 = 40 epochs at K=10")
    parser.add_argument("--sdm-epochs", type=int, default=20)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data")
    parser.add_argument("--out", type=Path, default=REPO / "outputs" / "class_sweep")
    args = parser.parse_args()
    device = train.pick_device()
    print(f"PyTorch {torch.__version__} on {device}; datasets {args.dataset}, chains {args.chains}, K {args.k}, {f'{args.steps} steps' if args.steps else f'{args.epochs} epochs'}", flush=True)

    for dataset in args.dataset:
        images, labels, class_names = data.LOADERS[dataset](args.data_dir)
        split = data.fixed_split(labels)
        for chain in range(args.chains):
            print(f"{dataset} chain {chain}: class order {[class_names[c] for c in data.class_order(chain)]}", flush=True)
            for k in args.k:
                folder = args.out / dataset / f"chain{chain}_K{k}"
                if (folder / "metrics.json").exists():
                    saved = json.loads((folder / "metrics.json").read_text())
                    if saved.get("evaluation_version") != metrics.EVALUATION_VERSION:
                        raise ValueError(f"{folder}: legacy evaluation metrics; preserve this run and use a new --out directory")
                    print(f"  K={k:2d}: done earlier, skipped", flush=True)
                    continue
                start = time.time()
                result, scores = run_one(dataset, images, labels, class_names, split, chain, k, args, device)
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / "scores.npz", **scores)
                (folder / "metrics.json").write_text(json.dumps(result, indent=1))  # written last: marks the run complete
                print(f"  K={k:2d}: val acc {result['val_accuracy']:.1%} (train {result['train_accuracy']:.1%}) | "
                      f"softmax 95%: cov {result['softmax_plain_95_coverage']:.1%} | "
                      f"SDM regions {result['sdm_regions'] or 'none'}, α≥0.95 cov {result['sdm_coverage_095']:.1%} | "
                      f"{result['epochs']} epochs | {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
