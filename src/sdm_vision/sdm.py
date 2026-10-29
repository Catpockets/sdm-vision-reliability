"""Fitting, scoring and diagnosing SDM, as in notebooks 05-10."""

import numpy as np
from reexpress_sdm import SDMModel, TorchTrainer, TrainingConfig


def fit_sdm(train_embeddings, train_labels, calibration_embeddings, calibration_labels, class_names, device, fingerprint,
            epochs=20, seed=0):
    # Calling the trainer directly keeps our splits; the command-line tool would pool and reshuffle them.
    trainer = TorchTrainer(TrainingConfig(number_of_classes=len(class_names), epochs=epochs, seed=seed), device=str(device))
    result = trainer.fit(
        train_embeddings, np.asarray(train_labels).tolist(),
        calibration_embeddings, np.asarray(calibration_labels).tolist(),
        train_ids=[f"train-{i}" for i in range(len(train_labels))],
        calibration_ids=[f"calibration-{i}" for i in range(len(calibration_labels))],
        class_names=list(class_names),
        representation_fingerprint=fingerprint,
    )
    return result.artifact


def score(artifact, embeddings, device):
    scores = SDMModel(artifact, device=str(device)).score(embeddings, nearest_exemplars=0)
    return {
        "prediction": np.array([s.prediction for s in scores]),
        "probability": np.array([s.sdm[s.prediction] for s in scores]),
        "centroid_alpha": np.array([s.centroid_region_alpha for s in scores]),
        "lower_alpha": np.array([s.lower_region_alpha for s in scores]),
        "is_ood": np.array([s.is_ood for s in scores]),
    }


def region_ladder(artifact, number_of_classes):
    # Replays SDM's nested calibration. At each alpha of the ladder, find the similarity cutoff that leaves
    # the fewest failing labels among the calibration images no earlier region took; if SDM formed a region
    # at that alpha, remove the images it accepts before moving on, exactly as the package does.
    rows = artifact.calibration_rows
    probs = np.array([row["sdm"] for row in rows], dtype=np.float32)
    q_prime = np.array([row["qPrime"] for row in rows], dtype=np.float32)
    labels = np.array([int(row["label"]) for row in rows])
    predictions = np.array([int(row["prediction"]) for row in rows])
    p_true = probs[np.arange(len(labels)), labels]
    configuration = artifact.manifest["configuration"]
    regions = {round(region["alpha"], 10): region for region in artifact.manifest["regions"]}
    residual = np.floor(q_prime) > configuration["oodLimit"]
    ladder = {}
    for step in range(1, round(0.5 / configuration["alphaResolution"]) + 1):
        alpha = round(1 - step * configuration["alphaResolution"], 10)
        if alpha <= 0.5 or not residual.any():
            break
        below = p_true < np.float32(alpha)
        best = None
        for cutoff in np.unique(q_prime[residual]):
            kept = residual & (q_prime >= cutoff)
            count = np.bincount(labels[kept], minlength=number_of_classes)
            misses = np.bincount(labels[kept & below], minlength=number_of_classes)
            allowed = np.floor(round(1 - alpha, 10) * count).astype(int)
            failing = (count == 0) | (misses > allowed)
            if best is None or failing.sum() < best["failing"].sum():
                best = {"failing": failing, "count": count, "misses": misses, "allowed": allowed}
        region = regions.get(alpha)
        ladder[alpha] = best | {"region": region, "rows left": int(residual.sum())}
        if region is not None:
            above = probs >= np.asarray(region["outputThresholds"], dtype=np.float32)
            accepted = (residual & (q_prime >= np.float32(region["minimumRescaledSimilarity"]))
                        & (above.sum(1) == 1) & above[np.arange(len(labels)), predictions])
            residual &= ~accepted
    return ladder


def ladder_is_consistent(artifact, number_of_classes):
    # 0 failing labels exactly at the alphas where SDM formed a region: the replay matches the package.
    return all((step["failing"].sum() == 0) == (step["region"] is not None)
               for step in region_ladder(artifact, number_of_classes).values())
