"""Accuracy, calibration and selective-prediction metrics, as in notebooks 04-10."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def wilson_interval(successes, n, z=1.96):
    # 95% confidence interval for a proportion; works on numbers or arrays.
    successes, n = np.asarray(successes, dtype=float), np.asarray(n, dtype=float)
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return center - half_width, center + half_width


def expected_calibration_error(confidence, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    table = pd.DataFrame({"bin": which, "confidence": confidence, "correct": correct}).groupby("bin").agg(
        count=("correct", "size"), confidence=("confidence", "mean"), accuracy=("correct", "mean"))
    return float((table["count"] / len(confidence) * (table["confidence"] - table["accuracy"]).abs()).sum())


def choose_threshold(confidence, correct, target, conservative=False):
    # The lowest confidence at which the answered images are at least `target` correct (plain),
    # or the lower end of their 95% Wilson interval is (conservative); None if no threshold reaches it.
    order = np.argsort(-confidence)
    answered = np.arange(1, len(order) + 1)
    hits = np.cumsum(correct[order])
    score = wilson_interval(hits, answered)[0] if conservative else hits / answered
    meets = np.flatnonzero(score >= target)
    return None if len(meets) == 0 else float(confidence[order][meets.max()])


def admitted(confidence, threshold):
    return confidence >= threshold if threshold is not None else np.zeros(len(confidence), bool)


def coverage_accuracy_curve(score, correct):
    # Coverage and accuracy when answering the k highest-scoring images, for every k.
    order = np.argsort(-score, kind="stable")
    answered = np.arange(1, len(order) + 1)
    return answered / len(order), np.cumsum(correct[order]) / answered


def aurc(score, correct):
    # Area under the risk-coverage curve: the average error rate over all coverage levels (lower is better).
    return float(np.mean(1 - coverage_accuracy_curve(score, correct)[1]))


def e_aurc(score, correct):
    # AURC minus the AURC of a perfect ranking with the same errors, so it is comparable across accuracy levels.
    correct = np.asarray(correct)
    return aurc(score, correct) - aurc(correct.astype(float), correct)


def auroc(score, positive):
    positive = np.asarray(positive, dtype=bool)
    return float(roc_auc_score(positive, score)) if 0 < positive.sum() < len(positive) else float("nan")


def accuracy(correct):
    correct = np.asarray(correct)
    return float(correct.mean()) if len(correct) else float("nan")


def worst_class_precision(admit, prediction, correct, min_answers=5):
    # Lowest accuracy among predicted classes with at least `min_answers` answers ("when it says lion, is it right?").
    precision = pd.Series(correct[admit]).groupby(prediction[admit])
    precision = precision.mean()[precision.size() >= min_answers]
    return float(precision.min()) if len(precision) else float("nan")


def paired_bootstrap(statistic, n, reps=1000, seed=0):
    # 95% percentile interval of statistic(index) over resamples of the n validation images, kept paired across methods.
    rng = np.random.default_rng(seed)
    values = np.array([statistic(rng.integers(0, n, n)) for _ in range(reps)], dtype=float)
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]
