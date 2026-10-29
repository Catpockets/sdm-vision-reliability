"""Accuracy, calibration and selective-prediction metrics, as in notebooks 04-10."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EVALUATION_VERSION = "whole-score-groups-calibration-match-1"


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
    confidence, correct = np.asarray(confidence), np.asarray(correct, dtype=bool)
    if not len(confidence):
        return None
    order = np.argsort(-confidence, kind="stable")
    sorted_scores = confidence[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(order) - 1]
    answered = ends + 1
    hits = np.cumsum(correct[order])[ends]
    score = wilson_interval(hits, answered)[0] if conservative else hits / answered
    meets = np.flatnonzero(score >= target)
    return None if len(meets) == 0 else float(sorted_scores[ends[meets[-1]]])


def admitted(confidence, threshold):
    return confidence >= threshold if threshold is not None else np.zeros(len(confidence), bool)


def coverage_accuracy_curve(score, correct):
    # Only whole equal-score groups are attainable by a confidence threshold.
    score, correct = np.asarray(score), np.asarray(correct, dtype=bool)
    if not len(score):
        return np.array([]), np.array([])
    order = np.argsort(-score, kind="stable")
    sorted_scores = score[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(order) - 1]
    answered = ends + 1
    return answered / len(order), np.cumsum(correct[order])[ends] / answered


def threshold_for_coverage(confidence, coverage):
    """Largest whole-group calibration coverage no greater than the budget."""
    if not 0 <= coverage <= 1:
        raise ValueError("Coverage must lie in [0, 1]")
    scores = np.sort(np.asarray(confidence))[::-1]
    if not len(scores):
        return None
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    eligible = ends[(ends + 1) / len(scores) <= coverage]
    return float(scores[eligible[-1]]) if len(eligible) else None


def aurc(score, correct):
    # Right-endpoint step integral, weighted by the size of each confidence group.
    coverage, accuracy = coverage_accuracy_curve(score, correct)
    return float(np.sum(np.diff(np.r_[0., coverage]) * (1 - accuracy))) if len(coverage) else float("nan")


def e_aurc(score, correct):
    # AURC minus the AURC of a perfect ranking with the same errors, so it is comparable across accuracy levels.
    correct = np.asarray(correct)
    if not len(correct):
        return float("nan")
    # The oracle can rank every correct image before every error without ties.
    oracle_risk = 1 - np.cumsum(np.sort(correct)[::-1]) / np.arange(1, len(correct) + 1)
    return aurc(score, correct) - float(oracle_risk.mean())


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
