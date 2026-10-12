"""Evaluate stage: pick the operating threshold and enforce the quality gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraudguard.features import build_features

SLICE_COLUMN = "card_present"


def predict_scores(model: Any, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(build_features(df))[:, 1]


def pick_threshold(y_true: np.ndarray, scores: np.ndarray, target_precision: float) -> float:
    """Lowest threshold whose precision meets the target, i.e. max recall at that precision."""
    precision, _, thresholds = precision_recall_curve(y_true, scores)
    ok = np.where(precision[:-1] >= target_precision)[0]
    if len(ok) == 0:
        return 0.5
    return float(thresholds[ok[0]])


def compute_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float, slices: pd.Series
) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    slice_recall = {}
    for value in sorted(slices.unique()):
        mask = (slices == value).to_numpy() & (y_true == 1)
        if mask.sum() >= 10:
            slice_recall[f"{SLICE_COLUMN}={value}"] = round(float(y_pred[mask].mean()), 4)
    gap = max(slice_recall.values()) - min(slice_recall.values()) if slice_recall else 0.0

    return {
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "threshold": round(threshold, 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "slice_recall": slice_recall,
        "slice_recall_gap": round(float(gap), 4),
    }


@dataclass
class GateResult:
    passed: bool
    failures: list[str]


def check_gate(metrics: dict, gate: dict[str, float]) -> GateResult:
    failures = []
    if metrics["roc_auc"] < gate["min_roc_auc"]:
        failures.append(f"roc_auc {metrics['roc_auc']} < {gate['min_roc_auc']}")
    if metrics["pr_auc"] < gate["min_pr_auc"]:
        failures.append(f"pr_auc {metrics['pr_auc']} < {gate['min_pr_auc']}")
    if metrics["recall"] < gate["min_recall_at_threshold"]:
        failures.append(f"recall {metrics['recall']} < {gate['min_recall_at_threshold']}")
    if metrics["slice_recall_gap"] > gate["max_slice_recall_gap"]:
        failures.append(
            f"slice recall gap {metrics['slice_recall_gap']} > {gate['max_slice_recall_gap']}"
        )
    return GateResult(passed=not failures, failures=failures)


def pr_curve_points(y_true: np.ndarray, scores: np.ndarray, n: int = 100) -> list[dict]:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    idx = np.linspace(0, len(precision) - 1, num=min(n, len(precision))).astype(int)
    return [
        {"recall": round(float(recall[i]), 4), "precision": round(float(precision[i]), 4)}
        for i in idx
    ]
