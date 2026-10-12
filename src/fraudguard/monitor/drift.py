"""Data drift, prediction drift and concept drift checks for production batches.

- Data drift: Population Stability Index per feature (KS p-values reported
  alongside; with large batches KS flags trivial shifts, so PSI drives the
  decision).
- Prediction drift: PSI on the model's score distribution.
- Concept drift: when delayed labels arrive, recall at the production
  threshold is compared to the recall recorded in the model manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from fraudguard.features import RAW_FEATURES

_EPS = 1e-6
CATEGORICAL = {"merchant_category", "card_present", "is_foreign", "new_device"}


def psi_numeric(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference) + _EPS
    cur = np.histogram(current, edges)[0] / len(current) + _EPS
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    levels = sorted(set(reference.unique()) | set(current.unique()), key=str)
    ref = reference.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy() + _EPS
    cur = current.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy() + _EPS
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, params: dict[str, float]) -> dict:
    features = {}
    for col in RAW_FEATURES:
        if col in CATEGORICAL:
            psi, pvalue = psi_categorical(reference[col], current[col]), None
        else:
            ref, cur = reference[col].to_numpy(float), current[col].to_numpy(float)
            psi, pvalue = psi_numeric(ref, cur), float(ks_2samp(ref, cur).pvalue)
        features[col] = {
            "psi": round(psi, 4),
            "ks_pvalue": None if pvalue is None else round(pvalue, 6),
            "drifted": psi > params["psi_threshold"],
        }
    share = sum(f["drifted"] for f in features.values()) / len(features)
    return {
        "features": features,
        "drifted_share": round(share, 4),
        "dataset_drift": share > params["max_drifted_share"],
    }


def run_drift_check(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    params: dict[str, float],
    model: Any | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "n_reference": len(reference),
        "n_current": len(current),
        **feature_drift(reference, current, params),
    }

    if model is not None:
        from fraudguard.evaluate import predict_scores

        ref_scores = predict_scores(model, reference)
        cur_scores = predict_scores(model, current)
        score_psi = psi_numeric(ref_scores, cur_scores)
        report["prediction_drift"] = {
            "score_psi": round(score_psi, 4),
            "drifted": score_psi > params["psi_threshold"],
        }

        if manifest is not None and "is_fraud" in current:
            threshold = manifest["model"]["threshold"]
            y = current["is_fraud"].to_numpy()
            caught = (cur_scores >= threshold)[y == 1]
            recall = float(caught.mean()) if len(caught) else float("nan")
            baseline = manifest["metrics"]["recall"]
            report["performance"] = {
                "baseline_recall": baseline,
                "current_recall": round(recall, 4),
                "recall_drop": round(baseline - recall, 4),
                "concept_drift": (baseline - recall) > 0.10,
            }

    report["alert"] = bool(
        report["dataset_drift"]
        or report.get("prediction_drift", {}).get("drifted")
        or report.get("performance", {}).get("concept_drift")
    )
    return report


def write_evidently_html(reference: pd.DataFrame, current: pd.DataFrame, path: Path) -> bool:
    """Optional human-readable drift report. Skipped if Evidently is not installed."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return False
    snapshot = Report([DataDriftPreset()]).run(current[RAW_FEATURES], reference[RAW_FEATURES])
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    return True
