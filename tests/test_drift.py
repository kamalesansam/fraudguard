import json

import numpy as np

from fraudguard.data.generate import generate_transactions
from fraudguard.integrity import load_verified
from fraudguard.monitor.drift import psi_numeric, run_drift_check

PARAMS = {"psi_threshold": 0.2, "max_drifted_share": 0.15}


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    assert psi_numeric(rng.normal(size=5000), rng.normal(size=5000)) < 0.02


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    assert psi_numeric(rng.normal(size=5000), rng.normal(1.0, size=5000)) > 0.5


def test_stable_batch_raises_no_alert(splits):
    train, _ = splits
    current = generate_transactions(4000, 0.03, seed=11)
    assert not run_drift_check(train, current, PARAMS)["alert"]


def test_covariate_shift_is_detected(splits):
    train, _ = splits
    current = generate_transactions(4000, 0.03, seed=12, drift=1.0)
    report = run_drift_check(train, current, PARAMS)
    assert report["dataset_drift"]
    assert report["features"]["avg_amount_30d"]["drifted"]


def test_concept_drift_detected_from_labels(splits, model_dir):
    """Inputs look identical, but a new fraud pattern slips past the model."""
    train, _ = splits
    manifest = json.loads((model_dir / "manifest.json").read_text())
    model = load_verified(model_dir / "model.skops", manifest["model"]["sha256"])
    current = generate_transactions(6000, 0.03, seed=13, concept_drift=True)
    report = run_drift_check(train, current, PARAMS, model=model, manifest=manifest)
    assert not report["dataset_drift"]
    assert report["performance"]["concept_drift"]
    assert report["alert"]
