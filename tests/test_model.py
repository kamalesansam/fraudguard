import json

import pytest
import skops.io as sio
from sklearn.preprocessing import FunctionTransformer

from fraudguard.evaluate import check_gate, pick_threshold, predict_scores
from fraudguard.integrity import IntegrityError, load_verified, save_model, sha256_file

GATE = {
    "min_roc_auc": 0.95,
    "min_pr_auc": 0.7,
    "min_recall_at_threshold": 0.55,
    "max_slice_recall_gap": 0.3,
}


def test_model_learns_signal(model_dir):
    metrics = json.loads((model_dir / "manifest.json").read_text())["metrics"]
    assert metrics["roc_auc"] > 0.95
    assert metrics["pr_auc"] > 0.7


def test_gate_blocks_weak_model():
    weak = {"roc_auc": 0.7, "pr_auc": 0.3, "recall": 0.2, "slice_recall_gap": 0.5}
    result = check_gate(weak, GATE)
    assert not result.passed
    assert len(result.failures) == 4


def test_threshold_meets_target_precision(model, splits):
    _, test = splits
    y = test["is_fraud"].to_numpy()
    scores = predict_scores(model, test)
    t = pick_threshold(y, scores, 0.8)
    flagged = scores >= t
    assert (y[flagged] == 1).mean() >= 0.8


def test_serialization_is_reproducible(model, tmp_path):
    a = save_model(model, tmp_path / "a.skops")
    b = save_model(model, tmp_path / "b.skops")
    assert a == b


def test_verified_load_roundtrip(model, model_dir, splits):
    manifest = json.loads((model_dir / "manifest.json").read_text())
    loaded = load_verified(model_dir / "model.skops", manifest["model"]["sha256"])
    _, test = splits
    assert (predict_scores(loaded, test) == predict_scores(model, test)).all()


def test_tampered_model_is_rejected(model_dir, tmp_path):
    original = model_dir / "model.skops"
    tampered = tmp_path / "model.skops"
    blob = bytearray(original.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    tampered.write_bytes(bytes(blob))
    with pytest.raises(IntegrityError, match="sha256 mismatch"):
        load_verified(tampered, sha256_file(original))


def test_artifact_with_dangerous_type_is_rejected(tmp_path):
    """A model smuggling a callable like eval must never be deserialized."""
    evil = FunctionTransformer(func=eval)
    path = tmp_path / "evil.skops"
    sio.dump(evil, path)
    with pytest.raises(IntegrityError, match="outside the allowlist"):
        load_verified(path, sha256_file(path))


def test_save_refuses_unreviewed_types(tmp_path):
    with pytest.raises(IntegrityError):
        save_model(FunctionTransformer(func=eval), tmp_path / "evil.skops")
    assert not (tmp_path / "evil.skops").exists()
