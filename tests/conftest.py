from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from fraudguard.data.generate import generate_transactions
from fraudguard.evaluate import SLICE_COLUMN, compute_metrics, pick_threshold, predict_scores
from fraudguard.integrity import save_model
from fraudguard.package import build_manifest, write_package
from fraudguard.train import fit

TRAIN_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    return generate_transactions(n_rows=8000, fraud_rate=0.03, seed=1)


@pytest.fixture(scope="session")
def splits(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        dataset, test_size=0.3, stratify=dataset["is_fraud"], random_state=1
    )
    return train, test


@pytest.fixture(scope="session")
def model(splits):
    return fit(splits[0], TRAIN_PARAMS, seed=1)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, model, splits) -> Path:
    """A packaged model directory (model.skops + manifest.json) like the pipeline emits."""
    out = tmp_path_factory.mktemp("model")
    train, test = splits
    save_model(model, out / "model.skops")
    train_file = out / "train.csv"
    train.to_csv(train_file, index=False)
    scores = predict_scores(model, test)
    threshold = pick_threshold(test["is_fraud"].to_numpy(), scores, 0.8)
    metrics = compute_metrics(test["is_fraud"].to_numpy(), scores, threshold, test[SLICE_COLUMN])
    params = {"seed": 1, "train": TRAIN_PARAMS}
    write_package(out, build_manifest(out / "model.skops", train_file, metrics, params, out))
    assert json.loads((out / "manifest.json").read_text())["model"]["format"] == "skops"
    return out
