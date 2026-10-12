"""Train stage: fit the full preprocessing + model pipeline on the train split."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fraudguard.data.generate import MERCHANT_CATEGORIES
from fraudguard.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features


def build_pipeline(train_params: dict[str, Any], seed: int) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "merchant",
                OneHotEncoder(categories=[MERCHANT_CATEGORIES], handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        sparse_threshold=0.0,
    )
    model = HistGradientBoostingClassifier(
        learning_rate=train_params["learning_rate"],
        max_iter=train_params["max_iter"],
        max_leaf_nodes=train_params["max_leaf_nodes"],
        min_samples_leaf=train_params["min_samples_leaf"],
        l2_regularization=train_params["l2_regularization"],
        early_stopping=train_params["early_stopping"],
        random_state=seed,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def fit(train_df: pd.DataFrame, train_params: dict[str, Any], seed: int) -> Pipeline:
    pipeline = build_pipeline(train_params, seed)
    pipeline.fit(build_features(train_df), train_df["is_fraud"].to_numpy())
    return pipeline
