"""Feature engineering shared by training and serving.

There is exactly one implementation of every derived feature. The API calls
the same `build_features` as the training stage, which rules out
training/serving skew from two diverging copies of the logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_FEATURES = [
    "amount",
    "hour",
    "merchant_category",
    "distance_from_home_km",
    "card_present",
    "is_foreign",
    "txn_count_24h",
    "avg_amount_30d",
    "account_age_days",
    "new_device",
]

CATEGORICAL_FEATURES = ["merchant_category"]

NUMERIC_FEATURES = [
    "log_amount",
    "amount_to_avg_ratio",
    "log_distance",
    "hour_sin",
    "hour_cos",
    "is_night",
    "card_present",
    "is_foreign",
    "txn_count_24h",
    "account_age_days",
    "new_device",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw transaction fields to the model's input columns."""
    missing = set(RAW_FEATURES) - set(raw.columns)
    if missing:
        raise ValueError(f"missing raw features: {sorted(missing)}")

    hour = raw["hour"].astype(float)
    out = pd.DataFrame(index=raw.index)
    out["merchant_category"] = raw["merchant_category"].astype(str)
    out["log_amount"] = np.log1p(raw["amount"].astype(float))
    out["amount_to_avg_ratio"] = raw["amount"] / (raw["avg_amount_30d"] + 1.0)
    out["log_distance"] = np.log1p(raw["distance_from_home_km"].astype(float))
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["is_night"] = ((hour < 6) | (hour >= 23)).astype(int)
    for col in ("card_present", "is_foreign", "txn_count_24h", "account_age_days", "new_device"):
        out[col] = raw[col].astype(int)
    return out[MODEL_FEATURES]
