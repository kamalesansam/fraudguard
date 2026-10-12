import pandas as pd
import pytest

from fraudguard.data.generate import MERCHANT_CATEGORIES, generate_transactions
from fraudguard.data.validate import validate_transactions
from fraudguard.features import MODEL_FEATURES, build_features
from fraudguard.serve.schemas import MerchantCategory

GATE = {"fraud_rate_min": 0.005, "fraud_rate_max": 0.06, "max_duplicate_share": 0.001}


def test_generator_is_deterministic():
    a = generate_transactions(2000, 0.02, seed=3)
    b = generate_transactions(2000, 0.02, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_generator_hits_fraud_rate():
    df = generate_transactions(20000, 0.02, seed=3)
    assert 0.015 < df["is_fraud"].mean() < 0.03


def test_generator_rejects_bad_drift():
    with pytest.raises(ValueError):
        generate_transactions(100, 0.02, seed=1, drift=1.5)


def test_clean_data_passes_contract(dataset):
    assert validate_transactions(dataset, **GATE).passed


@pytest.mark.parametrize(
    ("column", "value"),
    [("amount", -10.0), ("hour", 25), ("merchant_category", "casino"), ("card_present", 2)],
)
def test_contract_rejects_bad_values(dataset, column, value):
    bad = dataset.copy()
    bad.loc[bad.index[0], column] = value
    report = validate_transactions(bad, **GATE)
    assert not report.passed
    assert any(column in e for e in report.errors)


def test_contract_rejects_unknown_columns(dataset):
    assert not validate_transactions(dataset.assign(ssn="000-00-0000"), **GATE).passed


def test_contract_catches_label_flipping(dataset):
    """Flipping 10% of labels to fraud pushes the label rate out of band."""
    poisoned = dataset.copy()
    flip = poisoned.sample(frac=0.10, random_state=0).index
    poisoned.loc[flip, "is_fraud"] = 1
    report = validate_transactions(poisoned, **GATE)
    assert not report.passed
    assert any(e.startswith("label-rate") for e in report.errors)


def test_contract_catches_duplicates(dataset):
    dup = pd.concat(
        [dataset, dataset.head(200).assign(transaction_id=lambda d: d.transaction_id + "x")]
    )
    assert not validate_transactions(dup, **GATE).passed


def test_features_are_complete_and_finite(dataset):
    feats = build_features(dataset)
    assert list(feats.columns) == MODEL_FEATURES
    assert not feats.drop(columns="merchant_category").isna().any().any()


def test_features_require_raw_columns(dataset):
    with pytest.raises(ValueError, match="missing raw features"):
        build_features(dataset.drop(columns="amount"))


def test_api_schema_matches_generator_categories():
    assert set(MerchantCategory.__args__) == set(MERCHANT_CATEGORIES)
