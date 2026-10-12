"""Data contract for raw transactions, enforced before any training happens.

Column-level rules live in a pandera schema. Dataset-level rules (label-rate
band, duplicate share) catch problems a per-row schema cannot see, such as a
broken upstream join or a crude label-flipping attack.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from fraudguard.data.generate import MERCHANT_CATEGORIES

_BINARY = pa.Check.isin([0, 1])

SCHEMA = pa.DataFrameSchema(
    {
        "transaction_id": pa.Column(str, unique=True),
        "amount": pa.Column(float, pa.Check.in_range(0.01, 50_000)),
        "hour": pa.Column(int, pa.Check.in_range(0, 23)),
        "merchant_category": pa.Column(str, pa.Check.isin(MERCHANT_CATEGORIES)),
        "distance_from_home_km": pa.Column(float, pa.Check.in_range(0, 20_000)),
        "card_present": pa.Column(int, _BINARY),
        "is_foreign": pa.Column(int, _BINARY),
        "txn_count_24h": pa.Column(int, pa.Check.in_range(0, 500)),
        "avg_amount_30d": pa.Column(float, pa.Check.in_range(0, 50_000)),
        "account_age_days": pa.Column(int, pa.Check.in_range(0, 36_500)),
        "new_device": pa.Column(int, _BINARY),
        "is_fraud": pa.Column(int, _BINARY),
    },
    strict=True,
)


@dataclass
class ValidationReport:
    passed: bool
    n_rows: int
    fraud_rate: float
    duplicate_share: float
    errors: list[str] = field(default_factory=list)


def validate_transactions(
    df: pd.DataFrame,
    fraud_rate_min: float,
    fraud_rate_max: float,
    max_duplicate_share: float,
) -> ValidationReport:
    errors: list[str] = []
    try:
        SCHEMA.validate(df, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases
        for _, row in cases.head(20).iterrows():
            errors.append(
                f"schema: column={row['column']} check={row['check']} value={row['failure_case']}"
            )
        if len(cases) > 20:
            errors.append(f"schema: ... {len(cases) - 20} more failure cases")

    fraud_rate = float(df["is_fraud"].mean()) if "is_fraud" in df else float("nan")
    if not fraud_rate_min <= fraud_rate <= fraud_rate_max:
        errors.append(
            f"label-rate: fraud rate {fraud_rate:.4f} outside [{fraud_rate_min}, {fraud_rate_max}]"
        )

    content_cols = [c for c in df.columns if c != "transaction_id"]
    duplicate_share = float(df.duplicated(subset=content_cols).mean()) if len(df) else 0.0
    if duplicate_share > max_duplicate_share:
        errors.append(f"duplicates: {duplicate_share:.4%} of rows are exact duplicates")

    return ValidationReport(
        passed=not errors,
        n_rows=len(df),
        fraud_rate=round(fraud_rate, 6),
        duplicate_share=round(duplicate_share, 6),
        errors=errors,
    )


def write_report(report: ValidationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2) + "\n")
