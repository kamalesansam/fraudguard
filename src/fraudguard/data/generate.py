"""Seeded synthetic card-transaction generator.

Real fraud datasets are either PCA-anonymised (no feature meaning) or gated
behind licences, which breaks reproducibility. This generator produces
interpretable features with three fraud archetypes, so every run of the
pipeline is reproducible from source with a single seed.

`drift` and `concept_drift` simulate what production looks like months later:
covariate shift (inflation, more night/online traffic, new devices) and a new
fraud pattern the training data never contained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery",
    "restaurant",
    "gas",
    "online_retail",
    "electronics",
    "travel",
    "jewelry",
    "gaming",
    "other",
]
ONLINE_CATEGORIES = {"online_retail", "gaming"}

_LEGIT_MERCHANT_P = [0.25, 0.18, 0.12, 0.15, 0.06, 0.05, 0.02, 0.04, 0.13]
_DAY_HOUR_P = np.array(
    [1, 1, 1, 1, 1, 2, 3, 5, 6, 7, 7, 8, 9, 8, 7, 7, 8, 9, 9, 8, 6, 4, 3, 2], dtype=float
)
_NIGHT_HOUR_P = np.array(
    [8, 9, 9, 8, 7, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8], dtype=float
)


def _hours(rng: np.random.Generator, n: int, night_weight: float) -> np.ndarray:
    p = (1 - night_weight) * _DAY_HOUR_P / _DAY_HOUR_P.sum() + night_weight * (
        _NIGHT_HOUR_P / _NIGHT_HOUR_P.sum()
    )
    return rng.choice(24, size=n, p=p / p.sum())


def _legit(rng: np.random.Generator, n: int, drift: float) -> pd.DataFrame:
    merchant_p = np.array(_LEGIT_MERCHANT_P)
    if drift:
        # Online share grows, in-store share shrinks.
        shift = np.array([-0.05, -0.03, -0.02, 0.08, 0.0, 0.0, 0.0, 0.02, 0.0])
        merchant_p = merchant_p + drift * shift
        merchant_p = np.clip(merchant_p, 0.005, None)
    merchant = rng.choice(MERCHANT_CATEGORIES, size=n, p=merchant_p / merchant_p.sum())
    online = np.isin(merchant, list(ONLINE_CATEGORIES))

    avg_30d = rng.lognormal(np.log(60 * (1 + 0.5 * drift)), 0.6, n)
    amount = avg_30d * rng.lognormal(0.0, 0.5, n)
    return pd.DataFrame(
        {
            "amount": amount,
            "hour": _hours(rng, n, night_weight=0.05 + 0.15 * drift),
            "merchant_category": merchant,
            "distance_from_home_km": rng.lognormal(np.log(5), 1.0, n),
            "card_present": np.where(online, rng.random(n) < 0.05, rng.random(n) < 0.75),
            "is_foreign": np.where(
                merchant == "travel", rng.random(n) < 0.25, rng.random(n) < 0.03
            ),
            "txn_count_24h": rng.poisson(2.5 + drift, n),
            "avg_amount_30d": avg_30d,
            "account_age_days": rng.gamma(2.0, 600.0, n).astype(int) + 30,
            "new_device": rng.random(n) < (0.04 + 0.10 * drift),
        }
    )


def _fraud(rng: np.random.Generator, n: int, drift: float, concept_drift: bool) -> pd.DataFrame:
    weights = [0.30, 0.40, 0.30, 0.0]
    if concept_drift:
        # A new archetype appears that mimics normal daytime spend.
        weights = [0.15, 0.20, 0.15, 0.50]
    archetype = rng.choice(4, size=n, p=weights)
    avg_30d = rng.lognormal(np.log(60 * (1 + 0.5 * drift)), 0.6, n)

    amount = np.empty(n)
    hour = np.empty(n, dtype=int)
    merchant = np.empty(n, dtype=object)
    distance = np.empty(n)
    card_present = np.empty(n, dtype=bool)
    foreign = np.empty(n, dtype=bool)
    velocity = np.empty(n, dtype=int)
    new_device = np.empty(n, dtype=bool)
    age = rng.gamma(2.0, 600.0, n).astype(int) + 30

    for a in range(4):
        m = archetype == a
        k = int(m.sum())
        if k == 0:
            continue
        if a == 0:  # card testing: many tiny online charges from a new device
            amount[m] = rng.uniform(0.5, 15.0, k)
            hour[m] = _hours(rng, k, night_weight=0.6)
            merchant[m] = rng.choice(
                ["online_retail", "gaming", "other"], size=k, p=[0.5, 0.4, 0.1]
            )
            distance[m] = rng.lognormal(np.log(40), 1.2, k)
            card_present[m] = rng.random(k) < 0.02
            foreign[m] = rng.random(k) < 0.25
            velocity[m] = rng.poisson(9, k)
            new_device[m] = rng.random(k) < 0.7
        elif a == 1:  # account takeover: large spend far from home, at night
            amount[m] = avg_30d[m] * rng.lognormal(np.log(5), 0.6, k)
            hour[m] = _hours(rng, k, night_weight=0.7)
            merchant[m] = rng.choice(
                ["electronics", "jewelry", "online_retail", "travel"],
                size=k,
                p=[0.35, 0.2, 0.3, 0.15],
            )
            distance[m] = rng.lognormal(np.log(150), 1.0, k)
            card_present[m] = rng.random(k) < 0.15
            foreign[m] = rng.random(k) < 0.35
            velocity[m] = rng.poisson(4, k)
            new_device[m] = rng.random(k) < 0.6
            age[m] = rng.gamma(2.0, 400.0, k).astype(int) + 30
        elif a == 2:  # stolen physical card: in-store, resale-friendly goods
            amount[m] = avg_30d[m] * rng.lognormal(np.log(3), 0.7, k)
            hour[m] = _hours(rng, k, night_weight=0.3)
            merchant[m] = rng.choice(["electronics", "jewelry", "gas"], size=k, p=[0.5, 0.3, 0.2])
            distance[m] = rng.lognormal(np.log(60), 1.2, k)
            card_present[m] = rng.random(k) < 0.8
            foreign[m] = rng.random(k) < 0.2
            velocity[m] = rng.poisson(5, k)
            new_device[m] = rng.random(k) < 0.1
        else:  # concept drift archetype: looks like normal daytime grocery spend
            amount[m] = avg_30d[m] * rng.lognormal(0.2, 0.5, k)
            hour[m] = _hours(rng, k, night_weight=0.05)
            merchant[m] = rng.choice(["grocery", "restaurant", "other"], size=k)
            distance[m] = rng.lognormal(np.log(8), 1.0, k)
            card_present[m] = rng.random(k) < 0.6
            foreign[m] = rng.random(k) < 0.03
            velocity[m] = rng.poisson(3, k)
            new_device[m] = rng.random(k) < 0.05

    return pd.DataFrame(
        {
            "amount": amount,
            "hour": hour,
            "merchant_category": merchant.astype(str),
            "distance_from_home_km": distance,
            "card_present": card_present,
            "is_foreign": foreign,
            "txn_count_24h": velocity,
            "avg_amount_30d": avg_30d,
            "account_age_days": age,
            "new_device": new_device,
        }
    )


def generate_transactions(
    n_rows: int,
    fraud_rate: float,
    seed: int,
    drift: float = 0.0,
    concept_drift: bool = False,
) -> pd.DataFrame:
    """Return `n_rows` shuffled transactions with an `is_fraud` label."""
    if not 0.0 <= drift <= 1.0:
        raise ValueError("drift must be within [0, 1]")
    rng = np.random.default_rng(seed)
    n_fraud = round(n_rows * fraud_rate)
    legit = _legit(rng, n_rows - n_fraud, drift).assign(is_fraud=0)
    fraud = _fraud(rng, n_fraud, drift, concept_drift).assign(is_fraud=1)

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # 1% label noise: chargebacks get filed late or wrongly, real labels are never clean.
    noise = rng.random(len(df)) < 0.01
    df.loc[noise & (df["is_fraud"] == 0) & (rng.random(len(df)) < 0.05), "is_fraud"] = 1

    for col in ("card_present", "is_foreign", "new_device"):
        df[col] = df[col].astype(int)
    df["amount"] = df["amount"].clip(0.5, 50_000).round(2)
    df["avg_amount_30d"] = df["avg_amount_30d"].round(2)
    df["distance_from_home_km"] = df["distance_from_home_km"].clip(0, 20_000).round(2)
    df.insert(0, "transaction_id", [f"tx_{seed}_{i:07d}" for i in range(len(df))])
    return df
