"""Request/response contracts. Unknown fields, wrong types, NaN/inf and
out-of-range values are rejected with 422 before reaching the model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MerchantCategory = Literal[
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


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    amount: float = Field(gt=0, le=50_000, examples=[129.99])
    hour: int = Field(ge=0, le=23, examples=[2])
    merchant_category: MerchantCategory = Field(examples=["electronics"])
    distance_from_home_km: float = Field(ge=0, le=20_000, examples=[312.5])
    card_present: bool = Field(examples=[False])
    is_foreign: bool = Field(examples=[True])
    txn_count_24h: int = Field(ge=0, le=500, examples=[7])
    avg_amount_30d: float = Field(ge=0, le=50_000, examples=[54.2])
    account_age_days: int = Field(ge=0, le=36_500, examples=[210])
    new_device: bool = Field(examples=[True])


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[Transaction] = Field(min_length=1, max_length=100)


class Prediction(BaseModel):
    fraud_score: float
    is_fraud: bool


class PredictionResponse(Prediction):
    request_id: str
    threshold: float
    model_version: str


class BatchResponse(BaseModel):
    request_id: str
    threshold: float
    model_version: str
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    version: str
    sha256: str
    threshold: float
    created_at: str
    git_commit: str
    metrics: dict[str, float]
