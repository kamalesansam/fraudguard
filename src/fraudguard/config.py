"""Paths and parameter loading shared by every pipeline stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path.cwd()
PARAMS_FILE = ROOT / "params.yaml"

DATA_RAW = ROOT / "data" / "raw" / "transactions.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "artifacts" / "model"
REPORTS = ROOT / "reports"


def load_params(path: Path = PARAMS_FILE) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh)
