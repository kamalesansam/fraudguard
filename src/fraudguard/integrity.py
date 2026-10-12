"""Model artifact integrity: safe serialization, hashing and verified loading.

`pickle` and `joblib` run arbitrary code while loading (CWE-502), so a
tampered model file is remote code execution on the inference server. This
module uses skops instead, which reconstructs objects from a declarative
schema, and adds two further checks before anything is deserialized:

1. The file's SHA-256 must match the digest recorded in the signed manifest.
2. Every type skops reports as "untrusted" must be on a reviewed allowlist.
   Anything else (e.g. `builtins.eval`, `os.system`) aborts the load.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import skops.io as sio

# Reviewed list of types the HistGradientBoosting pipeline needs beyond skops'
# built-in trusted set (found with `skops.io.get_untrusted_types`).
# Adding a type here is a security decision: review it.
ALLOWED_SKOPS_TYPES: frozenset[str] = frozenset(
    {"sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"}
)


class IntegrityError(RuntimeError):
    """Raised when a model artifact fails a hash or type-allowlist check."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_types(path: Path) -> list[str]:
    untrusted = sio.get_untrusted_types(file=path)
    disallowed = sorted(set(untrusted) - ALLOWED_SKOPS_TYPES)
    if disallowed:
        raise IntegrityError(f"{path.name} contains types outside the allowlist: {disallowed}")
    return untrusted


def canonicalize(blob: bytes) -> bytes:
    """Make a skops archive byte-for-byte reproducible.

    skops names array files and node `__id__`s after `id()` of the in-memory
    objects and stamps each zip entry with the wall clock, so the same model
    hashes differently on every run. Renumbering ids and arrays by first use in
    the schema and pinning timestamps gives a stable digest: retraining the same
    commit on the same data and platform reproduces the SHA-256 in the manifest.
    """
    src = zipfile.ZipFile(io.BytesIO(blob))
    schema = json.loads(src.read("schema.json"))
    renames: dict[str, str] = {}
    ids: dict[int, int] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "file" and isinstance(value, str) and value in src.NameToInfo:
                    node[key] = renames.setdefault(value, f"{len(renames):06d}{Path(value).suffix}")
                elif key == "__id__" and isinstance(value, int):
                    node[key] = ids.setdefault(value, len(ids))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as dst:
        entries = [("schema.json", json.dumps(schema, sort_keys=True).encode())]
        entries += sorted((new, src.read(old)) for old, new in renames.items())
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            dst.writestr(info, data)
    return out.getvalue()


def save_model(model: Any, path: Path) -> str:
    """Serialize with skops, refuse to keep the file if it needs unreviewed types."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonicalize(sio.dumps(model)))
    try:
        _check_types(path)
    except IntegrityError:
        path.unlink(missing_ok=True)
        raise
    return sha256_file(path)


def load_verified(path: Path, expected_sha256: str) -> Any:
    """Load a model only if its digest matches and its types are allowlisted."""
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected_sha256):
        raise IntegrityError(
            f"{path.name} sha256 mismatch: expected {expected_sha256}, got {actual}"
        )
    untrusted = _check_types(path)
    return sio.load(path, trusted=untrusted)
