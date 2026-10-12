"""API-key authentication and per-client rate limiting.

Only SHA-256 digests of API keys are configured (FRAUDGUARD_API_KEY_HASHES),
so a leaked config or environment dump does not leak usable credentials.
With no keys configured the API fails closed and rejects every request.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from dataclasses import dataclass, field


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


@dataclass
class KeyStore:
    digests: tuple[str, ...]

    @classmethod
    def from_env_value(cls, value: str) -> KeyStore:
        digests = tuple(d.strip().lower() for d in value.split(",") if d.strip())
        for d in digests:
            if len(d) != 64 or any(c not in "0123456789abcdef" for c in d):
                raise ValueError("FRAUDGUARD_API_KEY_HASHES must hold hex SHA-256 digests")
        return cls(digests)

    def identify(self, api_key: str | None) -> str | None:
        """Return a short client id for a valid key, else None. Compares every digest."""
        if not api_key or not self.digests:
            return None
        candidate = hash_key(api_key)
        match = None
        for digest in self.digests:
            if hmac.compare_digest(candidate, digest):
                match = digest
        return match[:12] if match else None


@dataclass
class RateLimiter:
    """Token bucket per client: `rate_per_minute` sustained, bursts up to the same size."""

    rate_per_minute: int
    _buckets: dict[str, tuple[float, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow(self, client: str) -> tuple[bool, float]:
        capacity = float(self.rate_per_minute)
        refill = capacity / 60.0
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(client, (capacity, now))
            tokens = min(capacity, tokens + (now - last) * refill)
            if tokens >= 1.0:
                self._buckets[client] = (tokens - 1.0, now)
                return True, 0.0
            self._buckets[client] = (tokens, now)
            return False, (1.0 - tokens) / refill
