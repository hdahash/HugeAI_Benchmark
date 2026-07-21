"""A minimal global rate limiter for benchmarking real, rate-limited services.

Enforces a uniform minimum spacing between successive requests -- simpler
than a token bucket, but sufficient to stay under a fixed-window cap (e.g.
"120 requests per 60 seconds") without bursting past it. One instance is
meant to be shared across every client hitting the same account (router
traffic and judge traffic both count against the same limit), since the
limit lives on the service's side, not per client object.
"""
from __future__ import annotations

import asyncio
import time


class RateLimiter:
    def __init__(self, max_requests_per_second: float | None):
        self.min_interval_s = (1.0 / max_requests_per_second) if max_requests_per_second and max_requests_per_second > 0 else None
        self._lock = asyncio.Lock()
        self._next_allowed_at = 0.0

    async def acquire(self) -> None:
        if self.min_interval_s is None:
            return
        async with self._lock:
            now = time.monotonic()
            wait_s = self._next_allowed_at - now
            if wait_s > 0:
                await asyncio.sleep(wait_s)
                now = time.monotonic()
            self._next_allowed_at = now + self.min_interval_s
