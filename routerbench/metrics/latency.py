"""Latency and throughput statistics."""
from __future__ import annotations

import math
from typing import Any

from routerbench.models import RequestResult, RequestStatus


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (pct / 100)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def latency_stats(latencies_s: list[float]) -> dict[str, float]:
    if not latencies_s:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0, "min_ms": 0.0}
    values = sorted(v * 1000 for v in latencies_s)
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def batch_stats(group: list[RequestResult], wall_time_s: float) -> dict[str, Any]:
    """Latency + throughput for one batch of requests, given its measured wall-clock time.

    wall_time_s must be measured by the caller (start-to-finish of the whole
    concurrent batch) -- it cannot be reconstructed from individual request
    latencies once requests overlap.
    """
    successes = [r for r in group if r.response.status == RequestStatus.SUCCESS]
    latencies = [r.response.total_latency_s for r in successes]
    stats = latency_stats(latencies)
    return {
        **stats,
        "success_count": len(successes),
        "total_requests": len(group),
        "error_rate": 1 - (len(successes) / len(group)) if group else 0.0,
        "wall_time_s": wall_time_s,
        "throughput_rps": (len(successes) / wall_time_s) if wall_time_s > 0 else 0.0,
    }
