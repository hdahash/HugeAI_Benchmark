"""Reliability metrics: error rates, timeout behavior, retry effectiveness."""
from __future__ import annotations

from collections import defaultdict

from routerbench.models import RequestResult, RequestStatus


def reliability_summary(results: list[RequestResult]) -> dict:
    total = len(results)
    if total == 0:
        return {"total_requests": 0}

    by_status: dict[str, int] = defaultdict(int)
    for r in results:
        by_status[r.response.status.value] += 1

    retried = [r for r in results if r.response.attempts > 1]
    retried_then_succeeded = [r for r in retried if r.response.status == RequestStatus.SUCCESS]

    error_types: dict[str, int] = defaultdict(int)
    for r in results:
        if r.response.status != RequestStatus.SUCCESS and r.response.error:
            key = r.response.error.split(":")[0]
            error_types[key] += 1

    return {
        "total_requests": total,
        "success_count": by_status.get(RequestStatus.SUCCESS.value, 0),
        "error_count": by_status.get(RequestStatus.ERROR.value, 0),
        "timeout_count": by_status.get(RequestStatus.TIMEOUT.value, 0),
        "success_rate": by_status.get(RequestStatus.SUCCESS.value, 0) / total,
        "error_rate": 1 - (by_status.get(RequestStatus.SUCCESS.value, 0) / total),
        "retried_requests": len(retried),
        "retry_recovery_rate": (len(retried_then_succeeded) / len(retried)) if retried else None,
        "error_type_breakdown": dict(error_types),
    }
