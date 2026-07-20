"""Routing regret: how much quality did the router give up (if any) by not
always calling the baseline (typically largest/most expensive) model,
weighed against how much it saved by not doing so.

This is the metric that actually answers "is this a good router" -- routing
accuracy and cost alone can't distinguish "cheap and just as good" from
"cheap and worse".
"""
from __future__ import annotations

from routerbench.models import RequestResult, RequestStatus


def routing_regret_summary(routed: list[RequestResult], baseline: list[RequestResult], top_n: int = 5) -> dict:
    baseline_by_id = {r.item.id: r for r in baseline}

    deltas = []
    for r in routed:
        b = baseline_by_id.get(r.item.id)
        if (
            b is None
            or r.response.status != RequestStatus.SUCCESS
            or b.response.status != RequestStatus.SUCCESS
            or r.quality_score is None
            or b.quality_score is None
        ):
            continue
        deltas.append(
            {
                "id": r.item.id,
                "category": r.item.category,
                "routed_model": r.response.model_used,
                "baseline_model": b.response.model_used,
                "routed_quality": r.quality_score,
                "baseline_quality": b.quality_score,
                # positive = router lost quality vs. baseline; negative/zero = router matched or beat it
                "quality_regret": b.quality_score - r.quality_score,
                # positive = router was cheaper than baseline
                "cost_savings_usd": (b.cost_usd or 0.0) - (r.cost_usd or 0.0),
            }
        )

    if not deltas:
        return {"comparable_requests": 0}

    avg_regret = sum(d["quality_regret"] for d in deltas) / len(deltas)
    total_savings = sum(d["cost_savings_usd"] for d in deltas)
    no_quality_loss = sum(1 for d in deltas if d["quality_regret"] <= 0)

    worst = sorted(deltas, key=lambda d: d["quality_regret"], reverse=True)[:top_n]

    return {
        "comparable_requests": len(deltas),
        "avg_quality_regret": avg_regret,
        "pct_no_quality_loss": no_quality_loss / len(deltas),
        "total_cost_savings_usd": total_savings,
        "avg_cost_savings_usd": total_savings / len(deltas),
        "worst_regret_items": worst,
    }
