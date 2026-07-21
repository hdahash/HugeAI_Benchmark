"""Cost computation and savings analysis."""
from __future__ import annotations

from routerbench.config import ModelPricing
from routerbench.models import RequestResult, RequestStatus


def resolve_pricing_key(model: str, pricing: dict[str, ModelPricing]) -> str | None:
    """Match a reported model name against the pricing table.

    Real APIs often echo a dated snapshot ("gpt-4o-2024-08-06") even when the
    pricing table is keyed by the bare family name ("gpt-4o"). Falls back to
    the longest pricing key that the reported model starts with, so a more
    specific key (e.g. "gpt-4o" over "gpt-4") wins when both would match.
    """
    if model in pricing:
        return model
    candidates = [key for key in pricing if model.startswith(key)]
    return max(candidates, key=len) if candidates else None


def request_cost_usd(model: str | None, input_tokens: int, output_tokens: int, pricing: dict[str, ModelPricing]) -> float | None:
    if model is None:
        return None
    key = resolve_pricing_key(model, pricing)
    if key is None:
        return None
    p = pricing[key]
    return (input_tokens / 1000) * p.input_per_1k + (output_tokens / 1000) * p.output_per_1k


def cost_summary(results: list[RequestResult], pricing: dict[str, ModelPricing], baseline_model: str | None = None) -> dict:
    priced = [r for r in results if r.cost_usd is not None]
    total_cost = sum(r.cost_usd for r in priced)
    by_model: dict[str, dict[str, float]] = {}
    for r in priced:
        model = r.response.model_used or "unknown"
        entry = by_model.setdefault(model, {"count": 0, "total_cost_usd": 0.0})
        entry["count"] += 1
        entry["total_cost_usd"] += r.cost_usd or 0.0

    summary = {
        "total_cost_usd": total_cost,
        "priced_requests": len(priced),
        "avg_cost_per_request_usd": (total_cost / len(priced)) if priced else 0.0,
        "by_model": by_model,
    }

    if baseline_model and baseline_model in pricing:
        successes = [r for r in results if r.response.status == RequestStatus.SUCCESS]
        baseline_total = sum(
            request_cost_usd(baseline_model, r.response.input_tokens, r.response.output_tokens, pricing) or 0.0
            for r in successes
        )
        summary["baseline_model"] = baseline_model
        summary["baseline_total_cost_usd"] = baseline_total
        summary["savings_usd"] = baseline_total - total_cost
        summary["savings_pct"] = ((baseline_total - total_cost) / baseline_total * 100) if baseline_total > 0 else 0.0

    return summary
