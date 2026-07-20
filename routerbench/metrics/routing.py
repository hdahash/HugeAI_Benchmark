"""Routing accuracy and quality metrics."""
from __future__ import annotations

from collections import defaultdict

from routerbench.models import RequestResult, RequestStatus


def routing_accuracy(results: list[RequestResult]) -> dict:
    """Top-1 routing accuracy against DatasetItem.expected_model, plus a confusion matrix
    and per-category breakdown. Items with no expected_model are excluded from accuracy
    but still appear in the model distribution.
    """
    checkable = [r for r in results if r.item.expected_model and r.response.status == RequestStatus.SUCCESS]
    correct = sum(1 for r in checkable if r.routing_correct)

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in checkable:
        actual = r.response.model_used or "unknown"
        confusion[r.item.expected_model][actual] += 1

    by_category: dict[str, dict] = {}
    for r in checkable:
        cat = by_category.setdefault(r.item.category, {"total": 0, "correct": 0})
        cat["total"] += 1
        cat["correct"] += 1 if r.routing_correct else 0
    for cat, stats in by_category.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0

    distribution: dict[str, int] = defaultdict(int)
    for r in results:
        if r.response.status == RequestStatus.SUCCESS:
            distribution[r.response.model_used or "unknown"] += 1

    return {
        "checkable_requests": len(checkable),
        "correct": correct,
        "accuracy": (correct / len(checkable)) if checkable else None,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "by_category": by_category,
        "model_distribution": dict(distribution),
    }


def quality_summary(results: list[RequestResult]) -> dict:
    scored = [r for r in results if r.quality_score is not None]
    by_category: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        by_category[r.item.category].append(r.quality_score)

    return {
        "scored_requests": len(scored),
        "avg_quality_score": (sum(r.quality_score for r in scored) / len(scored)) if scored else None,
        "by_category": {
            cat: sum(scores) / len(scores) for cat, scores in by_category.items()
        },
    }
