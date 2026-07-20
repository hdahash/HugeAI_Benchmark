"""Pluggable quality scoring for router responses.

The default HeuristicScorer only checks for required substrings, which is
enough to catch outright routing/model failures on objective QA items
without needing an external judge model or API key. Swap in a real
LLM-as-judge by implementing the Scorer protocol and wiring it into
scenarios/accuracy.py.
"""
from __future__ import annotations

from typing import Protocol

from routerbench.models import DatasetItem


class Scorer(Protocol):
    def score(self, item: DatasetItem, content: str) -> float | None:
        """Return a 0.0-1.0 quality score, or None if this item can't be scored."""
        ...


class HeuristicScorer:
    """Scores 1.0 if all expected substrings appear (case-insensitive), else 0.0.

    Items with no expected_answer_contains are left unscored (None) since
    there's nothing objective to check them against.
    """

    def score(self, item: DatasetItem, content: str) -> float | None:
        if not item.expected_answer_contains:
            return None
        haystack = content.lower()
        hits = sum(1 for needle in item.expected_answer_contains if needle.lower() in haystack)
        return hits / len(item.expected_answer_contains)


class NullScorer:
    def score(self, item: DatasetItem, content: str) -> float | None:
        return None


def build_scorer(mode: str) -> Scorer:
    if mode == "heuristic":
        return HeuristicScorer()
    if mode == "none":
        return NullScorer()
    raise ValueError(f"Unknown scorer mode: {mode}")
