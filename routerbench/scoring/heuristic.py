"""Substring-match scoring for objective QA items with a known-good answer."""
from __future__ import annotations

from routerbench.models import DatasetItem


class HeuristicScorer:
    """Scores 1.0 if all expected substrings appear (case-insensitive), else 0.0.

    Items with no expected_answer_contains are left unscored (None) since
    there's nothing objective to check them against.
    """

    async def score(self, item: DatasetItem, content: str) -> float | None:
        if not item.expected_answer_contains:
            return None
        haystack = content.lower()
        hits = sum(1 for needle in item.expected_answer_contains if needle.lower() in haystack)
        return hits / len(item.expected_answer_contains)
