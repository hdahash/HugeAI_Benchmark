"""Scorer protocol shared by all quality-scoring strategies.

Async because most real scorers (LLM judge, code execution) need to await
I/O; the trivial ones (heuristic, null) are just async functions that
return immediately.
"""
from __future__ import annotations

from typing import Protocol

from routerbench.models import DatasetItem


class Scorer(Protocol):
    async def score(self, item: DatasetItem, content: str) -> float | None:
        """Return a 0.0-1.0 quality score, or None if this item can't be scored."""
        ...


class NullScorer:
    async def score(self, item: DatasetItem, content: str) -> float | None:
        return None
