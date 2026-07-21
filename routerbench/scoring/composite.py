"""Dispatches each item to whichever scoring strategy actually applies to it.

Precedence: verifiable checks before subjective ones -- an LLM judge should
never grade something we can check deterministically.
  1. test_cases in metadata -> CodeExecutionScorer (ground truth)
  2. expected_answer_contains and/or expected_answer_excludes set -> HeuristicScorer (ground truth)
  3. otherwise, if a judge is configured -> LLMJudgeScorer (best-effort proxy)
  4. otherwise -> unscored (None)
"""
from __future__ import annotations

from routerbench.models import DatasetItem
from routerbench.scoring.base import Scorer


class CompositeScorer:
    def __init__(
        self,
        heuristic: Scorer,
        code_exec: Scorer | None = None,
        judge: Scorer | None = None,
    ):
        self.heuristic = heuristic
        self.code_exec = code_exec
        self.judge = judge

    async def score(self, item: DatasetItem, content: str) -> float | None:
        if self.code_exec is not None and item.metadata.get("test_cases"):
            return await self.code_exec.score(item, content)
        if item.expected_answer_contains or item.expected_answer_excludes:
            return await self.heuristic.score(item, content)
        if self.judge is not None:
            return await self.judge.score(item, content)
        return None
