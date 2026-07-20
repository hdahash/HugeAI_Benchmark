"""LLM-as-judge scoring for open-ended items with no verifiable ground truth
(reasoning, creative writing, long-context summarization, etc).

Reuses RouterClient rather than a bespoke HTTP client -- the judge is just
another OpenAI-compatible chat endpoint, possibly the same router under
test with a different `model` forced.
"""
from __future__ import annotations

import re

from routerbench.client import RouterClient
from routerbench.config import JudgeConfig
from routerbench.models import DatasetItem, RequestStatus

_SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")

_JUDGE_PROMPT_TEMPLATE = """You are grading an AI assistant's response for quality.

Task category: {category}
Grading rubric: {rubric}

User's prompt:
\"\"\"
{prompt}
\"\"\"

Assistant's response:
\"\"\"
{content}
\"\"\"

Score the response from 0 to 10 based on the rubric above. Reply with ONLY the integer score and nothing else."""


class LLMJudgeScorer:
    def __init__(self, client: RouterClient, config: JudgeConfig):
        self.client = client
        self.config = config

    async def score(self, item: DatasetItem, content: str) -> float | None:
        if not content.strip():
            return 0.0

        rubric = self.config.rubric.get(item.category, self.config.default_rubric)
        judge_prompt = _JUDGE_PROMPT_TEMPLATE.format(
            category=item.category, rubric=rubric, prompt=item.prompt, content=content
        )

        response = await self.client.send(judge_prompt, force_model=self.config.model or None)
        if response.status != RequestStatus.SUCCESS:
            return None

        match = _SCORE_RE.search(response.content)
        if not match:
            return None

        raw_score = float(match.group())
        return max(0.0, min(raw_score, 10.0)) / 10.0
