"""Core data structures shared across the benchmark framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RequestStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class RouterResponse:
    """Normalized result of a single call to the router service under test."""

    request_id: str
    status: RequestStatus
    model_used: str | None = None
    content: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    ttfb_s: float | None = None
    total_latency_s: float = 0.0
    http_status: int | None = None
    error: str | None = None
    attempts: int = 1
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetItem:
    """A single benchmark case: a prompt plus the metadata needed to score it."""

    id: str
    prompt: str
    category: str = "general"
    complexity: str | None = None
    expected_model: str | None = None
    expected_answer_contains: list[str] = field(default_factory=list)
    # Substrings that must NOT appear in the response -- e.g. testing whether
    # an injected instruction can override a server-side output policy
    # (see data/hugeai_prompts.jsonl's "security" category), or whether a
    # PII value survives redaction into the response. Symmetric to
    # expected_answer_contains; either or both may be set.
    expected_answer_excludes: list[str] = field(default_factory=list)
    # Prior turns before `prompt`, e.g. [{"role": "user", "content": "..."},
    # {"role": "assistant", "content": "..."}]. Empty for a single-turn item
    # (the common case) -- `prompt` is always the final user turn sent.
    conversation: list[dict[str, str]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestResult:
    """One dataset item joined with the router's response, ready for scoring/metrics."""

    item: DatasetItem
    response: RouterResponse
    cost_usd: float | None = None
    quality_score: float | None = None
    routing_correct: bool | None = None
    scenario: str = ""
    concurrency_level: int | None = None


@dataclass
class ScenarioResult:
    name: str
    scenario_type: str
    results: list[RequestResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
