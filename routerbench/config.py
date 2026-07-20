"""Configuration schema and YAML loading for routerbench.

Kept as plain dataclasses (no pydantic dependency) so the framework has a
minimal footprint. Every field extracting data from the router's JSON
response is a dot-path (e.g. "choices.0.message.content") so the framework
adapts to whatever response shape the target router actually returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RouterConfig:
    base_url: str = "http://localhost:8000"
    chat_path: str = "/v1/chat/completions"
    method: str = "POST"
    api_key_env: str | None = None
    auth_header_name: str = "Authorization"
    auth_header_format: str = "Bearer {api_key}"
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 30.0
    max_retries: int = 2
    retry_backoff_s: float = 0.5
    # Dot-paths into the request/response JSON. Defaults match the
    # OpenAI-compatible chat/completions contract most LLM routers speak.
    request_model_field: str | None = None  # set to force a model, e.g. for baseline runs
    request_extra_fields: dict[str, Any] = field(default_factory=dict)
    response_model_path: str = "model"
    response_content_path: str = "choices.0.message.content"
    response_input_tokens_path: str = "usage.prompt_tokens"
    response_output_tokens_path: str = "usage.completion_tokens"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RouterConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelPricing:
    input_per_1k: float = 0.0
    output_per_1k: float = 0.0


@dataclass
class DatasetConfig:
    path: str = "data/sample_prompts.jsonl"


@dataclass
class AccuracyScenarioConfig:
    enabled: bool = True
    name: str = "routing_accuracy"
    concurrency: int = 5


@dataclass
class LoadScenarioConfig:
    enabled: bool = True
    name: str = "load_test"
    concurrency_levels: list[int] = field(default_factory=lambda: [1, 5, 20])
    requests_per_level: int = 20


@dataclass
class ReliabilityScenarioConfig:
    enabled: bool = True
    name: str = "reliability"
    concurrency: int = 10
    requests: int = 40
    inject_faults: bool = True
    fault_header: str = "X-Inject-Fault"
    request_timeout_s: float = 5.0


@dataclass
class ScorerConfig:
    mode: str = "heuristic"  # "heuristic" | "none"


@dataclass
class OutputConfig:
    dir: str = "reports"
    formats: list[str] = field(default_factory=lambda: ["json", "html"])


@dataclass
class BenchmarkConfig:
    router: RouterConfig = field(default_factory=RouterConfig)
    pricing: dict[str, ModelPricing] = field(default_factory=dict)
    baseline_model: str | None = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    accuracy: AccuracyScenarioConfig = field(default_factory=AccuracyScenarioConfig)
    load: LoadScenarioConfig = field(default_factory=LoadScenarioConfig)
    reliability: ReliabilityScenarioConfig = field(default_factory=ReliabilityScenarioConfig)
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BenchmarkConfig":
        pricing = {
            name: ModelPricing(**vals)
            for name, vals in (raw.get("pricing") or {}).items()
        }
        return cls(
            router=RouterConfig.from_dict(raw.get("router") or {}),
            pricing=pricing,
            baseline_model=raw.get("baseline_model"),
            dataset=DatasetConfig(**(raw.get("dataset") or {})),
            accuracy=AccuracyScenarioConfig(**(raw.get("accuracy") or {})),
            load=LoadScenarioConfig(**(raw.get("load") or {})),
            reliability=ReliabilityScenarioConfig(**(raw.get("reliability") or {})),
            scorer=ScorerConfig(**(raw.get("scorer") or {})),
            output=OutputConfig(**(raw.get("output") or {})),
        )


def get_by_path(obj: Any, dot_path: str) -> Any:
    """Resolve a dot-path like 'choices.0.message.content' against nested dicts/lists."""
    current = obj
    for part in dot_path.split("."):
        if current is None:
            return None
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current
