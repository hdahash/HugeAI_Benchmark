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
    # If the router caches responses server-side (many do, sometimes
    # unconditionally -- see hugeai's FORCE_CACHE_ENABLED, which overrides
    # whatever the client requests), repeated/identical prompts across
    # scenarios come back near-instantly from cache instead of hitting the
    # real backend, silently turning "throughput under load" into "cache-hit
    # latency". When true, RouterClient appends a fresh per-call nonce to
    # every prompt so no two requests -- across accuracy/load/reliability, or
    # across separate runs within the cache's TTL -- are ever identical.
    cache_bust: bool = False

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
    # If true (and baseline_model is set), also fires each prompt forced to
    # baseline_model and compares quality/cost against the routed response --
    # this is the only way to tell "cheap and good enough" apart from "cheap
    # and worse". Roughly doubles request volume for this scenario.
    compute_routing_regret: bool = False


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
class JudgeConfig:
    """LLM-as-judge scoring for open-ended items (no verifiable ground truth).

    Any field left None inherits the corresponding value from `router:` --
    by default the judge just calls the same endpoint with a different
    `model`, which lets the mock router act as its own judge for demo
    purposes. Point these at a separate, stronger model/endpoint for real use.
    """

    enabled: bool = False
    model: str = ""
    base_url: str | None = None
    chat_path: str | None = None
    api_key_env: str | None = None
    auth_header_name: str | None = None
    auth_header_format: str | None = None
    timeout_s: float = 30.0
    max_retries: int = 1
    response_content_path: str | None = None
    default_rubric: str = "Score how helpful, accurate, and relevant this response is to the prompt."
    rubric: dict[str, str] = field(default_factory=dict)

    def to_router_config(self, base: RouterConfig) -> RouterConfig:
        return RouterConfig(
            base_url=self.base_url or base.base_url,
            chat_path=self.chat_path or base.chat_path,
            method=base.method,
            api_key_env=self.api_key_env if self.api_key_env is not None else base.api_key_env,
            auth_header_name=self.auth_header_name or base.auth_header_name,
            auth_header_format=self.auth_header_format or base.auth_header_format,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            response_content_path=self.response_content_path or base.response_content_path,
        )


@dataclass
class CodeExecConfig:
    """Sandboxed local execution of model-generated code against test cases.

    SECURITY: this runs whatever code the router's model produced. It is
    isolated with `python -I` (ignores user env/site-packages), a stripped
    environment, and CPU/memory/wall-clock limits -- but it is a best-effort
    local sandbox, not a security boundary. Only enable this against
    ephemeral/disposable infrastructure, never on shared hosts or anywhere
    with reachable secrets or network access worth protecting.
    """

    enabled: bool = True
    timeout_s: float = 5.0
    memory_limit_mb: int = 256
    python_executable: str = "python3"


@dataclass
class ScorerConfig:
    mode: str = "auto"  # "auto" (dispatch per item) | "heuristic" | "none"


@dataclass
class OutputConfig:
    dir: str = "reports"
    formats: list[str] = field(default_factory=lambda: ["json", "html"])


@dataclass
class BenchmarkConfig:
    router: RouterConfig = field(default_factory=RouterConfig)
    pricing: dict[str, ModelPricing] = field(default_factory=dict)
    baseline_model: str | None = None
    # Caps aggregate request rate across ALL traffic this run generates
    # (accuracy + load + reliability + judge calls) -- the limit lives on the
    # service's account, not per scenario, so it's enforced globally via one
    # shared RateLimiter rather than per-scenario. None disables throttling
    # (fine for a local mock router; set this for a real, rate-limited service).
    rate_limit_rps: float | None = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    accuracy: AccuracyScenarioConfig = field(default_factory=AccuracyScenarioConfig)
    load: LoadScenarioConfig = field(default_factory=LoadScenarioConfig)
    reliability: ReliabilityScenarioConfig = field(default_factory=ReliabilityScenarioConfig)
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    code_exec: CodeExecConfig = field(default_factory=CodeExecConfig)
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
            rate_limit_rps=raw.get("rate_limit_rps"),
            dataset=DatasetConfig(**(raw.get("dataset") or {})),
            accuracy=AccuracyScenarioConfig(**(raw.get("accuracy") or {})),
            load=LoadScenarioConfig(**(raw.get("load") or {})),
            reliability=ReliabilityScenarioConfig(**(raw.get("reliability") or {})),
            scorer=ScorerConfig(**(raw.get("scorer") or {})),
            judge=JudgeConfig(**(raw.get("judge") or {})),
            code_exec=CodeExecConfig(**(raw.get("code_exec") or {})),
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
