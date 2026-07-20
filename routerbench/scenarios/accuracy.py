"""Routing-accuracy / quality / cost scenario: run the full dataset once at a
fixed, modest concurrency and score each response."""
from __future__ import annotations

from routerbench.config import AccuracyScenarioConfig
from routerbench.metrics.cost import cost_summary
from routerbench.metrics.routing import quality_summary, routing_accuracy
from routerbench.models import DatasetItem, ScenarioResult
from routerbench.scenarios.base import ExecutionContext, run_batch


async def run_accuracy_scenario(
    ctx: ExecutionContext,
    dataset: list[DatasetItem],
    config: AccuracyScenarioConfig,
    pricing,
    baseline_model: str | None,
) -> ScenarioResult:
    results = await run_batch(ctx, dataset, config.name, config.concurrency)

    metrics = {
        "routing": routing_accuracy(results),
        "quality": quality_summary(results),
        "cost": cost_summary(results, pricing, baseline_model),
    }
    return ScenarioResult(name=config.name, scenario_type="accuracy", results=results, metrics=metrics)
