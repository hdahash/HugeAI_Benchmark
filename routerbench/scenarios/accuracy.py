"""Routing-accuracy / quality / cost scenario: run the full dataset once at a
fixed, modest concurrency and score each response.

If `config.compute_routing_regret` is set (and a baseline_model is
configured), also fires each prompt forced to the baseline model so quality
and cost can be compared against what the router actually chose -- see
metrics/regret.py for why this matters beyond routing accuracy alone.
"""
from __future__ import annotations

import asyncio

from routerbench.config import AccuracyScenarioConfig
from routerbench.metrics.cost import cost_summary
from routerbench.metrics.regret import routing_regret_summary
from routerbench.metrics.routing import quality_summary, routing_accuracy
from routerbench.models import DatasetItem, ScenarioResult
from routerbench.scenarios.base import ExecutionContext, run_one


async def run_accuracy_scenario(
    ctx: ExecutionContext,
    dataset: list[DatasetItem],
    config: AccuracyScenarioConfig,
    pricing,
    baseline_model: str | None,
) -> ScenarioResult:
    semaphore = asyncio.Semaphore(config.concurrency)
    routed_tasks = [run_one(ctx, item, config.name, semaphore) for item in dataset]

    compute_regret = config.compute_routing_regret and baseline_model is not None
    baseline_tasks = (
        [
            run_one(ctx, item, f"{config.name}_baseline", semaphore, force_model=baseline_model)
            for item in dataset
        ]
        if compute_regret
        else []
    )

    all_results = await asyncio.gather(*routed_tasks, *baseline_tasks)
    results = list(all_results[: len(routed_tasks)])
    baseline_results = list(all_results[len(routed_tasks):])

    metrics = {
        "routing": routing_accuracy(results),
        "quality": quality_summary(results),
        "cost": cost_summary(results, pricing, baseline_model),
    }
    if compute_regret:
        metrics["routing_regret"] = routing_regret_summary(results, baseline_results)

    # Baseline calls are included in the raw results for transparency/debugging,
    # but deliberately excluded from the metrics above -- they're synthetic
    # traffic for comparison, not router traffic being measured.
    return ScenarioResult(
        name=config.name, scenario_type="accuracy", results=results + baseline_results, metrics=metrics
    )
