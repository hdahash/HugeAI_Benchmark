"""Load/throughput scenario: sweep concurrency levels and measure latency
percentiles + real throughput at each level."""
from __future__ import annotations

import time

from routerbench.config import LoadScenarioConfig
from routerbench.metrics.latency import batch_stats
from routerbench.models import DatasetItem, RequestResult, ScenarioResult
from routerbench.scenarios.base import ExecutionContext, run_batch


async def run_load_scenario(
    ctx: ExecutionContext,
    dataset: list[DatasetItem],
    config: LoadScenarioConfig,
) -> ScenarioResult:
    if not dataset:
        raise ValueError("Load scenario requires a non-empty dataset to draw prompts from")

    all_results: list[RequestResult] = []
    by_level_metrics: dict[int, dict] = {}

    for level in config.concurrency_levels:
        items = [dataset[i % len(dataset)] for i in range(config.requests_per_level)]
        start = time.perf_counter()
        results = await run_batch(ctx, items, config.name, level, concurrency_level=level, score_quality=False)
        wall_time_s = time.perf_counter() - start

        all_results.extend(results)
        by_level_metrics[level] = batch_stats(results, wall_time_s)

    metrics = {"by_concurrency": by_level_metrics}
    return ScenarioResult(name=config.name, scenario_type="load", results=all_results, metrics=metrics)
