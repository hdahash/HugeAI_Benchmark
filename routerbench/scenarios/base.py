"""Shared scenario execution primitives."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from routerbench.client import RouterClient
from routerbench.config import ModelPricing
from routerbench.metrics.cost import request_cost_usd
from routerbench.models import DatasetItem, RequestResult, RequestStatus
from routerbench.scoring.base import Scorer


@dataclass
class ExecutionContext:
    client: RouterClient
    pricing: dict[str, ModelPricing]
    scorer: Scorer


async def run_one(
    ctx: ExecutionContext,
    item: DatasetItem,
    scenario_name: str,
    semaphore: asyncio.Semaphore,
    force_model: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout_s: float | None = None,
    concurrency_level: int | None = None,
    score_quality: bool = True,
) -> RequestResult:
    async with semaphore:
        response = await ctx.client.send(
            item.prompt,
            request_id=item.id,
            force_model=force_model,
            extra_headers=extra_headers,
            timeout_s=timeout_s,
        )

    result = RequestResult(item=item, response=response, scenario=scenario_name, concurrency_level=concurrency_level)

    if response.status == RequestStatus.SUCCESS:
        result.cost_usd = request_cost_usd(response.model_used, response.input_tokens, response.output_tokens, ctx.pricing)
        # Quality scoring (especially an LLM judge or code execution) is only
        # worth its cost/latency where correctness is actually in question --
        # load and reliability scenarios fire the same prompts repeatedly
        # just to measure timing/error behavior, not to re-grade them.
        if score_quality:
            result.quality_score = await ctx.scorer.score(item, response.content)
        if item.expected_model is not None:
            result.routing_correct = response.model_used == item.expected_model

    return result


async def run_batch(
    ctx: ExecutionContext,
    items: list[DatasetItem],
    scenario_name: str,
    concurrency: int,
    **kwargs,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [run_one(ctx, item, scenario_name, semaphore, **kwargs) for item in items]
    return await asyncio.gather(*tasks)
