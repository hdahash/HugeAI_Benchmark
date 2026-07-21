"""Top-level orchestrator: wires config -> client -> scenarios -> report."""
from __future__ import annotations

from contextlib import AsyncExitStack

from routerbench.client import RouterClient
from routerbench.config import BenchmarkConfig
from routerbench.dataset import load_dataset
from routerbench.models import ScenarioResult
from routerbench.ratelimit import RateLimiter
from routerbench.scenarios.accuracy import run_accuracy_scenario
from routerbench.scenarios.base import ExecutionContext
from routerbench.scenarios.load import run_load_scenario
from routerbench.scenarios.reliability import run_reliability_scenario
from routerbench.scoring import build_scorer


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config

    async def run(self) -> list[ScenarioResult]:
        dataset = load_dataset(self.config.dataset.path)

        # Shared across the router client AND the judge client -- the limit
        # lives on the target account, so judge traffic against the same
        # service (the common case: judge.model set, no separate base_url)
        # has to count against the same budget, not get a budget of its own.
        rate_limiter = RateLimiter(self.config.rate_limit_rps)

        async with AsyncExitStack() as stack:
            client = await stack.enter_async_context(RouterClient(self.config.router, rate_limiter=rate_limiter))

            judge_client = None
            if self.config.judge.enabled:
                judge_router_config = self.config.judge.to_router_config(self.config.router)
                judge_client = await stack.enter_async_context(
                    RouterClient(judge_router_config, rate_limiter=rate_limiter)
                )

            scorer = build_scorer(self.config, judge_client)
            ctx = ExecutionContext(client=client, pricing=self.config.pricing, scorer=scorer)
            scenario_results: list[ScenarioResult] = []

            if self.config.accuracy.enabled:
                scenario_results.append(
                    await run_accuracy_scenario(
                        ctx, dataset, self.config.accuracy, self.config.pricing, self.config.baseline_model
                    )
                )
            if self.config.load.enabled:
                scenario_results.append(await run_load_scenario(ctx, dataset, self.config.load))
            if self.config.reliability.enabled:
                scenario_results.append(await run_reliability_scenario(ctx, dataset, self.config.reliability))

        return scenario_results
