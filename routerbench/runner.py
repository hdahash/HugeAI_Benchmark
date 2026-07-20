"""Top-level orchestrator: wires config -> client -> scenarios -> report."""
from __future__ import annotations

from routerbench.client import RouterClient
from routerbench.config import BenchmarkConfig
from routerbench.dataset import load_dataset
from routerbench.models import ScenarioResult
from routerbench.scenarios.accuracy import run_accuracy_scenario
from routerbench.scenarios.base import ExecutionContext
from routerbench.scenarios.load import run_load_scenario
from routerbench.scenarios.reliability import run_reliability_scenario
from routerbench.scorer import build_scorer


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config

    async def run(self) -> list[ScenarioResult]:
        dataset = load_dataset(self.config.dataset.path)
        scorer = build_scorer(self.config.scorer.mode)

        async with RouterClient(self.config.router) as client:
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
