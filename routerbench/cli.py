"""Command-line entrypoint: routerbench run --config configs/default.yaml"""
from __future__ import annotations

import argparse
import asyncio
import sys

from rich.console import Console
from rich.table import Table

from routerbench.config import BenchmarkConfig
from routerbench.report import save_report
from routerbench.runner import BenchmarkRunner

console = Console()


def _print_summary(scenario_results) -> None:
    for scenario in scenario_results:
        console.print(f"\n[bold]{scenario.name}[/bold] ([italic]{scenario.scenario_type}[/italic])")

        if scenario.scenario_type == "accuracy":
            routing = scenario.metrics["routing"]
            quality = scenario.metrics["quality"]
            cost = scenario.metrics["cost"]
            table = Table(show_header=True)
            table.add_column("Metric")
            table.add_column("Value")
            acc = routing["accuracy"]
            table.add_row("Routing accuracy", f"{acc * 100:.1f}%" if acc is not None else "n/a")
            table.add_row("Avg quality score", f"{quality['avg_quality_score']:.2f}" if quality["avg_quality_score"] is not None else "n/a")
            table.add_row("Total cost (USD)", f"${cost['total_cost_usd']:.4f}")
            if "savings_pct" in cost:
                table.add_row("Savings vs baseline", f"{cost['savings_pct']:.1f}%")
            console.print(table)

            regret = scenario.metrics.get("routing_regret")
            if regret and regret.get("comparable_requests"):
                regret_table = Table(show_header=True, title="Routing regret vs. baseline")
                regret_table.add_column("Metric")
                regret_table.add_column("Value")
                regret_table.add_row("Comparable requests", str(regret["comparable_requests"]))
                regret_table.add_row("Avg quality regret", f"{regret['avg_quality_regret']:+.3f}")
                regret_table.add_row("Requests with no quality loss", f"{regret['pct_no_quality_loss'] * 100:.1f}%")
                regret_table.add_row("Avg cost savings/request", f"${regret['avg_cost_savings_usd']:.5f}")
                regret_table.add_row("Total cost savings", f"${regret['total_cost_savings_usd']:.4f}")
                console.print(regret_table)

        elif scenario.scenario_type == "load":
            table = Table(show_header=True)
            table.add_column("Concurrency")
            table.add_column("Throughput (req/s)")
            table.add_column("p50 (ms)")
            table.add_column("p95 (ms)")
            table.add_column("Error rate")
            for level, stats in scenario.metrics["by_concurrency"].items():
                table.add_row(
                    str(level),
                    f"{stats['throughput_rps']:.2f}",
                    f"{stats['p50_ms']:.0f}",
                    f"{stats['p95_ms']:.0f}",
                    f"{stats['error_rate'] * 100:.1f}%",
                )
            console.print(table)

        elif scenario.scenario_type == "reliability":
            table = Table(show_header=True)
            table.add_column("Segment")
            table.add_column("Success rate")
            table.add_column("Errors")
            table.add_column("Timeouts")
            table.add_column("Retried")
            table.add_column("Retry recovery")
            for seg, stats in scenario.metrics.items():
                recovery = stats["retry_recovery_rate"]
                table.add_row(
                    seg,
                    f"{stats['success_rate'] * 100:.1f}%",
                    str(stats["error_count"]),
                    str(stats["timeout_count"]),
                    str(stats["retried_requests"]),
                    f"{recovery * 100:.0f}%" if recovery is not None else "n/a",
                )
            console.print(table)


def cmd_run(args: argparse.Namespace) -> int:
    config = BenchmarkConfig.load(args.config)
    console.print(f"[bold cyan]Running benchmark against {config.router.base_url}[/bold cyan]")

    runner = BenchmarkRunner(config)
    scenario_results = asyncio.run(runner.run())

    _print_summary(scenario_results)

    paths = save_report(scenario_results, config.output.dir, config.output.formats)
    console.print("\n[bold green]Reports written:[/bold green]")
    for p in paths:
        console.print(f"  {p}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="routerbench", description="Benchmark an LLM router service.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run the benchmark suite defined in a config file.")
    run_parser.add_argument("--config", "-c", default="configs/default.yaml", help="Path to YAML config.")
    run_parser.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
