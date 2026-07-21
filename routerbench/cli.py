"""Command-line entrypoint: routerbench run --config configs/default.yaml"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from routerbench.compare import aggregate_comparison, run_all, save_comparison
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


def _labels_for(configs: list[str], labels: list[str] | None) -> list[str]:
    if labels:
        if len(labels) != len(configs):
            raise SystemExit(f"--labels must have the same count as --configs ({len(configs)})")
        return labels
    return [Path(c).stem for c in configs]


def _warn_if_datasets_differ(configs: dict[str, BenchmarkConfig]) -> None:
    paths = {label: cfg.dataset.path for label, cfg in configs.items()}
    if len(set(paths.values())) > 1:
        console.print("[bold yellow]Warning:[/bold yellow] providers are using different datasets -- this comparison won't be apples-to-apples:")
        for label, path in paths.items():
            console.print(f"  {label}: {path}")


def _print_comparison(comparison: dict) -> None:
    providers = comparison["providers"]

    table = Table(title="Quality & cost", show_header=True)
    table.add_column("Metric")
    for p in providers:
        table.add_column(p)
    table.add_row(
        "Avg quality score",
        *[f"{comparison['quality'][p]:.2f}" if comparison["quality"].get(p) is not None else "n/a" for p in providers],
    )
    table.add_row(
        "Routing accuracy",
        *[
            f"{comparison['routing'][p]['accuracy'] * 100:.1f}%"
            if comparison["routing"].get(p) and comparison["routing"][p].get("accuracy") is not None
            else "n/a"
            for p in providers
        ],
    )
    table.add_row(
        "Total cost (USD)",
        *[
            f"${comparison['cost'][p]['total_usd']:.4f}" if comparison["cost"].get(p) and comparison["cost"][p].get("total_usd") is not None else "n/a"
            for p in providers
        ],
    )
    console.print(table)

    rel_table = Table(title="Reliability", show_header=True)
    rel_table.add_column("Metric")
    for p in providers:
        rel_table.add_column(p)
    rel_table.add_row(
        "Success rate",
        *[
            f"{comparison['reliability'][p]['success_rate'] * 100:.1f}%"
            if comparison["reliability"].get(p) and comparison["reliability"][p].get("success_rate") is not None
            else "n/a"
            for p in providers
        ],
    )
    console.print(rel_table)

    for level in comparison["concurrency_levels"]:
        lat_table = Table(title=f"Latency & throughput @ concurrency {level}", show_header=True)
        lat_table.add_column("Metric")
        for p in providers:
            lat_table.add_column(p)
        stats_by_provider = {p: comparison["latency"].get(p, {}).get(level) for p in providers}
        lat_table.add_row("Throughput (req/s)", *[f"{s['throughput_rps']:.2f}" if s else "n/a" for s in stats_by_provider.values()])
        lat_table.add_row("p50 (ms)", *[f"{s['p50_ms']:.0f}" if s else "n/a" for s in stats_by_provider.values()])
        lat_table.add_row("p95 (ms)", *[f"{s['p95_ms']:.0f}" if s else "n/a" for s in stats_by_provider.values()])
        lat_table.add_row("Error rate", *[f"{s['error_rate'] * 100:.1f}%" if s else "n/a" for s in stats_by_provider.values()])
        console.print(lat_table)


def cmd_compare_run(args: argparse.Namespace) -> int:
    labels = _labels_for(args.configs, args.labels)
    configs = {label: BenchmarkConfig.load(path) for label, path in zip(labels, args.configs)}

    _warn_if_datasets_differ(configs)

    for label, config in configs.items():
        console.print(f"[bold cyan]Running {label} against {config.router.base_url}[/bold cyan]")

    reports = asyncio.run(run_all(configs))
    comparison = aggregate_comparison(reports)

    _print_comparison(comparison)

    output_dir = args.output or next(iter(configs.values())).output.dir
    paths = save_comparison(comparison, output_dir)
    console.print("\n[bold green]Comparison written:[/bold green]")
    for p in paths:
        console.print(f"  {p}")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    labels = _labels_for(args.reports, args.labels)
    reports = {label: json.loads(Path(path).read_text()) for label, path in zip(labels, args.reports)}

    comparison = aggregate_comparison(reports)
    _print_comparison(comparison)

    paths = save_comparison(comparison, args.output)
    console.print("\n[bold green]Comparison written:[/bold green]")
    for p in paths:
        console.print(f"  {p}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="routerbench", description="Benchmark an LLM router service.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run the benchmark suite defined in a config file.")
    run_parser.add_argument("--config", "-c", default="configs/default.yaml", help="Path to YAML config.")
    run_parser.set_defaults(func=cmd_run)

    compare_run_parser = sub.add_parser(
        "compare-run", help="Run the benchmark suite against multiple provider configs and compare them."
    )
    compare_run_parser.add_argument("--configs", nargs="+", required=True, help="Paths to each provider's YAML config.")
    compare_run_parser.add_argument(
        "--labels", nargs="+", default=None, help="Display name per provider (default: config filename stem)."
    )
    compare_run_parser.add_argument(
        "--output", default=None, help="Output dir for the comparison report (default: first config's output.dir)."
    )
    compare_run_parser.set_defaults(func=cmd_compare_run)

    compare_parser = sub.add_parser(
        "compare", help="Compare previously-saved report JSON files (from `routerbench run`) without re-running anything."
    )
    compare_parser.add_argument("--reports", nargs="+", required=True, help="Paths to each provider's report-*.json file.")
    compare_parser.add_argument(
        "--labels", nargs="+", default=None, help="Display name per provider (default: report filename stem)."
    )
    compare_parser.add_argument("--output", default="reports", help="Output dir for the comparison report.")
    compare_parser.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
