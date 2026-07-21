"""Cross-provider comparison: runs (or loads) multiple providers' reports and
aggregates them into one side-by-side view -- the point of routerbench being
provider-agnostic is exactly this: benchmark hugeai and any number of
competing router services against the identical dataset and see how they
actually compare, rather than trusting each vendor's own marketing numbers.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Template

from routerbench.config import BenchmarkConfig
from routerbench.report import to_serializable
from routerbench.runner import BenchmarkRunner

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>LLM Router Comparison</title>
<style>
  body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 2rem; color: #1a1a1a; background: #fff; }
  @media (prefers-color-scheme: dark) { body { color: #e8e8e8; background: #14161a; } table { background: #1d2026; } th { background: #262a32; } td, th { border-color: #333; } }
  h1 { font-size: 1.6rem; }
  h2 { margin-top: 2.5rem; border-bottom: 2px solid #ccc; padding-bottom: .3rem; }
  h3 { margin-top: 1.5rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
  th, td { border: 1px solid #ccc; padding: 0.4rem 0.7rem; text-align: left; }
  th { background: #f0f0f0; }
  .muted { opacity: 0.6; font-size: 0.85rem; }
</style>
</head>
<body>
<h1>LLM Router Comparison</h1>
<p class="muted">Generated {{ generated_at }} &mdash; comparing: {{ providers|join(", ") }}</p>

<h2>Quality &amp; cost</h2>
<table>
  <tr><th>Metric</th>{% for p in providers %}<th>{{ p }}</th>{% endfor %}</tr>
  <tr><td>Avg quality score</td>{% for p in providers %}<td>{{ "%.2f"|format(quality[p]) if quality.get(p) is not none else "n/a" }}</td>{% endfor %}</tr>
  <tr><td>Routing accuracy</td>{% for p in providers %}<td>{{ ("%.1f"|format(routing[p].accuracy * 100)) + "%" if routing.get(p) and routing[p].accuracy is not none else "n/a" }}</td>{% endfor %}</tr>
  <tr><td>Total cost (USD)</td>{% for p in providers %}<td>{{ "$%.4f"|format(cost[p].total_usd) if cost.get(p) and cost[p].total_usd is not none else "n/a" }}</td>{% endfor %}</tr>
  <tr><td>Avg cost/request (USD)</td>{% for p in providers %}<td>{{ "$%.5f"|format(cost[p].avg_per_request_usd) if cost.get(p) and cost[p].avg_per_request_usd is not none else "n/a" }}</td>{% endfor %}</tr>
</table>

<h2>Reliability</h2>
<table>
  <tr><th>Metric</th>{% for p in providers %}<th>{{ p }}</th>{% endfor %}</tr>
  <tr><td>Success rate</td>{% for p in providers %}<td>{{ ("%.1f"|format(reliability[p].success_rate * 100)) + "%" if reliability.get(p) and reliability[p].success_rate is not none else "n/a" }}</td>{% endfor %}</tr>
  <tr><td>Errors</td>{% for p in providers %}<td>{{ reliability[p].error_count if reliability.get(p) is not none else "n/a" }}</td>{% endfor %}</tr>
  <tr><td>Timeouts</td>{% for p in providers %}<td>{{ reliability[p].timeout_count if reliability.get(p) is not none else "n/a" }}</td>{% endfor %}</tr>
</table>

<h2>Latency &amp; throughput by concurrency</h2>
{% for level in concurrency_levels %}
<h3>Concurrency {{ level }}</h3>
<table>
  <tr><th>Metric</th>{% for p in providers %}<th>{{ p }}</th>{% endfor %}</tr>
  <tr><td>Throughput (req/s)</td>{% for p in providers %}<td>{{ "%.2f"|format(latency[p][level].throughput_rps) if latency.get(p, {}).get(level) else "n/a" }}</td>{% endfor %}</tr>
  <tr><td>p50 (ms)</td>{% for p in providers %}<td>{{ "%.0f"|format(latency[p][level].p50_ms) if latency.get(p, {}).get(level) else "n/a" }}</td>{% endfor %}</tr>
  <tr><td>p95 (ms)</td>{% for p in providers %}<td>{{ "%.0f"|format(latency[p][level].p95_ms) if latency.get(p, {}).get(level) else "n/a" }}</td>{% endfor %}</tr>
  <tr><td>Error rate</td>{% for p in providers %}<td>{{ ("%.1f"|format(latency[p][level].error_rate * 100)) + "%" if latency.get(p, {}).get(level) else "n/a" }}</td>{% endfor %}</tr>
</table>
{% endfor %}

<h2>Model distribution</h2>
{% for p in providers %}
<h3>{{ p }}</h3>
<table>
  <tr><th>Model</th><th>Requests</th></tr>
  {% for model, n in routing.get(p, {}).get("model_distribution", {}).items() %}
  <tr><td>{{ model }}</td><td>{{ n }}</td></tr>
  {% endfor %}
</table>
{% endfor %}

</body>
</html>
"""


def _find_scenario(report: dict, scenario_type: str) -> dict | None:
    for s in report.get("scenarios", []):
        if s.get("scenario_type") == scenario_type:
            return s
    return None


async def run_all(labeled_configs: dict[str, BenchmarkConfig]) -> dict[str, dict]:
    """Runs each provider's config sequentially and returns {label: report}
    in the same shape save_report() writes to JSON.

    Sequential, not concurrent -- each provider has its own rate limits
    (rate_limit_rps) tuned for its own account; running them at the same
    time would just make each look artificially slower under contention
    with all the others, with no way to attribute latency to the right one.
    """
    reports: dict[str, dict] = {}
    for label, config in labeled_configs.items():
        runner = BenchmarkRunner(config)
        scenario_results = await runner.run()
        reports[label] = to_serializable(scenario_results, include_raw_results=True)
    return reports


def aggregate_comparison(labeled_reports: dict[str, dict]) -> dict:
    """Extracts the headline metrics from each provider's report into one
    side-by-side comparison structure. Works whether reports came fresh from
    run_all() or were json.load()-ed from disk -- concurrency-level keys are
    normalized to strings either way, since JSON always turns int keys into
    strings on a round-trip through disk but not when passed in-memory.
    """
    providers = list(labeled_reports.keys())
    quality: dict[str, float | None] = {}
    cost: dict[str, dict] = {}
    latency: dict[str, dict] = {}
    reliability: dict[str, dict] = {}
    routing: dict[str, dict] = {}

    for label, report in labeled_reports.items():
        accuracy = _find_scenario(report, "accuracy")
        if accuracy:
            m = accuracy["metrics"]
            quality[label] = m.get("quality", {}).get("avg_quality_score")
            cost[label] = {
                "total_usd": m.get("cost", {}).get("total_cost_usd"),
                "avg_per_request_usd": m.get("cost", {}).get("avg_cost_per_request_usd"),
            }
            routing[label] = {
                "accuracy": m.get("routing", {}).get("accuracy"),
                "model_distribution": m.get("routing", {}).get("model_distribution", {}),
            }

        load = _find_scenario(report, "load")
        if load:
            latency[label] = {str(k): v for k, v in load["metrics"].get("by_concurrency", {}).items()}

        rel = _find_scenario(report, "reliability")
        if rel:
            overall = rel["metrics"].get("overall", {})
            reliability[label] = {
                "success_rate": overall.get("success_rate"),
                "error_count": overall.get("error_count"),
                "timeout_count": overall.get("timeout_count"),
            }

    all_levels = {level for levels in latency.values() for level in levels}
    concurrency_levels = sorted(all_levels, key=int)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "providers": providers,
        "quality": quality,
        "cost": cost,
        "routing": routing,
        "latency": latency,
        "reliability": reliability,
        "concurrency_levels": concurrency_levels,
    }


def render_comparison_html(comparison: dict) -> str:
    return Template(HTML_TEMPLATE).render(**comparison)


def save_comparison(comparison: dict, output_dir: str) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    written: list[Path] = []
    json_path = out_dir / f"comparison-{timestamp}.json"
    json_path.write_text(json.dumps(comparison, indent=2, default=str))
    written.append(json_path)

    html_path = out_dir / f"comparison-{timestamp}.html"
    html_path.write_text(render_comparison_html(comparison))
    written.append(html_path)

    return written
