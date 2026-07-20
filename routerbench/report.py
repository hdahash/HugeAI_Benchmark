"""Aggregation and rendering of benchmark results into JSON / Markdown / HTML reports."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Template

from routerbench.models import RequestResult, ScenarioResult

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Router Benchmark Report</title>
<style>
  body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 2rem; color: #1a1a1a; background: #fff; }
  @media (prefers-color-scheme: dark) { body { color: #e8e8e8; background: #14161a; } table { background: #1d2026; } th { background: #262a32; } td, th { border-color: #333; } }
  h1 { font-size: 1.6rem; }
  h2 { margin-top: 2.5rem; border-bottom: 2px solid #ccc; padding-bottom: .3rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
  th, td { border: 1px solid #ccc; padding: 0.4rem 0.7rem; text-align: left; }
  th { background: #f0f0f0; }
  .metric-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
  .metric-card { border: 1px solid #ccc; border-radius: 8px; padding: 0.8rem 1.2rem; min-width: 160px; }
  .metric-card .label { font-size: 0.75rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.03em; }
  .metric-card .value { font-size: 1.4rem; font-weight: 600; }
  .muted { opacity: 0.6; font-size: 0.85rem; }
</style>
</head>
<body>
<h1>LLM Router Benchmark Report</h1>
<p class="muted">Generated {{ generated_at }}</p>

{% for scenario in scenarios %}
<h2>{{ scenario.name }} <span class="muted">({{ scenario.scenario_type }})</span></h2>

{% if scenario.scenario_type == "accuracy" %}
  <div class="metric-grid">
    <div class="metric-card"><div class="label">Routing Accuracy</div><div class="value">{{ "%.1f"|format((scenario.metrics.routing.accuracy or 0) * 100) }}%</div></div>
    <div class="metric-card"><div class="label">Avg Quality Score</div><div class="value">{{ "%.2f"|format(scenario.metrics.quality.avg_quality_score or 0) }}</div></div>
    <div class="metric-card"><div class="label">Total Cost (USD)</div><div class="value">${{ "%.4f"|format(scenario.metrics.cost.total_cost_usd) }}</div></div>
    {% if scenario.metrics.cost.savings_pct is defined %}
    <div class="metric-card"><div class="label">Savings vs Baseline</div><div class="value">{{ "%.1f"|format(scenario.metrics.cost.savings_pct) }}%</div></div>
    {% endif %}
  </div>

  <h3>Routing accuracy by category</h3>
  <table>
    <tr><th>Category</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>
    {% for cat, stats in scenario.metrics.routing.by_category.items() %}
    <tr><td>{{ cat }}</td><td>{{ stats.correct }}</td><td>{{ stats.total }}</td><td>{{ "%.1f"|format(stats.accuracy * 100) }}%</td></tr>
    {% endfor %}
  </table>

  <h3>Model distribution</h3>
  <table>
    <tr><th>Model</th><th>Requests routed</th></tr>
    {% for model, n in scenario.metrics.routing.model_distribution.items() %}
    <tr><td>{{ model }}</td><td>{{ n }}</td></tr>
    {% endfor %}
  </table>

  <h3>Cost by model</h3>
  <table>
    <tr><th>Model</th><th>Requests</th><th>Total cost (USD)</th></tr>
    {% for model, stats in scenario.metrics.cost.by_model.items() %}
    <tr><td>{{ model }}</td><td>{{ stats.count }}</td><td>${{ "%.4f"|format(stats.total_cost_usd) }}</td></tr>
    {% endfor %}
  </table>
{% endif %}

{% if scenario.scenario_type == "load" %}
  <table>
    <tr><th>Concurrency</th><th>Requests</th><th>Success</th><th>Error rate</th><th>Throughput (req/s)</th><th>p50 (ms)</th><th>p95 (ms)</th><th>p99 (ms)</th></tr>
    {% for level, stats in scenario.metrics.by_concurrency.items() %}
    <tr>
      <td>{{ level }}</td><td>{{ stats.total_requests }}</td><td>{{ stats.success_count }}</td>
      <td>{{ "%.1f"|format(stats.error_rate * 100) }}%</td><td>{{ "%.2f"|format(stats.throughput_rps) }}</td>
      <td>{{ "%.0f"|format(stats.p50_ms) }}</td><td>{{ "%.0f"|format(stats.p95_ms) }}</td><td>{{ "%.0f"|format(stats.p99_ms) }}</td>
    </tr>
    {% endfor %}
  </table>
{% endif %}

{% if scenario.scenario_type == "reliability" %}
  <table>
    <tr><th>Segment</th><th>Requests</th><th>Success rate</th><th>Error count</th><th>Timeout count</th><th>Retry recovery rate</th></tr>
    {% for seg, stats in scenario.metrics.items() %}
    <tr>
      <td>{{ seg }}</td><td>{{ stats.total_requests }}</td>
      <td>{{ "%.1f"|format((stats.success_rate or 0) * 100) }}%</td>
      <td>{{ stats.error_count }}</td><td>{{ stats.timeout_count }}</td>
      <td>{{ "%.1f"|format((stats.retry_recovery_rate * 100)) if stats.retry_recovery_rate is not none else "n/a" }}{{ "%" if stats.retry_recovery_rate is not none else "" }}</td>
    </tr>
    {% endfor %}
  </table>
{% endif %}

{% endfor %}
</body>
</html>
"""


def _result_to_dict(r: RequestResult) -> dict:
    return {
        "id": r.item.id,
        "category": r.item.category,
        "scenario": r.scenario,
        "concurrency_level": r.concurrency_level,
        "expected_model": r.item.expected_model,
        "model_used": r.response.model_used,
        "status": r.response.status.value,
        "latency_s": r.response.total_latency_s,
        "attempts": r.response.attempts,
        "error": r.response.error,
        "cost_usd": r.cost_usd,
        "quality_score": r.quality_score,
        "routing_correct": r.routing_correct,
    }


def to_serializable(scenario_results: list[ScenarioResult], include_raw_results: bool = True) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": [
            {
                "name": s.name,
                "scenario_type": s.scenario_type,
                "metrics": s.metrics,
                **({"results": [_result_to_dict(r) for r in s.results]} if include_raw_results else {}),
            }
            for s in scenario_results
        ],
    }


def render_html(report: dict) -> str:
    return Template(HTML_TEMPLATE).render(**report)


def render_markdown(report: dict) -> str:
    lines = [f"# LLM Router Benchmark Report", "", f"_Generated {report['generated_at']}_", ""]
    for scenario in report["scenarios"]:
        lines.append(f"## {scenario['name']} ({scenario['scenario_type']})")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(scenario["metrics"], indent=2, default=str))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def save_report(scenario_results: list[ScenarioResult], output_dir: str, formats: list[str]) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    report = to_serializable(scenario_results, include_raw_results=True)
    written: list[Path] = []

    if "json" in formats:
        path = out_dir / f"report-{timestamp}.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        written.append(path)

    if "html" in formats:
        # HTML template doesn't need the (large) per-request raw results.
        html_report = to_serializable(scenario_results, include_raw_results=False)
        path = out_dir / f"report-{timestamp}.html"
        path.write_text(render_html(html_report))
        written.append(path)

    if "markdown" in formats:
        path = out_dir / f"report-{timestamp}.md"
        path.write_text(render_markdown(report))
        written.append(path)

    return written
