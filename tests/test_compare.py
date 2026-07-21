from routerbench.compare import aggregate_comparison, render_comparison_html


def _accuracy_scenario(quality, total_cost, avg_cost, accuracy=None, model_distribution=None):
    return {
        "name": "routing_accuracy",
        "scenario_type": "accuracy",
        "metrics": {
            "routing": {"accuracy": accuracy, "model_distribution": model_distribution or {}},
            "quality": {"avg_quality_score": quality},
            "cost": {"total_cost_usd": total_cost, "avg_cost_per_request_usd": avg_cost},
        },
    }


def _load_scenario(by_concurrency):
    return {"name": "load_test", "scenario_type": "load", "metrics": {"by_concurrency": by_concurrency}}


def _reliability_scenario(success_rate, error_count=0, timeout_count=0):
    return {
        "name": "reliability",
        "scenario_type": "reliability",
        "metrics": {"overall": {"success_rate": success_rate, "error_count": error_count, "timeout_count": timeout_count}},
    }


def test_aggregate_extracts_quality_cost_routing():
    reports = {
        "hugeai": {
            "scenarios": [_accuracy_scenario(0.9, 0.05, 0.001, accuracy=0.8, model_distribution={"gpt-4o": 10})]
        },
        "openrouter": {
            "scenarios": [_accuracy_scenario(0.85, 0.03, 0.0008, accuracy=None, model_distribution={"claude-3-haiku": 10})]
        },
    }
    comparison = aggregate_comparison(reports)

    assert comparison["providers"] == ["hugeai", "openrouter"]
    assert comparison["quality"]["hugeai"] == 0.9
    assert comparison["quality"]["openrouter"] == 0.85
    assert comparison["cost"]["hugeai"]["total_usd"] == 0.05
    assert comparison["routing"]["hugeai"]["accuracy"] == 0.8
    assert comparison["routing"]["openrouter"]["accuracy"] is None
    assert comparison["routing"]["hugeai"]["model_distribution"] == {"gpt-4o": 10}


def test_aggregate_normalizes_concurrency_keys_to_strings():
    # in-memory (int keys, as load.py produces before JSON serialization)
    reports_in_memory = {"a": {"scenarios": [_load_scenario({1: {"throughput_rps": 5.0, "p50_ms": 100, "p95_ms": 200, "error_rate": 0.0}})]}}
    # loaded from disk (str keys, as JSON always produces)
    reports_from_disk = {"a": {"scenarios": [_load_scenario({"1": {"throughput_rps": 5.0, "p50_ms": 100, "p95_ms": 200, "error_rate": 0.0}})]}}

    comp_memory = aggregate_comparison(reports_in_memory)
    comp_disk = aggregate_comparison(reports_from_disk)

    assert comp_memory["latency"]["a"]["1"]["throughput_rps"] == 5.0
    assert comp_disk["latency"]["a"]["1"]["throughput_rps"] == 5.0
    assert comp_memory["concurrency_levels"] == ["1"]


def test_aggregate_sorts_concurrency_levels_numerically():
    reports = {
        "a": {
            "scenarios": [
                _load_scenario(
                    {
                        "20": {"throughput_rps": 1.0, "p50_ms": 1, "p95_ms": 2, "error_rate": 0.0},
                        "5": {"throughput_rps": 2.0, "p50_ms": 1, "p95_ms": 2, "error_rate": 0.0},
                        "1": {"throughput_rps": 3.0, "p50_ms": 1, "p95_ms": 2, "error_rate": 0.0},
                    }
                )
            ]
        }
    }
    comparison = aggregate_comparison(reports)
    assert comparison["concurrency_levels"] == ["1", "5", "20"]  # not lexicographic ("1","20","5")


def test_aggregate_handles_missing_scenarios_gracefully():
    reports = {"a": {"scenarios": [_accuracy_scenario(0.5, 0.01, 0.001)]}, "b": {"scenarios": []}}
    comparison = aggregate_comparison(reports)
    assert comparison["quality"]["a"] == 0.5
    assert "b" not in comparison["quality"]
    assert "b" not in comparison["latency"]


def test_aggregate_extracts_reliability():
    reports = {"a": {"scenarios": [_reliability_scenario(0.95, error_count=2, timeout_count=1)]}}
    comparison = aggregate_comparison(reports)
    assert comparison["reliability"]["a"]["success_rate"] == 0.95
    assert comparison["reliability"]["a"]["error_count"] == 2
    assert comparison["reliability"]["a"]["timeout_count"] == 1


def test_render_comparison_html_smoke():
    reports = {
        "hugeai": {"scenarios": [_accuracy_scenario(0.9, 0.05, 0.001), _reliability_scenario(1.0)]},
        "openrouter": {"scenarios": [_accuracy_scenario(0.8, 0.03, 0.0007), _reliability_scenario(0.98, error_count=1)]},
    }
    comparison = aggregate_comparison(reports)
    html = render_comparison_html(comparison)

    assert "hugeai" in html
    assert "openrouter" in html
    assert "<table>" in html
