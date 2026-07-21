from routerbench.config import ModelPricing
from routerbench.metrics.cost import cost_summary, request_cost_usd
from routerbench.metrics.latency import batch_stats, latency_stats, percentile
from routerbench.metrics.reliability import reliability_summary
from routerbench.metrics.routing import routing_accuracy
from routerbench.models import DatasetItem, RequestResult, RequestStatus, RouterResponse


def make_result(id_, model_used, expected_model=None, status=RequestStatus.SUCCESS, latency=0.1, cost=None, routing_correct=None, attempts=1, error=None):
    item = DatasetItem(id=id_, prompt="p", expected_model=expected_model)
    response = RouterResponse(
        request_id=id_, status=status, model_used=model_used, total_latency_s=latency, attempts=attempts, error=error,
    )
    r = RequestResult(item=item, response=response)
    r.cost_usd = cost
    r.routing_correct = routing_correct
    return r


def test_percentile_basic():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 50) == 30.0
    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 50.0


def test_latency_stats_empty():
    stats = latency_stats([])
    assert stats["count"] == 0
    assert stats["mean_ms"] == 0.0


def test_latency_stats_values():
    stats = latency_stats([0.1, 0.2, 0.3])
    assert stats["count"] == 3
    assert stats["min_ms"] == 100.0
    assert stats["max_ms"] == 300.0


def test_batch_stats_throughput():
    results = [make_result(f"r{i}", "small-model", latency=0.1) for i in range(10)]
    stats = batch_stats(results, wall_time_s=2.0)
    assert stats["success_count"] == 10
    assert stats["throughput_rps"] == 5.0
    assert stats["error_rate"] == 0.0


def test_batch_stats_with_errors():
    ok = [make_result(f"r{i}", "small-model") for i in range(3)]
    bad = [make_result("bad", None, status=RequestStatus.ERROR, error="HTTP 500")]
    stats = batch_stats(ok + bad, wall_time_s=1.0)
    assert stats["total_requests"] == 4
    assert stats["success_count"] == 3
    assert stats["error_rate"] == 0.25


def test_request_cost_usd():
    pricing = {"small-model": ModelPricing(input_per_1k=0.001, output_per_1k=0.002)}
    cost = request_cost_usd("small-model", 1000, 500, pricing)
    assert cost == 0.001 + 0.001


def test_request_cost_usd_unknown_model():
    assert request_cost_usd("unknown", 100, 100, {}) is None


def test_request_cost_usd_prefix_fallback_for_dated_snapshot():
    pricing = {"gpt-4o": ModelPricing(input_per_1k=0.0025, output_per_1k=0.0025)}
    cost = request_cost_usd("gpt-4o-2024-08-06", 1000, 1000, pricing)
    assert cost == 0.0025 + 0.0025


def test_request_cost_usd_prefix_fallback_picks_longest_match():
    pricing = {
        "gpt-4": ModelPricing(input_per_1k=0.03, output_per_1k=0.03),
        "gpt-4o": ModelPricing(input_per_1k=0.0025, output_per_1k=0.0025),
    }
    cost = request_cost_usd("gpt-4o-2024-08-06", 1000, 0, pricing)
    assert cost == 0.0025  # matched "gpt-4o", not the shorter "gpt-4"


def test_request_cost_usd_no_prefix_match():
    pricing = {"claude-3-haiku": ModelPricing(input_per_1k=0.00025, output_per_1k=0.00025)}
    assert request_cost_usd("gpt-4o-2024-08-06", 100, 100, pricing) is None


def test_cost_summary_with_baseline():
    r1 = make_result("r1", "small-model")
    r1.cost_usd = 0.001
    r1.response.input_tokens = 100
    r1.response.output_tokens = 50
    pricing = {
        "small-model": ModelPricing(input_per_1k=0.0001, output_per_1k=0.0002),
        "large-model": ModelPricing(input_per_1k=0.005, output_per_1k=0.015),
    }
    summary = cost_summary([r1], pricing, baseline_model="large-model")
    assert summary["total_cost_usd"] == 0.001
    assert summary["baseline_model"] == "large-model"
    assert summary["savings_usd"] > 0


def test_routing_accuracy():
    results = [
        make_result("r1", "small-model", expected_model="small-model", routing_correct=True),
        make_result("r2", "large-model", expected_model="small-model", routing_correct=False),
    ]
    for r in results:
        r.item.category = "simple_qa"
    acc = routing_accuracy(results)
    assert acc["checkable_requests"] == 2
    assert acc["correct"] == 1
    assert acc["accuracy"] == 0.5
    assert acc["confusion_matrix"]["small-model"]["small-model"] == 1
    assert acc["confusion_matrix"]["small-model"]["large-model"] == 1


def test_routing_accuracy_no_expected_model():
    results = [make_result("r1", "small-model")]
    acc = routing_accuracy(results)
    assert acc["checkable_requests"] == 0
    assert acc["accuracy"] is None
    assert acc["model_distribution"]["small-model"] == 1


def test_reliability_summary():
    results = [
        make_result("r1", "small-model", status=RequestStatus.SUCCESS),
        make_result("r2", None, status=RequestStatus.ERROR, error="HTTP 500", attempts=3),
        make_result("r3", None, status=RequestStatus.TIMEOUT, error="timeout"),
    ]
    summary = reliability_summary(results)
    assert summary["total_requests"] == 3
    assert summary["success_count"] == 1
    assert summary["error_count"] == 1
    assert summary["timeout_count"] == 1
    assert round(summary["success_rate"], 4) == round(1 / 3, 4)
    assert summary["retried_requests"] == 1


def test_reliability_summary_empty():
    assert reliability_summary([]) == {"total_requests": 0}
