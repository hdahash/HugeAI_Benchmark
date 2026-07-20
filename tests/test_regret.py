from routerbench.metrics.regret import routing_regret_summary
from routerbench.models import DatasetItem, RequestResult, RequestStatus, RouterResponse


def make_result(id_, model_used, quality, cost, category="reasoning", status=RequestStatus.SUCCESS):
    item = DatasetItem(id=id_, prompt="p", category=category)
    response = RouterResponse(request_id=id_, status=status, model_used=model_used, total_latency_s=0.1)
    r = RequestResult(item=item, response=response)
    r.quality_score = quality
    r.cost_usd = cost
    return r


def test_regret_no_comparable_requests():
    assert routing_regret_summary([], []) == {"comparable_requests": 0}


def test_regret_matches_by_item_id_and_computes_deltas():
    routed = [
        make_result("a", "small-model", quality=0.8, cost=0.001),
        make_result("b", "large-model", quality=0.9, cost=0.01),
    ]
    baseline = [
        make_result("a", "large-model", quality=0.9, cost=0.01),
        make_result("b", "large-model", quality=0.9, cost=0.01),
    ]
    summary = routing_regret_summary(routed, baseline)

    assert summary["comparable_requests"] == 2
    # item a: regret = 0.9 - 0.8 = 0.1 (router worse); item b: regret = 0 (matched)
    assert round(summary["avg_quality_regret"], 4) == round((0.1 + 0.0) / 2, 4)
    assert summary["pct_no_quality_loss"] == 0.5
    # item a saved 0.009, item b saved 0.0
    assert round(summary["total_cost_savings_usd"], 6) == round(0.009, 6)


def test_regret_skips_items_missing_from_baseline():
    routed = [make_result("a", "small-model", quality=0.8, cost=0.001)]
    baseline = [make_result("different-id", "large-model", quality=0.9, cost=0.01)]
    summary = routing_regret_summary(routed, baseline)
    assert summary["comparable_requests"] == 0


def test_regret_skips_unscored_or_failed_items():
    routed = [
        make_result("a", "small-model", quality=None, cost=0.001),  # unscored
        make_result("b", None, quality=0.5, cost=None, status=RequestStatus.ERROR),  # failed
        make_result("c", "small-model", quality=0.7, cost=0.001),
    ]
    baseline = [
        make_result("a", "large-model", quality=0.9, cost=0.01),
        make_result("b", "large-model", quality=0.9, cost=0.01),
        make_result("c", "large-model", quality=0.9, cost=0.01),
    ]
    summary = routing_regret_summary(routed, baseline)
    assert summary["comparable_requests"] == 1


def test_regret_worst_items_sorted_descending():
    routed = [
        make_result("a", "small-model", quality=0.5, cost=0.001),  # regret 0.4
        make_result("b", "small-model", quality=0.85, cost=0.001),  # regret 0.05
        make_result("c", "small-model", quality=0.2, cost=0.001),  # regret 0.7
    ]
    baseline = [make_result(i, "large-model", quality=0.9, cost=0.01) for i in ("a", "b", "c")]
    summary = routing_regret_summary(routed, baseline, top_n=2)
    worst_ids = [d["id"] for d in summary["worst_regret_items"]]
    assert worst_ids == ["c", "a"]
