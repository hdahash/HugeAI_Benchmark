from routerbench.models import DatasetItem
from routerbench.scoring.base import NullScorer
from routerbench.scoring.composite import CompositeScorer
from routerbench.scoring.heuristic import HeuristicScorer


async def test_heuristic_scorer_full_match():
    scorer = HeuristicScorer()
    item = DatasetItem(id="1", prompt="p", expected_answer_contains=["Paris"])
    assert await scorer.score(item, "The capital of France is Paris.") == 1.0


async def test_heuristic_scorer_no_match():
    scorer = HeuristicScorer()
    item = DatasetItem(id="1", prompt="p", expected_answer_contains=["Paris"])
    assert await scorer.score(item, "The capital of France is London.") == 0.0


async def test_heuristic_scorer_partial_match():
    scorer = HeuristicScorer()
    item = DatasetItem(id="1", prompt="p", expected_answer_contains=["Paris", "France"])
    assert await scorer.score(item, "It is Paris.") == 0.5


async def test_heuristic_scorer_case_insensitive():
    scorer = HeuristicScorer()
    item = DatasetItem(id="1", prompt="p", expected_answer_contains=["PARIS"])
    assert await scorer.score(item, "the answer is paris") == 1.0


async def test_heuristic_scorer_no_expectations():
    scorer = HeuristicScorer()
    item = DatasetItem(id="1", prompt="p")
    assert await scorer.score(item, "anything") is None


async def test_null_scorer():
    scorer = NullScorer()
    item = DatasetItem(id="1", prompt="p", expected_answer_contains=["x"])
    assert await scorer.score(item, "x") is None


async def test_heuristic_scorer_excludes_only_all_absent():
    scorer = HeuristicScorer()
    item = DatasetItem(id="1", prompt="p", expected_answer_excludes=["😀", "🎉"])
    assert await scorer.score(item, "The answer is 4.") == 1.0


async def test_heuristic_scorer_excludes_only_one_present():
    scorer = HeuristicScorer()
    item = DatasetItem(id="1", prompt="p", expected_answer_excludes=["😀", "🎉"])
    assert await scorer.score(item, "The answer is 4 🎉") == 0.5


async def test_heuristic_scorer_combined_contains_and_excludes():
    scorer = HeuristicScorer()
    item = DatasetItem(id="1", prompt="p", expected_answer_contains=["4"], expected_answer_excludes=["GATEWAY_POLICY"])
    assert await scorer.score(item, "The answer is 4.") == 1.0  # both checks pass
    assert await scorer.score(item, "GATEWAY_POLICY: the answer is 4.") == 0.5  # contains passes, excludes fails
    assert await scorer.score(item, "GATEWAY_POLICY: no idea.") == 0.0  # both fail


async def test_heuristic_scorer_no_expectations_and_no_excludes():
    scorer = HeuristicScorer()
    item = DatasetItem(id="1", prompt="p")
    assert await scorer.score(item, "anything") is None


class _StubScorer:
    def __init__(self, value):
        self.value = value
        self.called = False

    async def score(self, item, content):
        self.called = True
        return self.value


async def test_composite_prefers_code_exec_when_test_cases_present():
    code_exec = _StubScorer(1.0)
    heuristic = _StubScorer(0.5)
    judge = _StubScorer(0.7)
    composite = CompositeScorer(heuristic=heuristic, code_exec=code_exec, judge=judge)

    item = DatasetItem(id="1", prompt="p", expected_answer_contains=["x"], metadata={"test_cases": [{"call": "1", "expected": 1}]})
    score = await composite.score(item, "content")

    assert score == 1.0
    assert code_exec.called
    assert not heuristic.called
    assert not judge.called


async def test_composite_prefers_heuristic_when_no_test_cases():
    code_exec = _StubScorer(1.0)
    heuristic = _StubScorer(0.5)
    judge = _StubScorer(0.7)
    composite = CompositeScorer(heuristic=heuristic, code_exec=code_exec, judge=judge)

    item = DatasetItem(id="1", prompt="p", expected_answer_contains=["x"])
    score = await composite.score(item, "content")

    assert score == 0.5
    assert not code_exec.called
    assert heuristic.called
    assert not judge.called


async def test_composite_dispatches_to_heuristic_for_excludes_only():
    code_exec = _StubScorer(1.0)
    heuristic = _StubScorer(0.5)
    judge = _StubScorer(0.7)
    composite = CompositeScorer(heuristic=heuristic, code_exec=code_exec, judge=judge)

    item = DatasetItem(id="1", prompt="p", expected_answer_excludes=["GATEWAY_POLICY"])
    score = await composite.score(item, "content")

    assert score == 0.5
    assert not code_exec.called
    assert heuristic.called
    assert not judge.called


async def test_composite_falls_back_to_judge():
    composite = CompositeScorer(heuristic=HeuristicScorer(), code_exec=None, judge=_StubScorer(0.7))
    item = DatasetItem(id="1", prompt="p")
    score = await composite.score(item, "content")
    assert score == 0.7


async def test_composite_unscored_when_nothing_applies():
    composite = CompositeScorer(heuristic=HeuristicScorer(), code_exec=None, judge=None)
    item = DatasetItem(id="1", prompt="p")
    assert await composite.score(item, "content") is None
