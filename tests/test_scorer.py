from routerbench.models import DatasetItem
from routerbench.scorer import HeuristicScorer, NullScorer, build_scorer


def test_heuristic_scorer_full_match():
    scorer = HeuristicScorer()
    item = DatasetItem(id="1", prompt="p", expected_answer_contains=["Paris"])
    assert scorer.score(item, "The capital of France is Paris.") == 1.0


def test_heuristic_scorer_no_match():
    scorer = HeuristicScorer()
    item = DatasetItem(id="1", prompt="p", expected_answer_contains=["Paris"])
    assert scorer.score(item, "The capital of France is London.") == 0.0


def test_heuristic_scorer_partial_match():
    scorer = HeuristicScorer()
    item = DatasetItem(id="1", prompt="p", expected_answer_contains=["Paris", "France"])
    assert scorer.score(item, "It is Paris.") == 0.5


def test_heuristic_scorer_case_insensitive():
    scorer = HeuristicScorer()
    item = DatasetItem(id="1", prompt="p", expected_answer_contains=["PARIS"])
    assert scorer.score(item, "the answer is paris") == 1.0


def test_heuristic_scorer_no_expectations():
    scorer = HeuristicScorer()
    item = DatasetItem(id="1", prompt="p")
    assert scorer.score(item, "anything") is None


def test_null_scorer():
    scorer = NullScorer()
    item = DatasetItem(id="1", prompt="p", expected_answer_contains=["x"])
    assert scorer.score(item, "x") is None


def test_build_scorer():
    assert isinstance(build_scorer("heuristic"), HeuristicScorer)
    assert isinstance(build_scorer("none"), NullScorer)
    try:
        build_scorer("bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass
