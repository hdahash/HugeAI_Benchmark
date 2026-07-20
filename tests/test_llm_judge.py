import httpx
import respx

from routerbench.client import RouterClient
from routerbench.config import JudgeConfig, RouterConfig
from routerbench.models import DatasetItem
from routerbench.scoring.llm_judge import LLMJudgeScorer

ITEM = DatasetItem(id="1", prompt="Explain recursion.", category="reasoning")


def make_scorer(judge_config: JudgeConfig | None = None) -> LLMJudgeScorer:
    router_config = RouterConfig(base_url="http://judge-test", max_retries=0, timeout_s=1.0)
    client = RouterClient(router_config)
    return LLMJudgeScorer(client, judge_config or JudgeConfig(enabled=True, model="judge-model"))


@respx.mock
async def test_judge_parses_plain_integer_score():
    respx.post("http://judge-test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"model": "judge-model", "choices": [{"message": {"content": "8"}}], "usage": {}})
    )
    scorer = make_scorer()
    score = await scorer.score(ITEM, "some response content")
    await scorer.client.aclose()
    assert score == 0.8


@respx.mock
async def test_judge_extracts_first_number_from_prose_reply():
    respx.post("http://judge-test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"model": "judge-model", "choices": [{"message": {"content": "Score: 7 out of 10"}}], "usage": {}}
        )
    )
    scorer = make_scorer()
    score = await scorer.score(ITEM, "some response content")
    await scorer.client.aclose()
    assert score == 0.7


@respx.mock
async def test_judge_clamps_out_of_range_score():
    respx.post("http://judge-test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"model": "judge-model", "choices": [{"message": {"content": "15"}}], "usage": {}})
    )
    scorer = make_scorer()
    score = await scorer.score(ITEM, "content")
    await scorer.client.aclose()
    assert score == 1.0


@respx.mock
async def test_judge_returns_none_on_unparseable_reply():
    respx.post("http://judge-test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"model": "judge-model", "choices": [{"message": {"content": "not a score"}}], "usage": {}})
    )
    scorer = make_scorer()
    score = await scorer.score(ITEM, "content")
    await scorer.client.aclose()
    assert score is None


@respx.mock
async def test_judge_returns_none_on_http_error():
    respx.post("http://judge-test/v1/chat/completions").mock(return_value=httpx.Response(500))
    scorer = make_scorer()
    score = await scorer.score(ITEM, "content")
    await scorer.client.aclose()
    assert score is None


@respx.mock
async def test_judge_skips_call_for_empty_content():
    route = respx.post("http://judge-test/v1/chat/completions").mock(return_value=httpx.Response(200, json={}))
    scorer = make_scorer()
    score = await scorer.score(ITEM, "   ")
    await scorer.client.aclose()
    assert score == 0.0
    assert route.call_count == 0


@respx.mock
async def test_judge_uses_category_specific_rubric_and_forces_model():
    route = respx.post("http://judge-test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"model": "judge-model", "choices": [{"message": {"content": "9"}}], "usage": {}})
    )
    judge_config = JudgeConfig(enabled=True, model="judge-model", rubric={"reasoning": "CUSTOM_RUBRIC_TEXT"})
    scorer = make_scorer(judge_config)
    await scorer.score(ITEM, "content")
    await scorer.client.aclose()

    import json

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "judge-model"
    assert "CUSTOM_RUBRIC_TEXT" in sent["messages"][0]["content"]
