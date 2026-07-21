import httpx
import pytest
import respx

from routerbench.client import RouterClient
from routerbench.config import RouterConfig
from routerbench.models import RequestStatus


@pytest.fixture
def config():
    return RouterConfig(base_url="http://test-router", max_retries=1, retry_backoff_s=0.01, timeout_s=1.0)


class _CountingRateLimiter:
    def __init__(self):
        self.calls = 0

    async def acquire(self):
        self.calls += 1


@respx.mock
async def test_rate_limiter_called_once_per_attempt(config):
    route = respx.post("http://test-router/v1/chat/completions")
    route.side_effect = [
        httpx.Response(500),
        httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "ok"}}], "usage": {}}),
    ]
    limiter = _CountingRateLimiter()
    client = RouterClient(config, rate_limiter=limiter)
    response = await client.send("hi")
    await client.aclose()

    assert response.status == RequestStatus.SUCCESS
    assert response.attempts == 2
    assert limiter.calls == 2  # once per physical HTTP attempt, including the retry


@respx.mock
async def test_no_rate_limiter_by_default(config):
    respx.post("http://test-router/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "ok"}}], "usage": {}})
    )
    client = RouterClient(config)  # rate_limiter=None
    response = await client.send("hi")
    await client.aclose()
    assert response.status == RequestStatus.SUCCESS


@respx.mock
async def test_cache_bust_makes_identical_prompts_send_different_bodies():
    import json as _json

    config = RouterConfig(base_url="http://test-router", cache_bust=True)
    route = respx.post("http://test-router/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "ok"}}], "usage": {}})
    )
    client = RouterClient(config)
    await client.send("What is the capital of France?", request_id="qa-1")
    await client.send("What is the capital of France?", request_id="qa-1")
    await client.aclose()

    bodies = [_json.loads(c.request.content) for c in route.calls]
    contents = [b["messages"][0]["content"] for b in bodies]
    assert contents[0] != contents[1]
    assert "What is the capital of France?" in contents[0]
    assert "What is the capital of France?" in contents[1]


@respx.mock
async def test_cache_bust_disabled_sends_identical_prompt(config):
    import json as _json

    route = respx.post("http://test-router/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "ok"}}], "usage": {}})
    )
    client = RouterClient(config)  # cache_bust=False (default)
    await client.send("hello")
    await client.send("hello")
    await client.aclose()

    bodies = [_json.loads(c.request.content) for c in route.calls]
    contents = [b["messages"][0]["content"] for b in bodies]
    assert contents[0] == contents[1] == "hello"


@respx.mock
async def test_send_with_history_prepends_prior_turns(config):
    import json as _json

    route = respx.post("http://test-router/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "ok"}}], "usage": {}})
    )
    client = RouterClient(config)
    history = [
        {"role": "user", "content": "I'm planning a trip to France."},
        {"role": "assistant", "content": "Paris is a great city to visit."},
    ]
    await client.send("What is the capital of the country I mentioned?", history=history)
    await client.aclose()

    body = _json.loads(route.calls.last.request.content)
    assert body["messages"] == [
        {"role": "user", "content": "I'm planning a trip to France."},
        {"role": "assistant", "content": "Paris is a great city to visit."},
        {"role": "user", "content": "What is the capital of the country I mentioned?"},
    ]


@respx.mock
async def test_send_without_history_is_single_turn(config):
    import json as _json

    route = respx.post("http://test-router/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "ok"}}], "usage": {}})
    )
    client = RouterClient(config)
    await client.send("hi")
    await client.aclose()

    body = _json.loads(route.calls.last.request.content)
    assert body["messages"] == [{"role": "user", "content": "hi"}]


@respx.mock
async def test_cache_bust_with_history_only_nonces_final_turn(config):
    import json as _json

    config.cache_bust = True
    route = respx.post("http://test-router/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "ok"}}], "usage": {}})
    )
    client = RouterClient(config)
    history = [{"role": "user", "content": "context turn"}, {"role": "assistant", "content": "reply turn"}]
    await client.send("final question", history=history)
    await client.aclose()

    body = _json.loads(route.calls.last.request.content)
    assert body["messages"][0] == {"role": "user", "content": "context turn"}
    assert body["messages"][1] == {"role": "assistant", "content": "reply turn"}
    assert body["messages"][2]["content"].startswith("final question\n\n<!-- bench-nonce:")


@respx.mock
async def test_send_success(config):
    respx.post("http://test-router/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "small-model",
                "choices": [{"message": {"content": "hello there"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )
    )
    client = RouterClient(config)
    response = await client.send("hi", request_id="r1")
    await client.aclose()

    assert response.status == RequestStatus.SUCCESS
    assert response.model_used == "small-model"
    assert response.content == "hello there"
    assert response.input_tokens == 5
    assert response.output_tokens == 3
    assert response.attempts == 1


@respx.mock
async def test_send_retries_then_succeeds(config):
    route = respx.post("http://test-router/v1/chat/completions")
    route.side_effect = [
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "ok"}}], "usage": {}}),
    ]
    client = RouterClient(config)
    response = await client.send("hi")
    await client.aclose()

    assert response.status == RequestStatus.SUCCESS
    assert response.attempts == 2


@respx.mock
async def test_send_exhausts_retries_on_persistent_5xx(config):
    respx.post("http://test-router/v1/chat/completions").mock(return_value=httpx.Response(500))
    client = RouterClient(config)
    response = await client.send("hi")
    await client.aclose()

    assert response.status == RequestStatus.ERROR
    assert response.attempts == config.max_retries + 1


@respx.mock
async def test_send_4xx_does_not_retry(config):
    route = respx.post("http://test-router/v1/chat/completions").mock(return_value=httpx.Response(400, text="bad request"))
    client = RouterClient(config)
    response = await client.send("hi")
    await client.aclose()

    assert response.status == RequestStatus.ERROR
    assert response.attempts == 1
    assert route.call_count == 1


@respx.mock
async def test_send_timeout(config):
    respx.post("http://test-router/v1/chat/completions").mock(side_effect=httpx.TimeoutException("timed out"))
    client = RouterClient(config)
    response = await client.send("hi")
    await client.aclose()

    assert response.status == RequestStatus.TIMEOUT


@respx.mock
async def test_force_model_sets_request_field(config):
    route = respx.post("http://test-router/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"model": "large-model", "choices": [{"message": {"content": "x"}}], "usage": {}})
    )
    client = RouterClient(config)
    await client.send("hi", force_model="large-model")
    await client.aclose()

    sent_body = route.calls.last.request.content
    import json

    assert json.loads(sent_body)["model"] == "large-model"
