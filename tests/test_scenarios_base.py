import asyncio

from routerbench.models import DatasetItem, RequestStatus, RouterResponse
from routerbench.scenarios.base import ExecutionContext, run_one
from routerbench.scoring.base import NullScorer


class _StubClient:
    def __init__(self):
        self.calls = []

    async def send(self, prompt, request_id=None, force_model=None, extra_headers=None, timeout_s=None, history=None):
        self.calls.append({"prompt": prompt, "history": history})
        return RouterResponse(request_id=request_id, status=RequestStatus.SUCCESS, model_used="m", content="ok")


async def test_run_one_passes_conversation_as_history():
    client = _StubClient()
    ctx = ExecutionContext(client=client, pricing={}, scorer=NullScorer())
    item = DatasetItem(
        id="mt-1",
        prompt="final question",
        conversation=[{"role": "user", "content": "first turn"}, {"role": "assistant", "content": "reply"}],
    )
    semaphore = asyncio.Semaphore(1)

    await run_one(ctx, item, "test", semaphore)

    assert len(client.calls) == 1
    assert client.calls[0]["prompt"] == "final question"
    assert client.calls[0]["history"] == item.conversation


async def test_run_one_passes_none_history_for_single_turn_item():
    client = _StubClient()
    ctx = ExecutionContext(client=client, pricing={}, scorer=NullScorer())
    item = DatasetItem(id="qa-1", prompt="hi")
    semaphore = asyncio.Semaphore(1)

    await run_one(ctx, item, "test", semaphore)

    assert client.calls[0]["history"] is None
