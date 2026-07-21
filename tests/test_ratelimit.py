import asyncio
from unittest.mock import patch

from routerbench.ratelimit import RateLimiter


async def test_disabled_when_none_never_sleeps():
    limiter = RateLimiter(None)
    with patch("asyncio.sleep") as mock_sleep:
        for _ in range(5):
            await limiter.acquire()
    mock_sleep.assert_not_called()


async def test_enforces_minimum_spacing():
    limiter = RateLimiter(max_requests_per_second=2.0)  # min interval 0.5s
    fake_now = [1000.0]

    def fake_monotonic():
        return fake_now[0]

    async def fake_sleep(seconds):
        fake_now[0] += seconds

    with patch("time.monotonic", side_effect=fake_monotonic), patch("asyncio.sleep", side_effect=fake_sleep) as mock_sleep:
        await limiter.acquire()  # first call: no wait
        await limiter.acquire()  # second call: must wait ~0.5s
        await limiter.acquire()  # third call: must wait ~0.5s again

    assert mock_sleep.call_count == 2
    for call in mock_sleep.call_args_list:
        assert call.args[0] == 0.5


async def test_no_wait_if_calls_are_already_spaced_out():
    limiter = RateLimiter(max_requests_per_second=10.0)  # min interval 0.1s
    fake_now = [1000.0]

    def fake_monotonic():
        return fake_now[0]

    with patch("time.monotonic", side_effect=fake_monotonic), patch("asyncio.sleep") as mock_sleep:
        await limiter.acquire()
        fake_now[0] += 1.0  # plenty of real time has passed
        await limiter.acquire()

    mock_sleep.assert_not_called()


async def test_serializes_concurrent_acquire_calls():
    limiter = RateLimiter(max_requests_per_second=1000.0)
    results = await asyncio.gather(*[limiter.acquire() for _ in range(20)])
    assert len(results) == 20  # all completed without error/deadlock
