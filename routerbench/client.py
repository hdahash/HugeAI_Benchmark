"""HTTP client adapter for the router service under test.

Deliberately thin: the framework doesn't assume anything about the router's
internals, only that it exposes an HTTP endpoint accepting a prompt and
returning JSON. Field locations are all configurable dot-paths (see
config.RouterConfig) so this same client works against an OpenAI-compatible
endpoint, a custom router API, or the bundled mock router.
"""
from __future__ import annotations

import os
import time
import uuid

import httpx

from routerbench.config import RouterConfig, get_by_path
from routerbench.models import RequestStatus, RouterResponse
from routerbench.ratelimit import RateLimiter


class RouterClient:
    def __init__(self, config: RouterConfig, rate_limiter: RateLimiter | None = None):
        self.config = config
        self._rate_limiter = rate_limiter
        headers = dict(config.extra_headers)
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
            if api_key:
                headers[config.auth_header_name] = config.auth_header_format.format(api_key=api_key)
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers=headers,
            timeout=config.timeout_s,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "RouterClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _build_payload(self, prompt: str, force_model: str | None, extra_headers: dict[str, str] | None) -> tuple[dict, dict]:
        payload: dict = {
            "messages": [{"role": "user", "content": prompt}],
        }
        payload.update(self.config.request_extra_fields)
        if force_model:
            field_name = self.config.request_model_field or "model"
            payload[field_name] = force_model
        headers = dict(extra_headers or {})
        return payload, headers

    async def send(
        self,
        prompt: str,
        request_id: str | None = None,
        force_model: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> RouterResponse:
        request_id = request_id or str(uuid.uuid4())
        if self.config.cache_bust:
            # A fresh nonce every call, independent of request_id (which is
            # often a stable dataset item id reused across runs/scenarios and
            # so would defeat nothing) -- this must guarantee the router
            # never sees the same prompt text twice.
            prompt = f"{prompt}\n\n<!-- bench-nonce:{uuid.uuid4().hex} -->"
        payload, headers = self._build_payload(prompt, force_model, extra_headers)

        attempts = 0
        last_error: str | None = None
        last_status: int | None = None

        while attempts <= self.config.max_retries:
            attempts += 1
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire()
            start = time.perf_counter()
            try:
                resp = await self._client.request(
                    self.config.method,
                    self.config.chat_path,
                    json=payload,
                    headers=headers,
                    timeout=timeout_s or self.config.timeout_s,
                )
                elapsed = time.perf_counter() - start
                last_status = resp.status_code
                if resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}"
                    if attempts <= self.config.max_retries:
                        await self._sleep_backoff(attempts)
                        continue
                    return RouterResponse(
                        request_id=request_id,
                        status=RequestStatus.ERROR,
                        total_latency_s=elapsed,
                        http_status=resp.status_code,
                        error=last_error,
                        attempts=attempts,
                    )
                if resp.status_code >= 400:
                    return RouterResponse(
                        request_id=request_id,
                        status=RequestStatus.ERROR,
                        total_latency_s=elapsed,
                        http_status=resp.status_code,
                        error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                        attempts=attempts,
                    )

                data = resp.json()
                return RouterResponse(
                    request_id=request_id,
                    status=RequestStatus.SUCCESS,
                    model_used=get_by_path(data, self.config.response_model_path),
                    content=get_by_path(data, self.config.response_content_path) or "",
                    input_tokens=int(get_by_path(data, self.config.response_input_tokens_path) or 0),
                    output_tokens=int(get_by_path(data, self.config.response_output_tokens_path) or 0),
                    total_latency_s=elapsed,
                    http_status=resp.status_code,
                    attempts=attempts,
                    raw=data,
                )
            except httpx.TimeoutException:
                elapsed = time.perf_counter() - start
                last_error = "timeout"
                if attempts <= self.config.max_retries:
                    await self._sleep_backoff(attempts)
                    continue
                return RouterResponse(
                    request_id=request_id,
                    status=RequestStatus.TIMEOUT,
                    total_latency_s=elapsed,
                    error="request timed out",
                    attempts=attempts,
                )
            except httpx.HTTPError as exc:
                elapsed = time.perf_counter() - start
                last_error = str(exc)
                if attempts <= self.config.max_retries:
                    await self._sleep_backoff(attempts)
                    continue
                return RouterResponse(
                    request_id=request_id,
                    status=RequestStatus.ERROR,
                    total_latency_s=elapsed,
                    error=last_error,
                    attempts=attempts,
                )

        return RouterResponse(
            request_id=request_id,
            status=RequestStatus.ERROR,
            http_status=last_status,
            error=last_error or "unknown error",
            attempts=attempts,
        )

    async def _sleep_backoff(self, attempt: int) -> None:
        import asyncio

        await asyncio.sleep(self.config.retry_backoff_s * attempt)
