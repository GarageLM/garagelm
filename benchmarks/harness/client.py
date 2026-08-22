"""Async OpenAI-compatible chat client for mlx_lm.server.

Rules that matter for this server (mlx-lm 0.31.x, read from its source):
- never send `seed`: a seeded request is excluded from continuous batching,
  and the server reseeds per request so repeats would be identical anyway;
- `chat_template_kwargs` is honoured per request (thinking on/off);
- the response carries `message.reasoning` separately from `message.content`
  when the tokenizer has think tokens, and `usage.prompt_tokens` /
  `usage.completion_tokens`.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import httpx

from .config import Sampler


@dataclass
class Completion:
    text: str = ""
    reasoning: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    wall_s: float = 0.0
    ttft_s: Optional[float] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ChatClient:
    def __init__(self, base_url: str, concurrency: int = 8, model: str = "default_model"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(timeout=None,
                                         limits=httpx.Limits(max_connections=concurrency + 4))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> Dict[str, Any]:
        r = await self._client.get(f"{self.base_url}/models", timeout=10)
        r.raise_for_status()
        return r.json()

    def _body(self, messages, sampler: Sampler, max_tokens, enable_thinking, stop, stream):
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens if max_tokens is not None else sampler.max_tokens),
            "temperature": sampler.temperature,
            "top_p": sampler.top_p,
            "top_k": sampler.top_k,
            "presence_penalty": sampler.presence_penalty,
            "chat_template_kwargs": {
                "enable_thinking": sampler.enable_thinking if enable_thinking is None else enable_thinking
            },
            "stream": stream,
        }
        if stop:
            body["stop"] = stop
        assert "seed" not in body
        return body

    async def complete(self, messages: List[Dict[str, str]], sampler: Sampler,
                       max_tokens: Optional[int] = None, enable_thinking: Optional[bool] = None,
                       stop: Optional[List[str]] = None, stream: bool = False,
                       retries: int = 1, timeout_s: Optional[float] = None) -> Completion:
        body = self._body(messages, sampler, max_tokens, enable_thinking, stop, stream)
        # default: ~5 tok/s worst case per stream + slack. Callers that queue more
        # requests than the server's decode concurrency must pass a larger timeout.
        timeout = timeout_s if timeout_s else body["max_tokens"] / 5.0 + 60.0
        last_err = None
        for attempt in range(retries + 1):
            async with self.sem:
                t0 = time.time()
                try:
                    if stream:
                        c = await self._stream(body, timeout, t0)
                    else:
                        c = await self._once(body, timeout, t0)
                    c.wall_s = time.time() - t0
                    return c
                except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
                    last_err = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(1.0 + attempt)
        return Completion(error=last_err, wall_s=0.0)

    async def _once(self, body, timeout, t0) -> Completion:
        r = await self._client.post(f"{self.base_url}/chat/completions", json=body, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        ch = j["choices"][0]
        msg = ch.get("message", {})
        usage = j.get("usage") or {}
        return Completion(text=msg.get("content") or "", reasoning=msg.get("reasoning") or "",
                          prompt_tokens=usage.get("prompt_tokens"),
                          completion_tokens=usage.get("completion_tokens"),
                          finish_reason=ch.get("finish_reason"))

    async def _stream(self, body, timeout, t0) -> Completion:
        text, reasoning, ttft, finish, usage = [], [], None, None, {}
        async with self._client.stream("POST", f"{self.base_url}/chat/completions",
                                       json=body, timeout=timeout) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                j = json.loads(payload)
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices", []):
                    d = ch.get("delta", {}) or {}
                    if d.get("content") or d.get("reasoning"):
                        if ttft is None:
                            ttft = time.time() - t0
                    if d.get("content"):
                        text.append(d["content"])
                    if d.get("reasoning"):
                        reasoning.append(d["reasoning"])
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        return Completion(text="".join(text), reasoning="".join(reasoning),
                          prompt_tokens=usage.get("prompt_tokens"),
                          completion_tokens=usage.get("completion_tokens"),
                          finish_reason=finish, ttft_s=ttft)
