"""Any OpenAI-compatible ``/chat/completions`` endpoint over stdlib urllib.

Covers Ollama (``http://localhost:11434/v1``, the default), vLLM, llama.cpp
server, LM Studio, and hosted OpenAI-compatible APIs. No SDK dependency: one
POST, JSON in and out.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from ..lm import Completion, LMError, approx_tokens

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatProvider:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env, "")
        # temperature 0: optimization compares candidates, and sampling noise
        # would make a worse prompt win by luck.
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._sleep = sleep

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> Completion:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        last: LMError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                data = self._post(body)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                last = LMError(f"HTTP {exc.code}: {detail}", retryable=exc.code in RETRYABLE_STATUS, status=exc.code)
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = LMError(f"connection error: {exc}", retryable=True)
                retry_after = None
            if not last.retryable or attempt == self.max_retries:
                raise last
            # Exponential backoff with full jitter (spreads retries from 8 eval
            # threads instead of having them hammer the server in lockstep);
            # a server-provided Retry-After wins when present.
            delay = random.uniform(0, self.backoff_base * 2**attempt)
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
            self._sleep(delay)
        else:  # pragma: no cover - loop always breaks or raises
            raise last or LMError("unreachable")

        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LMError(f"malformed response: {str(data)[:300]}") from exc
        usage = data.get("usage") or {}
        return Completion(
            text,
            int(usage.get("prompt_tokens") or approx_tokens(prompt)),
            int(usage.get("completion_tokens") or approx_tokens(text)),
            self.model,
        )
