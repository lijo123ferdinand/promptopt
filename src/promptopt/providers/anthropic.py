"""Claude via the official ``anthropic`` SDK (``pip install 'promptopt[anthropic]'``).

Notes that shape this adapter:

* No ``temperature`` is sent. Current Claude models (e.g. Sonnet 5.5) reject
  non-default sampling parameters, and the cache in MeteredLM is what makes
  optimization reproducible, not the sampler.
* Thinking is adaptive by default on current models and thinking tokens count
  against ``max_tokens``, so the default budget is generous.
* ``refusal_fallback`` (on by default) enables Anthropic's server-side fallback
  so a safety-classifier decline is retried on a fallback model inside the same
  call. Turn it off when going through a proxy that does not support the beta.
* The SDK already retries 408/409/429/5xx with backoff; we only translate its
  exceptions into :class:`LMError` with the right ``retryable`` flag.
"""

from __future__ import annotations

import os
from typing import Any

from ..lm import Completion, LMError

DEFAULT_MODEL = "claude-sonnet-5-5"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        client: Any = None,
        refusal_fallback: bool = True,
        default_max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.refusal_fallback = refusal_fallback
        self.default_max_tokens = default_max_tokens
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError("AnthropicProvider needs the SDK: pip install 'promptopt[anthropic]'") from exc
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            # Without an explicit key the SDK resolves credentials itself
            # (ANTHROPIC_API_KEY, auth token, or an `ant auth login` profile).
            client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        self.client = client

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(max_tokens, self.default_max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            if self.refusal_fallback:
                resp = self.client.beta.messages.create(**kwargs, betas=[_FALLBACK_BETA], fallbacks="default")
            else:
                resp = self.client.messages.create(**kwargs)
        except Exception as exc:
            raise _translate(exc) from exc
        if getattr(resp, "stop_reason", None) == "refusal":
            raise LMError("model declined the request (stop_reason=refusal)", retryable=False)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return Completion(text, int(resp.usage.input_tokens), int(resp.usage.output_tokens), self.model)


def _translate(exc: Exception) -> Exception:
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return exc
    if isinstance(exc, anthropic.RateLimitError):
        return LMError(str(exc), retryable=True, status=429)
    if isinstance(exc, anthropic.APIStatusError):
        return LMError(str(exc), retryable=exc.status_code >= 500, status=exc.status_code)
    if isinstance(exc, anthropic.APIConnectionError):
        return LMError(str(exc), retryable=True)
    return exc
