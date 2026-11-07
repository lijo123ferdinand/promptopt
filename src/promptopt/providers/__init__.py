"""LLM providers. All implement ``complete(prompt, *, max_tokens) -> Completion``."""

from __future__ import annotations

from typing import Any

from ..lm import LM
from .anthropic import AnthropicProvider
from .mock import MockProvider
from .openai_compat import OpenAICompatProvider


def make_provider(kind: str, model: str | None = None, **kwargs: Any) -> LM:
    """Factory used by the CLI: ``mock`` (default, offline), ``anthropic``, ``openai``."""
    if kind == "mock":
        return MockProvider(model or "mock-1", **kwargs)
    if kind == "anthropic":
        return AnthropicProvider(model or "claude-sonnet-5-5", **kwargs)
    if kind in ("openai", "ollama"):
        if not model:
            raise ValueError("--model is required for the openai/ollama provider")
        return OpenAICompatProvider(model, **kwargs)
    raise ValueError(f"unknown provider {kind!r} (mock | anthropic | openai)")


__all__ = ["AnthropicProvider", "MockProvider", "OpenAICompatProvider", "make_provider"]
