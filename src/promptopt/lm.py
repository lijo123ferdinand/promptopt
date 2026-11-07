"""The provider interface, plus the metering/caching wrapper every run goes through.

Providers are intentionally tiny: ``complete(prompt) -> Completion``. Everything
cross-cutting (the (prompt, input) -> output cache, single-flight dedupe of
concurrent identical calls, token and cost accounting) lives in
:class:`MeteredLM`, so a new provider gets all of it for free.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    cached: bool = False


class LMError(RuntimeError):
    """A provider failure. ``retryable`` lets callers tell 429/5xx from 400s."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@runtime_checkable
class LM(Protocol):
    model: str

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> Completion: ...


def approx_tokens(text: str) -> int:
    """~4 characters per token. Used only when a provider does not report usage
    (the mock, or an OpenAI-compatible server that omits ``usage``)."""
    return max(1, (len(text) + 3) // 4)


# --------------------------------------------------------------------------- pricing


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.input_per_mtok + completion_tokens * self.output_per_mtok) / 1e6


# EXAMPLE prices only. They exist so a run can report a dollar figure; they are
# not authoritative and go stale. The Anthropic rows were taken from Anthropic's
# published list prices as of 2026-10; verify against the provider's pricing
# page and override with ``--prices my_prices.json`` before trusting a number.
EXAMPLE_PRICES: dict[str, Price] = {
    "mock": Price(0.0, 0.0),  # the mock is free; see bench for re-pricing at a real rate
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "claude-haiku-5-5": Price(0.10, 0.50),
    "claude-opus-5-5": Price(4.00, 20.00),
}


def load_prices(path: str | os.PathLike[str] | None = None) -> dict[str, Price]:
    """Example prices, optionally overridden by a JSON file
    ``{"model": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}``."""
    prices = dict(EXAMPLE_PRICES)
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for model, p in raw.items():
            prices[model] = Price(float(p["input_per_mtok"]), float(p["output_per_mtok"]))
    return prices


def price_for(model: str, prices: dict[str, Price]) -> Price:
    if model in prices:
        return prices[model]
    # "mock-xyz" and friends: match on the longest known prefix.
    for name in sorted(prices, key=len, reverse=True):
        if model.startswith(name):
            return prices[name]
    return Price(0.0, 0.0)  # unknown model: report tokens, not a made-up dollar figure


# --------------------------------------------------------------------------- usage


@dataclass
class Usage:
    calls: int = 0          # real provider calls (cache misses)
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    errors: int = 0

    def __sub__(self, other: "Usage") -> "Usage":
        return Usage(
            self.calls - other.calls,
            self.cache_hits - other.cache_hits,
            self.prompt_tokens - other.prompt_tokens,
            self.completion_tokens - other.completion_tokens,
            self.cost_usd - other.cost_usd,
            self.errors - other.errors,
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.calls + other.calls,
            self.cache_hits + other.cache_hits,
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cost_usd + other.cost_usd,
            self.errors + other.errors,
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "errors": self.errors,
        }


# --------------------------------------------------------------------------- caches


class MemoryCache:
    def __init__(self) -> None:
        self._d: dict[str, Completion] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Completion | None:
        with self._lock:
            return self._d.get(key)

    def put(self, key: str, value: Completion) -> None:
        with self._lock:
            self._d[key] = value

    def __len__(self) -> int:
        return len(self._d)


class JsonlCache(MemoryCache):
    """Append-only on-disk cache. Re-running an optimization (or resuming a
    crashed one) replays every prompt it has already paid for at zero cost.

    Append-only JSONL is chosen over sqlite/shelve: concurrent appends of short
    lines are safe enough for a single process, a torn last line is simply
    skipped on load, and the file is greppable when debugging a prompt.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn write from a crash
                self._d[rec["key"]] = Completion(
                    rec["text"], rec["prompt_tokens"], rec["completion_tokens"], rec["model"]
                )

    def put(self, key: str, value: Completion) -> None:
        super().put(key, value)
        rec = {
            "key": key,
            "text": value.text,
            "prompt_tokens": value.prompt_tokens,
            "completion_tokens": value.completion_tokens,
            "model": value.model,
        }
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- metering


@dataclass
class MeteredLM:
    """Wraps a provider with caching, single-flight dedupe and cost accounting.

    Cache key = sha256(model, max_tokens, prompt). The prompt already contains
    the instructions, demos and the input, so this is exactly the
    "(prompt, input) -> output" cache: successive halving re-scoring a survivor
    on a larger prefix of the dev set only pays for the new examples.

    Caching assumes deterministic decoding. That holds for the mock, and is the
    sane choice for real models during optimization too (otherwise you are
    comparing candidates on sampling noise); the OpenAI-compatible provider
    defaults to temperature 0 for that reason.
    """

    lm: LM
    cache: MemoryCache | None = field(default_factory=MemoryCache)
    prices: dict[str, Price] = field(default_factory=lambda: dict(EXAMPLE_PRICES))

    def __post_init__(self) -> None:
        self._usage = Usage()
        self._lock = threading.Lock()
        self._inflight: dict[str, Future[Completion]] = {}
        self.model = self.lm.model

    def usage(self) -> Usage:
        with self._lock:
            return replace(self._usage)

    def _key(self, prompt: str, max_tokens: int) -> str:
        h = hashlib.sha256()
        for part in (self.lm.model, str(max_tokens), prompt):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> Completion:
        key = self._key(prompt, max_tokens)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                with self._lock:
                    self._usage.cache_hits += 1
                return replace(hit, cached=True)

        # Single-flight: with 8 eval threads, two candidates sharing demos can
        # issue the exact same prompt at the same moment. Only one goes to the
        # provider; the other waits on its Future and counts as a cache hit.
        with self._lock:
            fut = self._inflight.get(key)
            owner = fut is None
            if owner:
                fut = Future()
                self._inflight[key] = fut
        assert fut is not None
        if not owner:
            result = fut.result()
            with self._lock:
                self._usage.cache_hits += 1
            return replace(result, cached=True)

        try:
            result = self.lm.complete(prompt, max_tokens=max_tokens)
        except BaseException as exc:
            with self._lock:
                self._usage.errors += 1
                self._inflight.pop(key, None)
            fut.set_exception(exc)
            raise
        cost = price_for(self.lm.model, self.prices).cost(result.prompt_tokens, result.completion_tokens)
        with self._lock:
            self._usage.calls += 1
            self._usage.prompt_tokens += result.prompt_tokens
            self._usage.completion_tokens += result.completion_tokens
            self._usage.cost_usd += cost
            self._inflight.pop(key, None)
        if self.cache is not None:
            self.cache.put(key, result)
        fut.set_result(result)
        return result
