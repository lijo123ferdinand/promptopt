import threading
import time

import pytest

from promptopt.lm import (Completion, JsonlCache, LMError, MemoryCache, MeteredLM, Price, Usage,
                          load_prices, price_for)


class Counting:
    def __init__(self, model="claude-sonnet-5-5", delay=0.0, fail=False):
        self.model, self.delay, self.fail, self.n = model, delay, fail, 0
        self.lock = threading.Lock()

    def complete(self, prompt, *, max_tokens=1024):
        with self.lock:
            self.n += 1
        time.sleep(self.delay)
        if self.fail:
            raise LMError("boom", retryable=True)
        return Completion(prompt.upper(), 1000, 100, self.model)


def test_cache_hits_are_free_and_counted():
    inner = Counting()
    lm = MeteredLM(inner)
    a = lm.complete("hi")
    b = lm.complete("hi")
    assert inner.n == 1 and not a.cached and b.cached and b.text == "HI"
    u = lm.usage()
    assert (u.calls, u.cache_hits, u.prompt_tokens, u.completion_tokens) == (1, 1, 1000, 100)
    assert u.cost_usd == pytest.approx((1000 * 2 + 100 * 10) / 1e6)
    lm.complete("hi", max_tokens=5)  # different params -> different key
    assert inner.n == 2


def test_single_flight_dedupes_concurrent_identical_calls():
    inner = Counting(delay=0.1)
    lm = MeteredLM(inner)
    threads = [threading.Thread(target=lm.complete, args=("same",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert inner.n == 1
    assert lm.usage().calls == 1 and lm.usage().cache_hits == 7


def test_errors_are_counted_and_not_cached():
    inner = Counting(fail=True)
    lm = MeteredLM(inner)
    for _ in range(2):
        with pytest.raises(LMError):
            lm.complete("x")
    assert inner.n == 2 and lm.usage().errors == 2 and lm.usage().calls == 0


def test_no_cache_mode():
    inner = Counting()
    lm = MeteredLM(inner, cache=None)
    lm.complete("a")
    lm.complete("a")
    assert inner.n == 2


def test_jsonl_cache_persists_and_skips_torn_lines(tmp_path):
    path = tmp_path / "c.jsonl"
    lm = MeteredLM(Counting(), cache=JsonlCache(path))
    lm.complete("persist me")
    with path.open("a") as f:
        f.write('{"key": "torn')  # simulated crash mid-write
    inner = Counting()
    lm2 = MeteredLM(inner, cache=JsonlCache(path))
    assert lm2.complete("persist me").cached and inner.n == 0


def test_prices(tmp_path):
    assert price_for("mock-1", load_prices()) == Price(0, 0)
    assert price_for("unknown-model", load_prices()) == Price(0, 0)
    f = tmp_path / "p.json"
    f.write_text('{"my-model": {"input_per_mtok": 1, "output_per_mtok": 3}}')
    assert price_for("my-model", load_prices(f)).cost(1_000_000, 1_000_000) == 4.0


def test_usage_arithmetic():
    a, b = Usage(3, 1, 10, 5, 0.5, 0), Usage(1, 1, 4, 2, 0.25, 0)
    assert (a - b).calls == 2 and (a + b).total_tokens == 21
    assert MemoryCache().get("nope") is None
