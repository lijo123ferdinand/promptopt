"""Shared optimizer plumbing: results, cost ledgers, scored evaluation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..evaluate import EvalResult, Metric, evaluate
from ..lm import LM, MeteredLM, Usage
from ..program import Predict
from ..signature import Example


def _usage(lm: LM) -> Usage:
    return lm.usage() if isinstance(lm, MeteredLM) else Usage()


class Ledger:
    """Cost of one optimization run, across every LM it touched.

    Snapshots the meters at start and reports deltas, so a shared MeteredLM
    (and its warm cache) can be reused across runs without double counting.
    """

    def __init__(self, *lms: LM) -> None:
        self._lms: list[LM] = []
        for lm in lms:
            if lm is not None and all(lm is not x for x in self._lms):
                self._lms.append(lm)
        self._start = [_usage(lm) for lm in self._lms]

    def usage(self) -> Usage:
        total = Usage()
        for lm, start in zip(self._lms, self._start):
            total = total + (_usage(lm) - start)
        return total


@dataclass
class OptimizationResult:
    program: Predict
    dev_score: float
    baseline_dev_score: float
    usage: Usage
    optimizer: str
    history: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        u = self.usage
        return (f"{self.optimizer}: dev {self.baseline_dev_score:.3f} -> {self.dev_score:.3f} | "
                f"{u.calls} LM calls ({u.cache_hits} cached), {u.total_tokens:,} tokens, ${u.cost_usd:.4f}")


class Scorer:
    """Evaluates programs on (prefixes of) a fixed dev set, memoised by program hash.

    Memoising per (program, n) on top of the LM cache keeps successive halving
    and random search from re-parsing hundreds of cached completions, and gives
    one place to count how many candidate evaluations an optimizer performed.
    """

    def __init__(self, lm: LM, metric: Metric, devset: Sequence[Example], *, threads: int = 8) -> None:
        self.lm, self.metric, self.devset, self.threads = lm, metric, list(devset), threads
        self._memo: dict[tuple[str, int], EvalResult] = {}

    def __call__(self, program: Predict, n: int | None = None) -> EvalResult:
        n = len(self.devset) if n is None else min(n, len(self.devset))
        key = (program.program_hash(), n)
        if key not in self._memo:
            self._memo[key] = evaluate(program, self.lm, self.devset[:n], self.metric, threads=self.threads)
        return self._memo[key]
