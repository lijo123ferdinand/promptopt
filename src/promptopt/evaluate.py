"""Parallel evaluation of a program against a dataset and a metric."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .lm import LM, LMError
from .program import Predict, Prediction
from .signature import Example

# A metric scores one prediction against its example, in [0, 1]. Returning a
# float (not a bool) lets tasks like entity extraction use F1, while
# optimizers that need a pass/fail decision apply their own threshold.
Metric = Callable[[Example, dict[str, str]], float]


@dataclass(frozen=True)
class EvalResult:
    scores: tuple[float, ...]
    predictions: tuple[Prediction | None, ...]
    errors: int = 0

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def score(self) -> float:
        return sum(self.scores) / self.n if self.n else 0.0

    @property
    def stderr(self) -> float:
        """Standard error of the mean. Reported next to every score so a reader
        can tell a real improvement from a lucky dev split."""
        if self.n < 2:
            return 0.0
        m = self.score
        var = sum((s - m) ** 2 for s in self.scores) / (self.n - 1)
        return math.sqrt(var / self.n)


def _run_one(program: Predict, lm: LM, ex: Example, metric: Metric) -> tuple[float, Prediction | None, bool]:
    try:
        pred = program(lm, **ex.inputs)
    except LMError:
        # A failed call scores 0 rather than aborting a 300-example run; the
        # error count is surfaced so a flaky provider can't hide in the mean.
        return 0.0, None, True
    return float(metric(ex, pred.outputs)), pred, False


def evaluate(
    program: Predict,
    lm: LM,
    dataset: Sequence[Example],
    metric: Metric,
    *,
    threads: int = 8,
) -> EvalResult:
    """Score ``program`` on ``dataset``. Results keep dataset order.

    Threads (not asyncio) because providers are blocking urllib/SDK calls and
    the work is I/O bound; the mock is CPU-trivial so threads cost nothing.
    """
    if not dataset:
        return EvalResult((), ())
    if threads <= 1:
        rows = [_run_one(program, lm, ex, metric) for ex in dataset]
    else:
        with ThreadPoolExecutor(max_workers=min(threads, len(dataset))) as pool:
            rows = list(pool.map(lambda ex: _run_one(program, lm, ex, metric), dataset))
    return EvalResult(
        scores=tuple(r[0] for r in rows),
        predictions=tuple(r[1] for r in rows),
        errors=sum(1 for r in rows if r[2]),
    )
