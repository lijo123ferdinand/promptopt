"""Overfitting guard: score baseline and optimized programs on a held-out test set.

Optimizers select on dev, so the dev score of the winner is biased upward
(you picked the max of many noisy estimates). The test set is never shown to
an optimizer; this report is the number to believe.
"""

from __future__ import annotations

import math

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .evaluate import Metric, evaluate
from .lm import LM
from .optimizers.base import OptimizationResult
from .program import Predict
from .signature import Example


@dataclass(frozen=True)
class HoldoutReport:
    optimizer: str
    baseline_dev: float
    optimized_dev: float
    baseline_test: float
    optimized_test: float
    test_n: int
    test_stderr: float
    warning: str | None

    @property
    def dev_gain(self) -> float:
        return self.optimized_dev - self.baseline_dev

    @property
    def test_gain(self) -> float:
        return self.optimized_test - self.baseline_test

    def to_dict(self) -> dict:
        return {**asdict(self), "dev_gain": round(self.dev_gain, 4), "test_gain": round(self.test_gain, 4)}

    def __str__(self) -> str:
        s = (f"[{self.optimizer}] dev {self.baseline_dev:.3f} -> {self.optimized_dev:.3f} | "
             f"held-out test (n={self.test_n}) {self.baseline_test:.3f} -> {self.optimized_test:.3f} "
             f"(+/- {self.test_stderr:.3f})")
        return s + (f"\n  WARNING: {self.warning}" if self.warning else "")


def holdout_report(
    baseline: Predict,
    result: OptimizationResult,
    testset: Sequence[Example],
    lm: LM,
    metric: Metric,
    *,
    devset: Sequence[Example] | None = None,
    threads: int = 8,
) -> HoldoutReport:
    """Score baseline and optimized programs on ``testset`` and check for overfitting.

    The check compares the optimized program's dev score (what selection saw)
    with its test score (unbiased). Selecting the max of many noisy dev
    estimates inflates the dev number ("winner's curse"); a gap larger than
    two standard errors of the difference is flagged. Pass ``devset`` to get
    the dev standard error from the data; otherwise only test error is used.
    """
    base = evaluate(baseline, lm, testset, metric, threads=threads)
    opt = evaluate(result.program, lm, testset, metric, threads=threads)
    se_dev = evaluate(result.program, lm, devset, metric, threads=threads).stderr if devset else 0.0
    slack = max(0.03, 2 * math.sqrt(se_dev**2 + opt.stderr**2))
    warning = None
    if opt.score < base.score - 2 * max(opt.stderr, base.stderr, 0.01):
        warning = "optimized program is worse than the baseline on held-out data"
    elif result.dev_score - opt.score > slack:
        warning = (f"dev score {result.dev_score:.3f} exceeds held-out test score {opt.score:.3f} by more "
                   f"than {slack:.3f} (2 SE): likely overfit to the dev set; use a larger dev set or fewer trials")
    return HoldoutReport(result.optimizer, result.baseline_dev_score, result.dev_score,
                         base.score, opt.score, opt.n, opt.stderr, warning)
