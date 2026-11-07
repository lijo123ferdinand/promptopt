"""Bootstrap few-shot: let the model write its own demonstrations.

1. **Bootstrap.** Run a *teacher* (the program plus a few random labelled train
   examples as demos) over the train set. Keep each trace whose output passes
   the metric: it is a demonstration in the model's own words, verified correct.
   The labelled teacher demos matter: a zero-shot teacher only produces traces
   for inputs it can already solve, which teach nothing new.
2. **Select.** Sample candidate demo subsets from the trace pool, score each on
   the dev set, keep the best. The un-modified program is always candidate 0,
   so the result can never score below the starting point on dev.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from ..evaluate import Metric
from ..lm import LM, LMError
from ..program import Predict
from ..signature import Example
from .base import Ledger, OptimizationResult, Scorer


class BootstrapFewShot:
    name = "bootstrap"

    def __init__(
        self,
        metric: Metric,
        *,
        demos_per_program: int = 4,
        teacher_demos: int = 3,
        num_candidates: int = 10,
        max_traces: int = 32,
        threshold: float = 1.0,
        threads: int = 8,
        seed: int = 0,
    ) -> None:
        self.metric = metric
        self.demos_per_program = demos_per_program
        self.teacher_demos = teacher_demos
        self.num_candidates = num_candidates
        self.max_traces = max_traces
        self.threshold = threshold
        self.threads = threads
        self.seed = seed

    def bootstrap(self, program: Predict, lm: LM, trainset: Sequence[Example]) -> list[Example]:
        """Return verified traces (inputs + the model's passing outputs)."""
        rng = random.Random(f"bootstrap:{self.seed}")
        jobs = []
        for i, ex in enumerate(trainset):
            others = [x for j, x in enumerate(trainset) if j != i]  # never demo an example to itself
            demos = rng.sample(others, min(self.teacher_demos, len(others)))
            jobs.append((ex, program.with_(demos=demos)))

        def run(job: tuple[Example, Predict]) -> Example | None:
            ex, teacher = job
            try:
                pred = teacher(lm, **ex.inputs)
            except LMError:
                return None
            if self.metric(ex, pred.outputs) >= self.threshold:
                return Example(dict(ex.inputs), dict(pred.outputs))
            return None

        with ThreadPoolExecutor(max_workers=max(1, self.threads)) as pool:
            traces = [t for t in pool.map(run, jobs) if t is not None]
        return traces[: self.max_traces]

    def compile(
        self, program: Predict, *, trainset: Sequence[Example], devset: Sequence[Example], lm: LM
    ) -> OptimizationResult:
        ledger = Ledger(lm)
        score = Scorer(lm, self.metric, devset, threads=self.threads)
        baseline = score(program).score

        traces = self.bootstrap(program, lm, trainset)
        rng = random.Random(f"select:{self.seed}")
        candidates = [program]
        k = min(self.demos_per_program, len(traces))
        seen = {program.program_hash()}
        for _ in range(self.num_candidates * 3):  # bounded retries to find distinct subsets
            if len(candidates) > self.num_candidates or k == 0:
                break
            cand = program.with_(demos=rng.sample(traces, k))
            if cand.program_hash() not in seen:
                seen.add(cand.program_hash())
                candidates.append(cand)

        history = []
        best, best_score = program, baseline
        for i, cand in enumerate(candidates):
            s = score(cand).score
            history.append({"candidate": i, "demos": len(cand.demos), "dev_score": round(s, 4)})
            if s > best_score:  # strict: ties keep the earlier (simpler) program
                best, best_score = cand, s

        best = best.with_(meta={**program.meta, "optimizer": self.name, "dev_score": round(best_score, 4),
                                "traces": len(traces), "candidates": len(candidates)})
        return OptimizationResult(best, best_score, baseline, ledger.usage(), self.name, history)
