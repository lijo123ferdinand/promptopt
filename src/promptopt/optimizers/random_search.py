"""Random search over (instruction, demo subset) with two kinds of early stopping.

The search space is the cross product of an instruction pool (proposer output)
and demo subsets drawn from a bootstrapped trace pool. Random search is a
strong, honest baseline for small discrete spaces and parallelises trivially.

Early stopping:

* **Per trial (bound pruning).** Score in chunks; after each chunk the best
  score the trial could still reach is ``(points so far + examples left) / n``.
  Once that cannot beat the incumbent, stop. This is exact: it never discards
  a configuration that could have won, it only skips work that cannot matter.
* **Per search (patience).** Stop after ``patience`` consecutive trials with
  no improvement.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from ..evaluate import Metric
from ..lm import LM
from ..program import Predict
from ..signature import Example
from .base import Ledger, OptimizationResult, Scorer
from .bootstrap import BootstrapFewShot
from .instructions import InstructionSearch


class RandomSearch:
    name = "random_search"

    def __init__(
        self,
        metric: Metric,
        *,
        demo_pool: Sequence[Example] | None = None,
        instructions: Sequence[str] | None = None,
        proposer_lm: LM | None = None,
        num_trials: int = 24,
        max_demos: int = 4,
        patience: int = 10,
        chunk: int = 8,
        num_proposals: int = 8,
        threads: int = 8,
        seed: int = 0,
    ) -> None:
        self.metric = metric
        self.demo_pool = list(demo_pool) if demo_pool is not None else None
        self.instructions = list(instructions) if instructions is not None else None
        self.proposer_lm = proposer_lm
        self.num_trials = num_trials
        self.max_demos = max_demos
        self.patience = patience
        self.chunk = chunk
        self.num_proposals = num_proposals
        self.threads = threads
        self.seed = seed

    def _score_with_pruning(self, cand: Predict, score: Scorer, best: float) -> tuple[float, int, bool]:
        """Returns (score, examples used, pruned)."""
        n = len(score.devset)
        m = 0
        while m < n:
            m = min(n, m + self.chunk)
            res = score(cand, m)
            upper = (sum(res.scores) + (n - m)) / n
            if m < n and upper <= best:
                return res.score, m, True
        return score(cand).score, n, False

    def compile(
        self, program: Predict, *, trainset: Sequence[Example], devset: Sequence[Example], lm: LM
    ) -> OptimizationResult:
        ledger = Ledger(lm, self.proposer_lm)
        rng = random.Random(f"random_search:{self.seed}")
        score = Scorer(lm, self.metric, devset, threads=self.threads)

        pool = self.demo_pool
        if pool is None:  # build the demo pool from bootstrapped traces
            pool = BootstrapFewShot(self.metric, threads=self.threads, seed=self.seed).bootstrap(program, lm, trainset)
        instrs = self.instructions
        if instrs is None:  # one round of failure-grounded proposals
            proposer = InstructionSearch(self.metric, proposer_lm=self.proposer_lm,
                                         num_candidates=self.num_proposals, threads=self.threads, seed=self.seed)
            instrs = proposer.propose(program, lm, trainset, rng)
        instrs = list(dict.fromkeys([program.effective_instructions, *instrs]))

        baseline = score(program).score
        best, best_score = program, baseline
        seen = {program.program_hash()}
        history: list[dict] = []
        stale = 0
        for t in range(self.num_trials):
            k = rng.randint(0, min(self.max_demos, len(pool)))
            cand = program.with_(instructions=rng.choice(instrs), demos=rng.sample(pool, k) if k else ())
            h = cand.program_hash()
            if h in seen:
                continue
            seen.add(h)
            s, used, pruned = self._score_with_pruning(cand, score, best_score)
            improved = not pruned and s > best_score
            history.append({"trial": t, "demos": k, "examples": used, "pruned": pruned, "dev_score": round(s, 4)})
            if improved:
                best, best_score, stale = cand, s, 0
            else:
                stale += 1
                if stale >= self.patience:
                    history.append({"trial": t, "note": f"early stop: no improvement in {self.patience} trials"})
                    break

        best = best.with_(meta={**program.meta, "optimizer": self.name, "dev_score": round(best_score, 4),
                                "pool": len(pool), "instructions": len(instrs)})
        return OptimizationResult(best, best_score, baseline, ledger.usage(), self.name, history)
