"""Instruction search: failure analysis -> LLM proposals -> successive halving.

Each round:

1. Run the incumbent on a sample of the *train* set and collect failures.
   (Failures come from train, not dev, so the proposer never sees the data
   candidates are selected on.)
2. Ask a proposer LM for ``num_candidates`` instruction rewrites grounded in
   those failures (:mod:`promptopt.proposer`).
3. Race the incumbent + proposals with **successive halving**: score everyone on
   a small prefix of dev, keep the top 1/eta, give survivors a larger prefix,
   repeat until one remains. Bad proposals die after a handful of examples, so
   the budget concentrates on the plausible ones.

Budget-awareness: with ``budget`` evaluations per round and R rungs, rung r
gives each of its S_r survivors ``budget // (R * S_r)`` dev examples (classic
SH allocation). Because rungs use growing *prefixes* of one fixed dev order and
the LM cache is keyed on the exact prompt, a survivor's earlier examples are
free when it is promoted, so real spend is well under the nominal budget.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from ..evaluate import Metric, evaluate
from ..lm import LM
from ..program import Predict
from ..proposer import Failure, propose_instructions
from ..signature import Example
from .base import Ledger, OptimizationResult, Scorer


class InstructionSearch:
    name = "instructions"

    def __init__(
        self,
        metric: Metric,
        *,
        proposer_lm: LM | None = None,
        num_candidates: int = 8,
        rounds: int = 2,
        eta: int = 2,
        budget: int | None = None,
        min_examples: int = 4,
        failure_sample: int = 32,
        max_failures_shown: int = 6,
        threads: int = 8,
        seed: int = 0,
    ) -> None:
        if eta < 2:
            raise ValueError("eta must be >= 2")
        self.metric = metric
        self.proposer_lm = proposer_lm
        self.num_candidates = num_candidates
        self.rounds = rounds
        self.eta = eta
        self.budget = budget
        self.min_examples = min_examples
        self.failure_sample = failure_sample
        self.max_failures_shown = max_failures_shown
        self.threads = threads
        self.seed = seed

    # ------------------------------------------------------------------ pieces

    def failures(self, program: Predict, lm: LM, trainset: Sequence[Example], rng: random.Random) -> list[Failure]:
        sample = rng.sample(list(trainset), min(self.failure_sample, len(trainset)))
        res = evaluate(program, lm, sample, self.metric, threads=self.threads)
        fails = [Failure(ex, p.outputs if p else {}) for ex, s, p in zip(sample, res.scores, res.predictions) if s < 1.0]
        return fails[: self.max_failures_shown]

    def propose(self, program: Predict, lm: LM, trainset: Sequence[Example], rng: random.Random,
                proposer: LM | None = None) -> list[str]:
        fails = self.failures(program, lm, trainset, rng)
        if not fails:
            return []
        return propose_instructions(proposer or self.proposer_lm or lm, program.signature,
                                    program.effective_instructions, fails, self.num_candidates)

    def successive_halving(self, candidates: list[Predict], score: Scorer) -> tuple[Predict, list[dict]]:
        n_dev = len(score.devset)
        n_rungs = max(1, math.ceil(math.log(len(candidates), self.eta))) + 1
        budget = self.budget if self.budget is not None else n_dev * n_rungs
        survivors = list(range(len(candidates)))
        rungs: list[dict] = []
        while True:
            m = min(n_dev, max(self.min_examples, budget // (n_rungs * len(survivors))))
            scored = [(score(candidates[i], m).score, i) for i in survivors]
            rungs.append({"examples": m, "scores": {i: round(s, 4) for s, i in scored}})
            if len(survivors) == 1:
                break
            keep = max(1, math.ceil(len(survivors) / self.eta))
            # Sort by score desc, then index asc: on ties the incumbent (index 0)
            # and earlier proposals win, which biases toward not changing things.
            survivors = [i for _, i in sorted(scored, key=lambda t: (-t[0], t[1]))[:keep]]
        return candidates[survivors[0]], rungs

    # ------------------------------------------------------------------ compile

    def compile(
        self, program: Predict, *, trainset: Sequence[Example], devset: Sequence[Example], lm: LM
    ) -> OptimizationResult:
        ledger = Ledger(lm, self.proposer_lm)
        score = Scorer(lm, self.metric, devset, threads=self.threads)
        rng = random.Random(f"instructions:{self.seed}")
        baseline = score(program).score
        incumbent, inc_score = program, baseline
        history = []
        for r in range(self.rounds):
            proposals = self.propose(incumbent, lm, trainset, rng)
            uniq = list(dict.fromkeys(p for p in proposals if p != incumbent.effective_instructions))
            if not uniq:
                history.append({"round": r, "proposals": 0, "note": "no failures or no new proposals"})
                break
            cands = [incumbent] + [incumbent.with_(instructions=p) for p in uniq]
            winner, rungs = self.successive_halving(cands, score)
            win_score = score(winner).score  # full dev, always, before accepting
            accepted = win_score > inc_score
            history.append({"round": r, "proposals": len(uniq), "rungs": rungs,
                            "winner_dev": round(win_score, 4), "accepted": accepted})
            if accepted:
                incumbent, inc_score = winner, win_score
        best = incumbent.with_(meta={**program.meta, "optimizer": self.name, "dev_score": round(inc_score, 4)})
        return OptimizationResult(best, inc_score, baseline, ledger.usage(), self.name, history)
