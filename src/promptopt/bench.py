"""Offline benchmark: every bundled task x every optimizer x several data seeds.

Everything runs against the MockProvider (a simulation, see
:mod:`promptopt.tasks.base`). Each optimizer run gets a *fresh* cache so its
cost is not subsidised by an earlier run, and held-out test scoring uses a
separate meter so it does not count as optimization cost.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from .lm import MeteredLM, Price
from .optimizers import BootstrapFewShot, InstructionSearch, RandomSearch
from .program import Predict
from .providers.mock import MockProvider
from .report import holdout_report
from .tasks import TASKS

# Token counts from the mock are ~real prompt sizes (prompt length / 4), so we
# re-price them at an example real-model rate to give the cost a scale.
REFERENCE_MODEL = "claude-sonnet-5-5"
REFERENCE_PRICE = Price(2.00, 10.00)  # example list price as of 2026-10; verify before quoting


@dataclass
class Row:
    task: str
    optimizer: str
    seed: int
    baseline_test: float
    optimized_test: float
    baseline_dev: float
    optimized_dev: float
    calls: int
    tokens: int
    est_cost_usd: float
    seconds: float
    demos: int
    instr_words: int
    warning: str | None


def make_optimizer(name: str, metric, seed: int):
    if name == "bootstrap":
        return BootstrapFewShot(metric, seed=seed)
    if name == "instructions":
        return InstructionSearch(metric, seed=seed)
    if name == "random_search":
        return RandomSearch(metric, seed=seed)
    raise ValueError(name)


def run(tasks: list[str] | None = None, optimizers: list[str] | None = None, seeds: int = 3,
        n_train: int = 48, n_dev: int = 64, n_test: int = 300) -> list[Row]:
    rows = []
    for tname in tasks or list(TASKS):
        task = TASKS[tname]
        for seed in range(seeds):
            split = task.make_split(seed, n_train, n_dev, n_test)
            baseline = Predict(task.signature)
            for oname in optimizers or ["bootstrap", "instructions", "random_search"]:
                lm = MeteredLM(MockProvider(seed=seed))
                t0 = time.perf_counter()
                result = make_optimizer(oname, task.metric, seed).compile(
                    baseline, trainset=split.train, devset=split.dev, lm=lm)
                secs = time.perf_counter() - t0
                test_lm = MeteredLM(MockProvider(seed=seed))
                rep = holdout_report(baseline, result, split.test, test_lm, task.metric, devset=split.dev)
                u = result.usage
                rows.append(Row(
                    tname, oname, seed, rep.baseline_test, rep.optimized_test, rep.baseline_dev,
                    rep.optimized_dev, u.calls, u.total_tokens,
                    REFERENCE_PRICE.cost(u.prompt_tokens, u.completion_tokens), secs,
                    len(result.program.demos), len(result.program.effective_instructions.split()), rep.warning,
                ))
    return rows


def markdown(rows: list[Row]) -> str:
    """Mean over seeds, with min-max of the optimized test score."""
    out = [
        "| task | optimizer | baseline test | optimized test (min-max) | dev (opt) | LM calls | tokens | est. cost* | wall s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    groups: dict[tuple[str, str], list[Row]] = {}
    for r in rows:
        groups.setdefault((r.task, r.optimizer), []).append(r)
    for (task, opt), rs in groups.items():
        mean = statistics.fmean
        out.append(
            f"| {task} | {opt} | {mean(r.baseline_test for r in rs):.3f} | "
            f"**{mean(r.optimized_test for r in rs):.3f}** ({min(r.optimized_test for r in rs):.3f}-"
            f"{max(r.optimized_test for r in rs):.3f}) | {mean(r.optimized_dev for r in rs):.3f} | "
            f"{mean(r.calls for r in rs):,.0f} | {mean(r.tokens for r in rs):,.0f} | "
            f"${mean(r.est_cost_usd for r in rs):.3f} | {mean(r.seconds for r in rs):.2f} |"
        )
    warnings = sorted({f"{r.task}/{r.optimizer}/seed{r.seed}: {r.warning}" for r in rows if r.warning})
    out.append("")
    out.append(f"*tokens re-priced at example {REFERENCE_MODEL} rates "
               f"(${REFERENCE_PRICE.input_per_mtok}/${REFERENCE_PRICE.output_per_mtok} per MTok).")
    out.append(f"Overfitting warnings: {len(warnings)}" + ("".join(f"\n- {w}" for w in warnings)))
    return "\n".join(out)
