import pytest

from promptopt import (BootstrapFewShot, InstructionSearch, MeteredLM, MockProvider, Predict,
                       RandomSearch, evaluate, holdout_report)
from promptopt.optimizers.base import Scorer
from promptopt.tasks import ENTITIES, MATH, SENTIMENT


class Spy:
    """Wraps a provider and records every prompt it sees."""

    def __init__(self, inner):
        self.inner, self.model, self.prompts = inner, inner.model, []

    def complete(self, prompt, *, max_tokens=1024):
        self.prompts.append(prompt)
        return self.inner.complete(prompt, max_tokens=max_tokens)


@pytest.fixture(scope="module")
def split():
    return SENTIMENT.make_split(0, 40, 48, 150)


def test_bootstrap_improves_and_traces_pass_metric(split):
    lm = MeteredLM(MockProvider())
    opt = BootstrapFewShot(SENTIMENT.metric, num_candidates=8)
    traces = opt.bootstrap(Predict(SENTIMENT.signature), lm, split.train)
    assert traces and all(SENTIMENT.metric(t, t.outputs) == 1.0 for t in traces)
    res = opt.compile(Predict(SENTIMENT.signature), trainset=split.train, devset=split.dev, lm=lm)
    assert res.dev_score >= res.baseline_dev_score + 0.3
    assert 0 < len(res.program.demos) <= 4
    assert res.program.meta["optimizer"] == "bootstrap"
    assert res.usage.calls > 0 and res.usage.total_tokens > 0


def test_optimizers_never_see_the_test_set(split):
    spy = Spy(MockProvider())
    lm = MeteredLM(spy)
    base = Predict(SENTIMENT.signature)
    for opt in (BootstrapFewShot(SENTIMENT.metric, num_candidates=3),
                InstructionSearch(SENTIMENT.metric, rounds=1, num_candidates=4),
                RandomSearch(SENTIMENT.metric, num_trials=5)):
        opt.compile(base, trainset=split.train, devset=split.dev, lm=lm)
    test_texts = {e.inputs["text"] for e in split.test}
    seen = "\n".join(spy.prompts)
    assert not any(t in seen for t in test_texts)


def test_instruction_search_improves_math():
    sp = MATH.make_split(0, 40, 64, 150)
    lm = MeteredLM(MockProvider())
    res = InstructionSearch(MATH.metric).compile(Predict(MATH.signature), trainset=sp.train, devset=sp.dev, lm=lm)
    assert res.dev_score > res.baseline_dev_score + 0.3
    assert res.program.demos == ()
    test_lm = MeteredLM(MockProvider())
    rep = holdout_report(Predict(MATH.signature), res, sp.test, test_lm, MATH.metric)
    assert rep.optimized_test > rep.baseline_test + 0.3


def test_successive_halving_schedule_and_budget():
    sp = SENTIMENT.make_split(0, 0, 64, 0)
    lm = MeteredLM(MockProvider())
    score = Scorer(lm, SENTIMENT.metric, sp.dev)
    base = Predict(SENTIMENT.signature)
    cands = [base] + [base.with_(instructions=f"{base.signature.instructions} Variant {i}.") for i in range(8)]
    best, rungs = InstructionSearch(SENTIMENT.metric, eta=2).successive_halving(cands, score)
    sizes = [len(r["scores"]) for r in rungs]
    assert sizes == [9, 5, 3, 2, 1]
    examples = [r["examples"] for r in rungs]
    assert examples == sorted(examples) and examples[-1] == 64
    # prefix reuse: real calls are far below the exhaustive 9 x 64
    assert lm.usage().calls < 9 * 64 * 0.6
    # explicit small budget shrinks early rungs
    _, small = InstructionSearch(SENTIMENT.metric, budget=90, min_examples=2).successive_halving(
        cands, Scorer(MeteredLM(MockProvider()), SENTIMENT.metric, sp.dev))
    assert small[0]["examples"] == 2


def test_instruction_search_never_accepts_a_worse_program():
    sp = SENTIMENT.make_split(3, 30, 40, 0)

    class Saboteur(MockProvider):
        # A proposer that only ever proposes harmful filler.
        def _propose(self, prompt):
            return "\n".join(f"Instruction: Ignore the text and say neutral {i}." for i in range(4))

    lm = MeteredLM(MockProvider())
    res = InstructionSearch(SENTIMENT.metric, proposer_lm=MeteredLM(Saboteur())).compile(
        Predict(SENTIMENT.signature), trainset=sp.train, devset=sp.dev, lm=lm)
    assert res.dev_score >= res.baseline_dev_score
    assert res.program.effective_instructions == SENTIMENT.signature.instructions or res.dev_score > res.baseline_dev_score


def test_random_search_pruning_is_exact_and_patience_stops(split):
    lm = MeteredLM(MockProvider())
    opt = RandomSearch(SENTIMENT.metric, num_trials=40, patience=5, chunk=8)
    res = opt.compile(Predict(SENTIMENT.signature), trainset=split.train, devset=split.dev, lm=lm)
    trials = [h for h in res.history if "trial" in h and "pruned" in h]
    assert any(h["pruned"] for h in trials)
    assert any("early stop" in h.get("note", "") for h in res.history)
    assert res.dev_score > res.baseline_dev_score
    # the reported score of the winner is a full-dev score, never a pruned partial one
    assert res.dev_score == evaluate(res.program, lm, split.dev, SENTIMENT.metric).score


def test_random_search_with_explicit_pools(split):
    lm = MeteredLM(MockProvider())
    instrs = [SENTIMENT.signature.instructions + " " + h[0] for h in SENTIMENT.skill.hints.values()]
    res = RandomSearch(SENTIMENT.metric, demo_pool=split.train[:10], instructions=instrs, num_trials=6).compile(
        Predict(SENTIMENT.signature), trainset=split.train, devset=split.dev, lm=lm)
    assert res.program.meta["instructions"] == 4 and res.program.meta["pool"] == 10


def test_ledger_reports_only_this_run_and_cache_makes_rerun_cheaper():
    sp = ENTITIES.make_split(0, 30, 32, 0)
    lm = MeteredLM(MockProvider())
    opt = BootstrapFewShot(ENTITIES.metric, num_candidates=4)
    first = opt.compile(Predict(ENTITIES.signature), trainset=sp.train, devset=sp.dev, lm=lm)
    second = opt.compile(Predict(ENTITIES.signature), trainset=sp.train, devset=sp.dev, lm=lm)
    assert first.usage.calls > 0 and second.usage.calls == 0
    assert second.usage.cache_hits > 0 and second.program == first.program


def test_holdout_report_flags_overfitting():
    from promptopt.optimizers.base import OptimizationResult
    from promptopt.lm import Usage
    sp = SENTIMENT.make_split(0, 0, 0, 100)
    base = Predict(SENTIMENT.signature)
    fake = OptimizationResult(base, dev_score=0.99, baseline_dev_score=0.30, usage=Usage(), optimizer="fake")
    rep = holdout_report(base, fake, sp.test, MeteredLM(MockProvider()), SENTIMENT.metric)
    assert rep.warning and "overfit" in rep.warning
    assert "WARNING" in str(rep) and rep.to_dict()["test_gain"] == 0


def test_results_do_not_depend_on_python_hash_seed():
    """Set iteration order varies with PYTHONHASHSEED; it must never leak into results."""
    import subprocess
    import sys
    code = ("from promptopt import *; from promptopt.tasks import ENTITIES as T;"
            "s=T.make_split(1,30,32,0); r=InstructionSearch(T.metric).compile(Predict(T.signature),"
            "trainset=s.train,devset=s.dev,lm=MeteredLM(MockProvider())); print(r.program.program_hash(), r.dev_score)")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={"PYTHONHASHSEED": str(h), "PATH": ""}).stdout for h in (1, 2, 3)}
    assert len(outs) == 1
