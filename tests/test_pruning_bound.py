"""Bound pruning must be exact: whenever a trial is pruned, its full-dev score
really could not have beaten the incumbent."""

from promptopt import MeteredLM, MockProvider, Predict, RandomSearch
from promptopt.optimizers.base import Scorer
from promptopt.tasks import SENTIMENT


def test_pruned_trials_could_not_have_won():
    sp = SENTIMENT.make_split(2, 30, 48, 0)
    lm = MeteredLM(MockProvider())
    score = Scorer(lm, SENTIMENT.metric, sp.dev)
    rs = RandomSearch(SENTIMENT.metric, chunk=8)
    base = Predict(SENTIMENT.signature)
    hints = [h[0] for h in SENTIMENT.skill.hints.values()]
    best = score(base.with_(instructions=" ".join([base.signature.instructions, *hints]))).score
    checked = 0
    for i in range(12):
        cand = base.with_(instructions=f"{base.signature.instructions} {hints[i % 3]} v{i}", demos=sp.train[i:i + i % 4])
        s, used, pruned = rs._score_with_pruning(cand, score, best)
        if pruned:
            checked += 1
            assert used < len(sp.dev)
            assert score(cand).score <= best
    assert checked > 0
