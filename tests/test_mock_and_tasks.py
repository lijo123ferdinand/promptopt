import random

import pytest

from promptopt import Example, MeteredLM, MockProvider, Predict, Signature, evaluate
from promptopt.proposer import Failure, parse_proposals, render_meta_prompt
from promptopt.tasks import ENTITIES, MATH, SENTIMENT, TASKS
from promptopt.tasks.entities import parse_entities
from promptopt.tasks.math_word import parse_number


@pytest.mark.parametrize("task", list(TASKS.values()), ids=list(TASKS))
def test_splits_are_disjoint_and_gold_scores_perfectly(task):
    sp = task.make_split(1, 30, 30, 60)
    keys = [e.key() for e in sp.train + sp.dev + sp.test]
    assert len(keys) == len(set(keys)) == 120
    for ex in sp.test:
        assert task.metric(ex, ex.outputs) == 1.0
    assert task.make_split(1, 30, 30, 60) == sp  # deterministic


@pytest.mark.parametrize("task", list(TASKS.values()), ids=list(TASKS))
def test_each_phenomenon_has_a_naive_mistake(task):
    sp = task.make_split(0, 0, 0, 200)
    for ph in task.skill.all_phenomena():
        exs = [e for e in sp.test if ph in task.skill.phenomena(e.inputs[task.input_field])]
        assert exs, f"generator never produced {ph}"
        naive = task.skill.all_phenomena() - {ph}
        wrong = sum(task.metric(e, task.skill.solve(e.inputs[task.input_field], naive)) < 1 for e in exs)
        assert wrong / len(exs) > 0.9


@pytest.mark.parametrize("task", list(TASKS.values()), ids=list(TASKS))
def test_hints_contain_keywords_and_decoys_do_not(task):
    sk = task.skill
    for ph, hints in sk.hints.items():
        for h in hints:
            assert any(k in h.lower() for k in sk.keywords[ph]), h
    for d in (*sk.decoys, task.signature.instructions):
        assert not any(k in d.lower() for kws in sk.keywords.values() for k in kws), d


def test_mock_is_deterministic():
    p = Predict(SENTIMENT.signature).prompt({"text": "The camera was not good."})
    a, b = MockProvider().complete(p), MockProvider().complete(p)
    assert a == b and a.text in ("positive", "negative", "neutral")


def test_instruction_keyword_fixes_the_phenomenon():
    lm = MockProvider(noise=0)
    base = Predict(SENTIMENT.signature)
    text = "The camera was not good."
    assert base(lm, text=text).outputs["sentiment"] == "positive"  # naive mistake
    fixed = base.with_(instructions="Classify sentiment. Handle negation carefully.")
    assert fixed(lm, text=text).outputs["sentiment"] == "negative"


def test_correct_demo_teaches_but_wrong_demo_does_not():
    lm = MockProvider(noise=0)
    base = Predict(SENTIMENT.signature)
    text = "The screen was never bad."
    good = Example({"text": "The battery was not great."}, {"sentiment": "negative"})
    bad = Example({"text": "The battery was not great."}, {"sentiment": "positive"})
    assert base.with_(demos=[good])(lm, text=text).outputs["sentiment"] == "positive"
    assert base.with_(demos=[bad])(lm, text=text).outputs["sentiment"] == "negative"  # naive


def test_too_many_demos_adds_noise():
    sp = MATH.make_split(0, 40, 0, 200)
    lm = MeteredLM(MockProvider())
    hints = " ".join(h[0] for h in MATH.skill.hints.values())
    good = Predict(MATH.signature, instructions=MATH.signature.instructions + " " + hints)
    few = evaluate(good, lm, sp.test, MATH.metric).score
    many = evaluate(good.with_(demos=sp.train[:16]), lm, sp.test, MATH.metric).score
    assert few > many + 0.05


def test_mock_proposer_reads_failures():
    sp = ENTITIES.make_split(0, 0, 0, 100)
    fails = [Failure(e, {"entities": "none"}) for e in sp.test
             if ENTITIES.skill.phenomena(e.inputs["sentence"]) == {"ambiguity"}][:5]
    assert fails
    meta = render_meta_prompt(ENTITIES.signature, "Extract entities.", fails, 6)
    props = parse_proposals(MockProvider().complete(meta).text)
    # Duplicate proposals are collapsed, so fewer than 6 distinct ones is normal.
    assert 1 <= len(props) <= 6 and all(p.startswith("Extract entities.") for p in props)
    assert sum("ambigu" in p.lower() for p in props) >= 2


def test_parse_proposals_is_lenient():
    text = "Sure!\n1. Instruction: A\n- Instruction: B\nInstruction: A\ninstruction:  C  \nnoise"
    assert parse_proposals(text) == ["A", "B", "C"]


def test_unknown_signature_still_runs_offline():
    sig = Signature.parse("poem -> critique")
    out = Predict(sig)(MockProvider(), poem="roses")
    assert out.outputs["critique"].startswith("[mock")


def test_metric_parsers():
    assert parse_entities("Alice Chen (PERSON); Pune (location)") == {("Alice Chen", "PERSON"), ("Pune", "LOCATION")}
    assert ENTITIES.metric(Example({}, {"entities": "A (PERSON); B (ORG)"}), {"entities": "A (PERSON)"}) == pytest.approx(2 / 3)
    assert ENTITIES.metric(Example({}, {"entities": "none"}), {"entities": "none"}) == 1.0
    assert float(parse_number("The answer is $1,250.50 dollars")) == 1250.5
    assert parse_number("no digits") is None
    assert MATH.metric(Example({}, {"answer": "2.5"}), {"answer": "2.50 hours"}) == 1.0
    assert SENTIMENT.metric(Example({}, {"sentiment": "negative"}), {"sentiment": " Negative."}) == 1.0


def test_corrupt_produces_wrong_answers():
    rng = random.Random(0)
    for task in TASKS.values():
        sp = task.make_split(0, 0, 0, 30)
        for ex in sp.test:
            text = ex.inputs[task.input_field]
            bad = task.skill.corrupt(text, ex.outputs, rng)
            if task is not ENTITIES or len(parse_entities(ex.outputs["entities"])) > 0:
                assert task.metric(ex, bad) < 1.0
