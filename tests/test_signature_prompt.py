import pytest

from promptopt.prompt import parse_completion, parse_fields, render_prompt, split_prompt
from promptopt.signature import Example, Field, Signature


def sig():
    return Signature.parse("question, context -> reasoning, answer", "Answer the question.",
                           descs={"answer": "a number"})


def test_parse_spec_and_prefixes():
    s = sig()
    assert s.input_names == ("question", "context")
    assert s.output_names == ("reasoning", "answer")
    assert s.outputs[1].prefix == "Answer" and s.outputs[1].desc == "a number"


def test_rejects_bad_signatures():
    with pytest.raises(ValueError):
        Signature.parse("no arrow here")
    with pytest.raises(ValueError):
        Signature((Field("a"),), (Field("a"),))
    with pytest.raises(ValueError):
        Signature((Field("a", prefix="X"),), (Field("b", prefix="x"),))


def test_render_and_split_roundtrip():
    s = sig()
    demo = Example({"question": "q1", "context": "c1"}, {"reasoning": "r1", "answer": "1"})
    p = render_prompt(s, "Do it.", [demo], {"question": "q2", "context": "c2  with\nnewline"})
    instr, fmt, demos, query = split_prompt(p)
    assert instr == "Do it."
    assert "Answer: a number" in fmt and "Context: ${context}" in fmt
    assert demos == ["Question: q1\nContext: c1\nReasoning: r1\nAnswer: 1"]
    assert query.endswith("Context: c2 with newline\nReasoning:")


def test_parse_completion_with_and_without_prefix():
    s = sig()
    assert parse_completion(s, " 3 + 4 = 7\nAnswer: 7") == {"reasoning": "3 + 4 = 7", "answer": "7"}
    assert parse_completion(s, "Reasoning: x\nAnswer: 9") == {"reasoning": "x", "answer": "9"}
    # missing field -> empty, never raises
    assert parse_completion(s, "just text") == {"reasoning": "just text", "answer": ""}


def test_parse_fields_multiline_and_stops_at_invented_next_example():
    text = "Reasoning: line one\nline two\nAnswer: 5\n\n---\n\nQuestion: invented\nAnswer: 99"
    assert parse_fields(text, {"Reasoning": "reasoning", "Answer": "answer"}) == {
        "reasoning": "line one line two", "answer": "5"}
    # a repeated prefix (the model starting another example) does not overwrite
    assert parse_fields("Answer: 1\nAnswer: 2", {"Answer": "answer"}) == {"answer": "1"}


def test_split_prompt_rejects_foreign_text():
    with pytest.raises(ValueError):
        split_prompt("hello")
