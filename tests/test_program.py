import json

import pytest

from promptopt import Example, Predict, Signature, diff_programs


def prog(**kw):
    s = Signature.parse("text -> label", "Label it.")
    return Predict(s, **kw)


def test_save_load_roundtrip(tmp_path):
    p = prog(instructions="Be careful.", demos=(Example({"text": "a"}, {"label": "x"}),), meta={"dev_score": 0.9})
    path = p.save(tmp_path / "p.json")
    q = Predict.load(path)
    assert q == p and q.meta == {"dev_score": 0.9}
    assert q.program_hash() == p.program_hash()


def test_json_is_canonical_and_diffable(tmp_path):
    p = prog(instructions="Be careful.")
    text = p.to_json()
    assert text.endswith("\n")
    d = json.loads(text)
    assert d["format"] == "promptopt.program" and d["schema_version"] == 1
    assert list(d) == sorted(d)  # sorted keys -> stable diffs
    assert p.to_json() == prog(instructions="Be careful.").to_json()


def test_hash_ignores_meta_but_tracks_behaviour():
    a = prog(meta={"x": 1})
    assert a.program_hash() == prog(meta={"x": 2}).program_hash()
    assert a.program_hash() != prog(instructions="other").program_hash()
    assert a.program_hash() != prog(demos=(Example({"text": "a"}, {"label": "b"}),)).program_hash()


def test_load_detects_tampering_and_unknown_versions(tmp_path):
    d = prog().to_dict()
    d["instructions"] = "hand edited"
    with pytest.raises(ValueError, match="program_hash"):
        Predict.from_dict(d)
    d = prog().to_dict()
    d["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        Predict.from_dict(d)
    with pytest.raises(ValueError):
        Predict.from_dict({"format": "something-else"})


def test_diff_programs():
    a = prog()
    b = prog(instructions="Better.", demos=(Example({"text": "a"}, {"label": "x"}),), meta={"dev_score": 0.8})
    out = diff_programs(a, b)
    assert "instructions changed" in out and "demos: +1 -0" in out and "b dev_score=0.8" in out
    assert '-  "instructions": "Label it."' in out
    assert "behaviourally identical" in diff_programs(a, prog(meta={"note": 1}))


def test_missing_input_raises():
    with pytest.raises(KeyError):
        prog().prompt({})
