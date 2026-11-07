import json

from promptopt.cli import main


def test_cli_end_to_end(tmp_path, capsys):
    assert main(["tasks"]) == 0
    assert "sentiment" in capsys.readouterr().out

    out = tmp_path / "math.json"
    cache = tmp_path / "cache.jsonl"
    assert main(["optimize", "--task", "math", "--optimizer", "instructions", "--out", str(out),
                 "--cache", str(cache)]) == 0
    text = capsys.readouterr().out
    assert "held-out test" in text and "saved" in text
    prog = json.loads(out.read_text())
    assert prog["meta"]["task"] == "math" and prog["meta"]["optimization_usage"]["calls"] > 0
    assert cache.exists()

    assert main(["eval", "--task", "math", "--program", str(out)]) == 0
    assert "math/test:" in capsys.readouterr().out

    assert main(["show", str(out), "--input", "Ana has 3 apples and buys 2 more."]) == 0
    assert "Follow the following format." in capsys.readouterr().out

    base = tmp_path / "base.json"
    from promptopt import Predict
    from promptopt.tasks import MATH
    Predict(MATH.signature).save(base)
    assert main(["diff", str(base), str(out)]) == 0
    assert "instructions changed" in capsys.readouterr().out


def test_cli_bench_small(tmp_path, capsys):
    j = tmp_path / "rows.json"
    assert main(["bench", "--tasks", "sentiment", "--seeds", "1", "--json", str(j)]) == 0
    assert "| sentiment | bootstrap |" in capsys.readouterr().out
    assert len(json.loads(j.read_text())) == 3
