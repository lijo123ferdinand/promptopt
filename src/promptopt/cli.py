"""``promptopt`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from . import bench
from .evaluate import evaluate
from .lm import JsonlCache, MemoryCache, MeteredLM, load_prices
from .optimizers import OPTIMIZERS
from .program import Predict, diff_programs
from .providers import make_provider
from .report import holdout_report
from .tasks import TASKS, get_task


def _lm(args: argparse.Namespace) -> MeteredLM:
    kwargs = {}
    if args.provider in ("openai", "ollama") and args.base_url:
        kwargs["base_url"] = args.base_url
    if args.provider == "mock":
        kwargs["seed"] = args.seed
    provider = make_provider(args.provider, args.model, **kwargs)
    cache = JsonlCache(args.cache) if args.cache else MemoryCache()
    return MeteredLM(provider, cache=cache, prices=load_prices(args.prices))


def _add_lm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default="mock", help="mock (default, offline) | anthropic | openai")
    p.add_argument("--model", default=None, help="model id (default: mock-1 / claude-sonnet-5-5)")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (default: local Ollama)")
    p.add_argument("--cache", default=None, help="JSONL file to persist the (prompt -> output) cache")
    p.add_argument("--prices", default=None, help="JSON file overriding the example price table")
    p.add_argument("--seed", type=int, default=0, help="data split seed (and mock seed)")


def cmd_tasks(_: argparse.Namespace) -> int:
    for t in TASKS.values():
        sig = t.signature
        print(f"{t.name:10s} {', '.join(sig.input_names)} -> {', '.join(sig.output_names)}   "
              f"phenomena: {', '.join(sorted(t.skill.all_phenomena()))}")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    task = get_task(args.task)
    split = task.make_split(args.seed)
    lm = _lm(args)
    baseline = Predict(task.signature)
    opt = OPTIMIZERS[args.optimizer](task.metric, seed=args.seed)
    result = opt.compile(baseline, trainset=split.train, devset=split.dev, lm=lm)
    print(result.summary())
    # Held-out scoring uses its own meter so it is not billed to the optimizer.
    test_lm = _lm(args)
    print(holdout_report(baseline, result, split.test, test_lm, task.metric, devset=split.dev))
    if args.out:
        prog = result.program.with_(meta={**result.program.meta, "task": task.name, "lm": lm.model,
                                          "optimization_usage": result.usage.to_dict()})
        print(f"saved {prog.save(args.out)}  ({prog.program_hash()})")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    task = get_task(args.task)
    split = task.make_split(args.seed)
    prog = Predict.load(args.program) if args.program else Predict(task.signature)
    data = {"train": split.train, "dev": split.dev, "test": split.test}[args.split]
    lm = _lm(args)
    res = evaluate(prog, lm, data, task.metric)
    print(f"{task.name}/{args.split}: {res.score:.3f} +/- {res.stderr:.3f} (n={res.n}, errors={res.errors})")
    print(json.dumps(lm.usage().to_dict()))
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    a, b = Predict.load(args.a), Predict.load(args.b)
    sys.stdout.write(diff_programs(a, b, a_name=args.a, b_name=args.b))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    prog = Predict.load(args.program)
    inputs = {n: (args.input or "<input>") for n in prog.signature.input_names}
    print(prog.prompt(inputs))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    rows = bench.run(tasks=args.tasks, seeds=args.seeds)
    print(bench.markdown(rows))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in rows], f, indent=2)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="promptopt", description="Framework-free automatic prompt optimization.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tasks", help="list bundled tasks").set_defaults(fn=cmd_tasks)

    p = sub.add_parser("optimize", help="optimize a bundled task and report held-out accuracy")
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--optimizer", default="bootstrap", choices=sorted(OPTIMIZERS))
    p.add_argument("--out", default=None, help="write the compiled program JSON here")
    _add_lm_args(p)
    p.set_defaults(fn=cmd_optimize)

    p = sub.add_parser("eval", help="score a program (or the zero-shot baseline) on a split")
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--program", default=None)
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    _add_lm_args(p)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("diff", help="semantic + JSON diff of two compiled programs")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("show", help="print the prompt a compiled program renders")
    p.add_argument("program")
    p.add_argument("--input", default=None)
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("bench", help="offline benchmark of all optimizers on all tasks (mock LM)")
    p.add_argument("--tasks", nargs="*", default=None, choices=sorted(TASKS))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--json", default=None, help="also write per-run rows as JSON")
    p.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
