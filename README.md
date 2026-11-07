# promptopt

**DSPy-style automatic prompt optimization in ~2,700 lines of plain Python. You declare the task and the metric, and the optimizer picks the instructions and few-shot demos.**

You describe a task as a *signature* (input fields -> output fields plus an instruction), give it a metric and a train/dev set, and pick an optimizer. You get back a compiled program: a versioned JSON file holding the winning instructions and demonstrations. You can check it into git, review its diff and load it in production. The package depends only on the standard library. Providers are Anthropic, any OpenAI-compatible server (Ollama, vLLM, llama.cpp) and a deterministic offline mock.

```
pip install -e .            # stdlib only
pip install -e '.[anthropic]'   # optional: Claude provider
```

## Why

Hand-tuned prompts tend to break in production. Someone adds "be careful with negation" because of one bug report, nobody measures the effect, and the next edit quietly undoes it. Prompt optimization treats the prompt as a parameter: you write down what "correct" means as a metric, and a search procedure picks the prompt on data.

[DSPy](https://github.com/stanfordnlp/dspy) made the idea popular, but it comes as a framework with modules, its own LM abstraction and teleprompters. promptopt is the core loop with nothing else around it. It is small enough to read in an afternoon and has no dependencies. Each piece is something you would otherwise build yourself before using it in production:

- **Three optimizers:** bootstrap few-shot, failure-driven instruction search with successive halving, and random search over demos × instructions with exact early stopping.
- **A `(prompt, input) -> output` cache** with single-flight dedupe. Re-scoring a survivor, re-running an optimization or resuming after a crash only pays for new prompts.
- **Per-run cost and token accounting** across the task model and the proposer model.
- **An overfitting guard.** A held-out test set is never shown to an optimizer, and its score is reported next to the dev score with a winner's-curse warning.
- **Compiled programs as canonical JSON.** Keys are sorted, the file carries a schema version, and a `program_hash` covers only the parts that change model behaviour. `promptopt diff` shows what changed.

## Quickstart (offline, no API key)

All commands and outputs below were run on this repo (Python 3.13, macOS). The model is the **MockProvider**, a deterministic simulation explained [below](#the-simulation-read-this-before-quoting-a-number).

```console
$ promptopt tasks
sentiment  text -> sentiment   phenomena: contrast, negation, sarcasm
entities   sentence -> entities   phenomena: ambiguity, multiword, sentence_initial
math       problem -> reasoning, answer   phenomena: distractor, multi_step, percent, unit

$ promptopt optimize --task sentiment --optimizer bootstrap --out out/sentiment_bootstrap.json
bootstrap: dev 0.375 -> 0.984 | 752 LM calls (0 cached), 96,323 tokens, $0.0000
[bootstrap] dev 0.375 -> 0.984 | held-out test (n=300) 0.317 -> 0.957 (+/- 0.012)
saved out/sentiment_bootstrap.json  (sha256:61f4a741015e5f62)

$ promptopt optimize --task sentiment --optimizer instructions --out out/sentiment_instr.json
instructions: dev 0.375 -> 0.984 | 396 LM calls (236 cached), 42,549 tokens, $0.0000
[instructions] dev 0.375 -> 0.984 | held-out test (n=300) 0.317 -> 0.940 (+/- 0.014)
  WARNING: dev score 0.984 exceeds held-out test score 0.940 by more than 0.042 (2 SE): likely overfit to the dev set; use a larger dev set or fewer trials
saved out/sentiment_instr.json  (sha256:f765598461acbb07)

$ promptopt diff out/sentiment_bootstrap.json out/sentiment_instr.json
out/sentiment_bootstrap.json: sha256:61f4a741015e5f62  ->  out/sentiment_instr.json: sha256:f765598461acbb07
instructions changed
demos: +0 -4 (now 0)
...unified JSON diff follows...
```

The dollar figure is $0 because the mock is free. The benchmark below re-prices the same token counts at an example real-model rate.

The same thing as a library (`examples/quickstart.py`):

```python
from promptopt import InstructionSearch, MeteredLM, MockProvider, Predict, holdout_report
from promptopt.tasks import MATH

split = MATH.make_split(seed=0)            # 48 train / 64 dev / 300 held-out test
lm = MeteredLM(MockProvider())             # or AnthropicProvider(), OpenAICompatProvider("llama3.2")
baseline = Predict(MATH.signature)

result = InstructionSearch(MATH.metric).compile(baseline, trainset=split.train, devset=split.dev, lm=lm)
print(holdout_report(baseline, result, split.test, MeteredLM(MockProvider()), MATH.metric, devset=split.dev))
result.program.save("out/math_optimized.json")
```

```
instructions: dev 0.281 -> 0.953 | 428 LM calls (213 cached), 50,741 tokens, $0.0000
[instructions] dev 0.281 -> 0.953 | held-out test (n=300) 0.297 -> 0.967 (+/- 0.010)
optimized instructions: Solve the math word problem. Ignore irrelevant numbers such as ages that do not
affect the quantity asked about. Check the unit the question asks for and convert (60 minutes = 1 hour,
100 cents = 1 dollar). Turn every percentage into a fraction of the original amount before subtracting.
Work step-by-step: track the quantity through each operation mentioned.
```

To use your own task, declare a signature and a metric (`examples/custom_task.py`):

```python
signature = Signature(
    inputs=(Field("ticket", "a customer support message"),),
    outputs=(Field("team", "one of: billing, shipping, technical"),),
    instructions="Route the support ticket to the team that should handle it.",
)
def metric(example, outputs): return float(outputs["team"].strip().lower() == example.outputs["team"])
```

Real models:

```console
$ export ANTHROPIC_API_KEY=...            # default model: claude-sonnet-5-5
$ promptopt optimize --task math --optimizer instructions --provider anthropic --cache runs/math.jsonl
$ promptopt optimize --task math --provider openai --model llama3.2   # local Ollama at :11434/v1
```

## Benchmark (measured, simulated model)

Command: `promptopt bench --seeds 5`. It covers 3 tasks × 3 optimizers × 5 data seeds. Each split is 48 train / 64 dev / 300 test, and all three are disjoint. Each cell is the **mean over 5 seeds**, with the min-max of the held-out test score in brackets. Every optimizer run starts with a fresh cache, so no run is subsidised by another. Test scoring is metered separately and is **not** included in the cost.

| task | optimizer | baseline test | optimized test (min-max) | dev (opt) | LM calls | tokens | est. cost* | wall s |
|---|---|---|---|---|---|---|---|---|
| sentiment | bootstrap | 0.353 | **0.797** (0.740-0.957) | 0.850 | 752 | 97,151 | $0.206 | 0.20 |
| sentiment | instructions | 0.353 | **0.954** (0.940-0.970) | 0.984 | 403 | 43,432 | $0.102 | 0.13 |
| sentiment | random_search | 0.353 | **0.917** (0.743-0.973) | 0.966 | 703 | 85,399 | $0.185 | 0.44 |
| entities | bootstrap | 0.566 | **0.980** (0.949-0.991) | 0.986 | 752 | 143,800 | $0.356 | 0.27 |
| entities | instructions | 0.566 | **0.984** (0.952-0.995) | 0.983 | 415 | 53,028 | $0.152 | 0.17 |
| entities | random_search | 0.566 | **0.963** (0.898-0.993) | 0.975 | 644 | 107,286 | $0.276 | 0.42 |
| math | bootstrap | 0.283 | **0.826** (0.733-0.957) | 0.841 | 752 | 146,169 | $0.335 | 0.19 |
| math | instructions | 0.283 | **0.930** (0.803-0.970) | 0.947 | 422 | 49,030 | $0.129 | 0.17 |
| math | random_search | 0.283 | **0.836** (0.650-0.970) | 0.875 | 948 | 146,692 | $0.349 | 0.90 |

\*The mock's token counts are real prompt sizes (characters / 4). They are re-priced at **example** `claude-sonnet-5-5` list rates ($2 / $10 per MTok, as of 2026-10). Verify current prices before quoting a figure, and override them with `--prices prices.json`.

What the numbers say, within the simulation:

- **Instruction search** gave the best held-out score on all three tasks at the lowest cost (~400 calls, about half of bootstrap's). Successive halving discards weak proposals after 4-8 examples, and growing dev prefixes plus the cache make promoting a survivor almost free. In the sentiment quickstart run, 236 of 632 requests were served from cache.
- **Bootstrap few-shot** has high variance (sentiment 0.74-0.96). Four random demos often miss a phenomenon, and the 10 candidate subsets don't always include one that covers everything.
- **Random search** sits in between. Bound pruning and patience cut its cost, but it varies the most from seed to seed (math 0.65-0.97). The search space is larger and 24 trials don't always find the covering combination.
- **The overfitting guard flagged 7 of 45 runs**, where the dev score of the selected program beat its test score by more than 2 standard errors. This is the winner's curse from picking the best of many candidates on 64 dev examples. Read the test column, not the dev column.

Results are bit-for-bit reproducible. A test runs the same optimization under three `PYTHONHASHSEED`s and requires identical output.

### The simulation (read this before quoting a number)

**The benchmark does not use a real LLM.** `MockProvider` is a deterministic model of how LLMs respond to prompts. Without it, the optimizers couldn't be developed, tested and measured offline:

1. Each bundled task is built from named **phenomena** with a systematic naive mistake: negation ("not good" read as positive), sarcasm, contrast clauses, multi-word entity names, ambiguous names, multi-step arithmetic, percentages, unit conversion and distractor numbers.
2. The mock parses the prompt the way the renderer wrote it. A phenomenon counts as *covered* if the instructions mention one of its keywords, or if a **correct** demo exhibits it. A mislabelled demo teaches nothing.
3. It answers with the covered rules applied and the naive rule everywhere else. It then adds 4% hash-seeded noise, plus extra noise when the prompt has more than 6 demos or more than 80 instruction words, so stuffing the prompt with everything doesn't win.
4. As an instruction proposer, it reads the failures in the meta-prompt and writes hint sentences for the phenomena it sees. It is imperfect on purpose: 20% of proposals are generic filler and 10% misdiagnose the problem.

This makes the optimization problem real: the search has to find instructions and demos that cover the phenomena that actually occur, and it is scored on held-out data. The **absolute numbers are properties of the simulation, not of any real model.** They show that the search, the caching, the budget logic and the overfitting guard work as designed. They do **not** predict how much a given LLM will improve. Run `promptopt optimize --provider anthropic|openai` on your own task to measure that.

## Architecture

```
               Signature (inputs -> outputs, instructions)      Metric(example, outputs) -> [0,1]
                                    |                                       |
 train ─┐                           v                                       |
 dev  ──┼──>  Optimizer.compile(program, trainset, devset, lm) ─────────────┤
 test ─┐│      ├─ BootstrapFewShot : teacher traces -> demo subsets -> pick by dev
       ││      ├─ InstructionSearch: train failures -> proposer LM -> successive halving on dev
       ││      └─ RandomSearch     : (instruction × demo subset) trials, bound pruning + patience
       ││                           |
       ││                    Scorer (memo per program hash × dev prefix)
       ││                           |
       ││                    evaluate()  ── ThreadPoolExecutor ──> Predict.__call__
       ││                                                              | render_prompt / parse_completion
       ││                                                              v
       ││                    MeteredLM: sha256(model, max_tokens, prompt) cache, single-flight,
       ││                               Usage {calls, cache_hits, tokens, cost}
       ││                                                              |
       ││                    providers: MockProvider | AnthropicProvider | OpenAICompatProvider(urllib)
       │└──> OptimizationResult(program, dev_score, usage, history) ──> program.json (canonical, hashed)
       └───> holdout_report(baseline, result, testset)  ->  test scores + overfitting warning
```

| module | role |
|---|---|
| `signature.py` | `Field`, `Signature` (`"question -> answer"` shorthand), `Example` |
| `prompt.py` | plain-text prompt layout, tolerant `Prefix: value` parser |
| `program.py` | `Predict` (signature + instructions + demos), JSON save/load/hash, `diff_programs` |
| `lm.py` | provider protocol, `MeteredLM` (cache + single-flight + metering), `JsonlCache`, example prices |
| `evaluate.py` | threaded evaluation, mean ± standard error |
| `proposer.py` | failure-grounded meta-prompt and its parser |
| `optimizers/` | `BootstrapFewShot`, `InstructionSearch`, `RandomSearch`, `Ledger`, `Scorer` |
| `report.py` | held-out evaluation and the overfitting check |
| `tasks/` | three tasks with synthetic generators, metrics, and the mock's "skills" |
| `providers/` | mock, Anthropic SDK, OpenAI-compatible HTTP |

## Design decisions and trade-offs

- **Plain-text prompts, not chat/JSON mode.** The `---`-separated layout behaves the same on every provider, and the cache key is just a string. A saved program reads like the prompt itself. The cost is that field values are flattened to one line.
- **Failures come from train, selection runs on dev, the test set is never shown.** The proposer sees train failures, so proposals aren't tuned to the examples they're judged on. The test split is passed only to `holdout_report`. A test records every prompt during three optimizations and asserts that no test input appears in any of them.
- **Successive halving with prefix reuse.** Rung *r* gives each survivor `budget // (rungs × survivors)` examples. Rungs are prefixes of one fixed dev order, so a promoted survivor's earlier examples are cache hits. With 9 candidates and 64 dev examples, measured calls are under 60% of the exhaustive 9 × 64 (asserted in tests). The incumbent is always in the race, ties go to the incumbent, and a winner is accepted only if its **full-dev** score is strictly better.
- **Exact early stopping in random search.** A trial stops once `(points so far + examples left) / n` can't beat the incumbent. This never prunes a configuration that could have won. A test re-scores pruned trials on the full dev set to check that. Patience stops the search itself.
- **Bootstrapping uses a labelled teacher.** A zero-shot teacher only produces traces for inputs it already solves, and those demos teach nothing new. The teacher gets 3 random labelled train demos (never the example itself), and only traces that pass the metric are kept.
- **Single-flight cache.** With 8 threads, candidates that share demos fire identical prompts at the same moment. Only one call reaches the provider and the rest wait on its future.
- **Deterministic decoding during optimization.** The OpenAI-compatible provider sends `temperature=0`. The Anthropic adapter sends no sampling parameters because current Claude models reject non-default ones. The cache is what makes reruns reproducible.
- **Anthropic adapter.** It enables server-side refusal fallback by default (`fallbacks="default"`; turn it off for proxies with `refusal_fallback=False`). It treats `stop_reason == "refusal"` as a non-retryable error, maps SDK exceptions to retryable/non-retryable, and leaves retries to the SDK's own backoff.
- **Canonical JSON with a behavioural hash.** `program_hash` covers the signature, instructions and demos, not metadata, so re-saving with new scores doesn't look like a prompt change. Loading a hand-edited file fails loudly, and an unknown `schema_version` is refused rather than guessed.

## Limitations

- **The headline numbers come from a simulation.** Real-model gains will differ, maybe a lot. Small or strong models may gain little from instructions they already follow implicitly.
- Single-step programs only (one `Predict`). There are no multi-module pipelines and no credit assignment across steps the way DSPy does it.
- Field values are flattened to one line. Long documents, or outputs that need newlines (code, JSON), need a different layout.
- The overfitting check is a heuristic based on 2 standard errors. It doesn't replace a bigger dev set.
- Instruction proposals are parsed one per line, so multi-line proposed instructions are truncated to their first line.
- No async provider path. Parallelism uses threads, which is fine for I/O-bound HTTP but has no rate-limit awareness beyond per-call retries.
- Prices are hard-coded examples. They will go stale.

## Roadmap

- Multi-step programs (chains of `Predict`) with bootstrapped traces per step
- Bayesian or TPE search over demos × instructions (MIPRO-style) as a fourth optimizer
- A concurrency limiter and token-per-minute budgets per provider
- Run the benchmark on real models (Claude Sonnet/Haiku, a local Llama) and publish the numbers next to the simulated ones
- Cost-aware objective (score − λ·tokens) so shorter prompts win ties
- A `promptopt serve` loader that hot-reloads a compiled program by hash

## Development

```console
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q            # 57 tests, ~6 s, fully offline
.venv/bin/python examples/quickstart.py
.venv/bin/promptopt bench --seeds 5
```

MIT © 2026 Lijo Ferdinand
