"""Optimize a bundled task end to end, offline, in about a second.

    python examples/quickstart.py

Uses the MockProvider (a deterministic SIMULATION of an LLM; see
promptopt/tasks/base.py). Swap in a real provider with one line:

    lm = MeteredLM(AnthropicProvider())                       # needs ANTHROPIC_API_KEY
    lm = MeteredLM(OpenAICompatProvider("llama3.2"))          # local Ollama
"""

from pathlib import Path

from promptopt import InstructionSearch, MeteredLM, MockProvider, Predict, diff_programs, holdout_report
from promptopt.tasks import MATH

split = MATH.make_split(seed=0)                 # 48 train / 64 dev / 300 held-out test
lm = MeteredLM(MockProvider())                  # caching + token/cost metering around any provider
baseline = Predict(MATH.signature)              # zero-shot: just the signature's instructions

result = InstructionSearch(MATH.metric).compile(baseline, trainset=split.train, devset=split.dev, lm=lm)
print(result.summary())

# The optimizer never saw the test set; this is the number to believe.
report = holdout_report(baseline, result, split.test, MeteredLM(MockProvider()), MATH.metric, devset=split.dev)
print(report)

out = Path("out")
baseline.save(out / "math_baseline.json")
result.program.save(out / "math_optimized.json")
print()
print(diff_programs(baseline, result.program, a_name="baseline", b_name="optimized").splitlines()[0])
print("optimized instructions:", result.program.effective_instructions)
