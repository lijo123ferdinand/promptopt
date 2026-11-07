"""Declare your own task: a signature, a metric and a few examples.

    python examples/custom_task.py                       # offline mock (plumbing only)
    PROVIDER=anthropic python examples/custom_task.py    # Claude (pip install 'promptopt[anthropic]')
    PROVIDER=ollama MODEL=llama3.2 python examples/custom_task.py

The mock only "knows" the bundled tasks, so on a custom signature it returns
placeholder text and every score is 0: run it offline to check the wiring,
then point it at a real model to actually optimize.
"""

import os

from promptopt import BootstrapFewShot, Example, Field, MeteredLM, Predict, Signature
from promptopt.providers import make_provider

signature = Signature(
    inputs=(Field("ticket", "a customer support message"),),
    outputs=(Field("team", "one of: billing, shipping, technical"),),
    instructions="Route the support ticket to the team that should handle it.",
)


def metric(example: Example, outputs: dict[str, str]) -> float:
    return float(outputs.get("team", "").strip().lower() == example.outputs["team"])


data = [
    ("I was charged twice for my order", "billing"),
    ("My refund has not arrived yet", "billing"),
    ("Can I get an invoice with my company name?", "billing"),
    ("The parcel says delivered but it is not here", "shipping"),
    ("Can you ship to a PO box?", "shipping"),
    ("My package is stuck in customs", "shipping"),
    ("The app crashes when I open settings", "technical"),
    ("I cannot reset my password, the link is broken", "technical"),
    ("Sync fails with error 504 since the update", "technical"),
    ("Why was my card declined at checkout?", "billing"),
    ("The tracking number does not work", "shipping"),
    ("Two-factor codes never reach my phone", "technical"),
]
examples = [Example({"ticket": t}, {"team": y}) for t, y in data]
train, dev = examples[:6], examples[6:]

kind = os.environ.get("PROVIDER", "mock")
lm = MeteredLM(make_provider(kind, os.environ.get("MODEL")))
result = BootstrapFewShot(metric, demos_per_program=3, num_candidates=4).compile(
    Predict(signature), trainset=train, devset=dev, lm=lm)
print(result.summary())
print(result.program.to_json())
