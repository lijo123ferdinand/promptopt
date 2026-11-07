"""Task = signature + metric + synthetic data + the simulated "skill" the mock uses.

The simulation contract (read this before trusting any offline number)
---------------------------------------------------------------------
Every bundled task is built from a small set of named *phenomena* (negation,
sarcasm, multi-step arithmetic, ...). Each phenomenon has a systematic naive
mistake: a model that has not been told about it gets it wrong the same way
every time, the way real models do ("not good" read as positive).

The MockProvider decides which phenomena a prompt *covers*:

* by an instruction keyword (an instruction that mentions "negation"), or
* by a correct demonstration whose input exhibits that phenomenon.

It then solves the input with the covered rules applied and the naive rule for
everything else, plus a small hash-seeded noise rate. So the optimizer's job is
real search over a real objective: find instructions and demos that cover the
phenomena that actually occur. The *mock's* reaction to a prompt is a model of
an LLM, not an LLM. Offline numbers show the optimizers work; they say nothing
about how much a given real model will gain.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from ..evaluate import Metric
from ..signature import Example, Signature


class Skill(ABC):
    """What the mock knows about one task."""

    #: output field names; how the mock recognises which task a prompt is for
    output_fields: tuple[str, ...]
    #: phenomenon -> instruction keywords (lower-case substrings) that cover it
    keywords: dict[str, tuple[str, ...]]
    #: phenomenon -> hint sentences a good proposer might write; each contains a keyword
    hints: dict[str, tuple[str, ...]]
    #: plausible-but-useless instruction sentences (no keywords)
    decoys: tuple[str, ...]

    @abstractmethod
    def phenomena(self, text: str) -> set[str]:
        """Which phenomena this input exhibits."""

    @abstractmethod
    def solve(self, text: str, covered: set[str]) -> dict[str, str]:
        """Answer with covered phenomena handled and the naive rule elsewhere."""

    @abstractmethod
    def corrupt(self, text: str, answer: dict[str, str], rng: random.Random) -> dict[str, str]:
        """A plausible wrong answer, used for noise."""

    def all_phenomena(self) -> set[str]:
        return set(self.keywords)

    def gold(self, text: str) -> dict[str, str]:
        return self.solve(text, self.all_phenomena())


@dataclass(frozen=True)
class Split:
    train: list[Example]
    dev: list[Example]
    test: list[Example]


@dataclass(frozen=True)
class Task:
    name: str
    signature: Signature
    metric: Metric
    skill: Skill
    generate: Callable[[random.Random], str]  # one random input text
    input_field: str

    def make_split(self, seed: int = 0, n_train: int = 48, n_dev: int = 64, n_test: int = 300) -> Split:
        """Generate disjoint train/dev/test sets.

        Inputs are deduplicated across all three splits so a test item can never
        be memorised from a demonstration: the held-out score has to be earned.
        """
        rng = random.Random(f"{self.name}:{seed}")
        need = n_train + n_dev + n_test
        seen: set[str] = set()
        texts: list[str] = []
        attempts = 0
        while len(texts) < need:
            attempts += 1
            if attempts > need * 200:
                raise RuntimeError(f"{self.name}: generator cannot produce {need} distinct inputs")
            t = self.generate(rng)
            if t not in seen:
                seen.add(t)
                texts.append(t)
        exs = [Example({self.input_field: t}, self.skill.gold(t)) for t in texts]
        return Split(exs[:n_train], exs[n_train : n_train + n_dev], exs[n_train + n_dev :])
