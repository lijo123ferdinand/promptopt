"""Sentiment classification (positive / negative / neutral) of short product reviews.

Phenomena and their naive mistakes:

* negation  - "The camera was not great."      naive: positive   (ignores "not")
* sarcasm   - "Oh great, the battery died..."   naive: positive   (takes "great" literally)
* contrast  - "X was good, but Y was awful."     naive: positive   (first opinion wins)
"""

from __future__ import annotations

import random
import re

from ..signature import Example, Field, Signature
from .base import Skill, Task

POS = ("good", "great", "excellent", "wonderful", "fantastic", "solid", "delightful", "impressive")
NEG = ("bad", "terrible", "awful", "disappointing", "poor", "mediocre", "flimsy", "frustrating")
SUBJECTS = (
    "the battery", "the camera", "the screen", "the delivery", "the support team", "the packaging",
    "the sound", "the keyboard", "the app", "the fabric", "the charger", "the manual", "the setup",
    "the price", "the speaker", "the strap",
)
EVENTS = ("died", "cracked", "stopped working", "fell apart", "overheated", "froze")
WHEN = ("after a day", "on the first use", "within a week", "the moment I opened it", "again")
NEUTRAL = (
    "{S} arrived on {D}.", "I bought it for {S2} last {D}.", "{S} comes in a box with a leaflet.",
    "I ordered it on {D} with {S2}.",
)
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

LABELS = ("positive", "negative", "neutral")
_WORD = re.compile(r"[a-z']+")


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def generate(rng: random.Random) -> str:
    s, s2 = rng.sample(SUBJECTS, 2)
    r = rng.random()
    if r < 0.40:  # plain
        kind = rng.random()
        if kind < 0.4:
            return f"{_cap(s)} was {rng.choice(POS)}."
        if kind < 0.8:
            return f"{_cap(s)} was {rng.choice(NEG)}."
        return rng.choice(NEUTRAL).format(S=_cap(s), S2=s2, D=rng.choice(DAYS))
    if r < 0.60:  # negation
        word = rng.choice(POS + NEG)
        neg = rng.choice(("not", "never"))
        tail = " at all" if rng.random() < 0.3 else ""
        return f"{_cap(s)} was {neg} {word}{tail}."
    if r < 0.80:  # sarcasm
        if rng.random() < 0.5:
            return f"Oh great, {s} {rng.choice(EVENTS)} {rng.choice(WHEN)}."
        return f"Just what I needed: {s} {rng.choice(EVENTS)} {rng.choice(WHEN)}."
    # contrast: the two clauses disagree, the last one decides
    if rng.random() < 0.5:
        a, b = rng.choice(POS), rng.choice(NEG)
    else:
        a, b = rng.choice(NEG), rng.choice(POS)
    return f"{_cap(s)} was {a}, but {s2} was {b}."


class SentimentSkill(Skill):
    output_fields = ("sentiment",)
    keywords = {
        "negation": ("negation", "negated", "negating"),
        "sarcasm": ("sarcasm", "sarcastic", "irony", "ironic"),
        "contrast": ("contrast", "final clause", "last clause"),
    }
    hints = {
        "negation": (
            "Handle negation: a negated positive word ('not good') is negative and a negated negative word ('not bad') is positive.",
            "Watch for negated opinion words such as 'not' or 'never'; they flip the polarity.",
        ),
        "sarcasm": (
            "Detect sarcasm: praise followed by a failure ('Oh great, it broke') is negative.",
            "Sarcastic openers like 'Oh great' or 'Just what I needed' signal a complaint.",
        ),
        "contrast": (
            "In a contrast sentence joined by 'but', the final clause decides the sentiment.",
            "When the review has a contrast, weigh the last clause over the first.",
        ),
    }
    decoys = (
        "Read the review carefully before answering.",
        "Answer with exactly one lowercase label.",
        "Consider the overall tone of the review.",
        "Be precise and concise.",
    )

    def phenomena(self, text: str) -> set[str]:
        t = text.lower()
        found = set()
        if re.search(r"\b(not|never)\b", t):
            found.add("negation")
        if t.startswith(("oh great", "just what i needed")):
            found.add("sarcasm")
        if ", but " in t:
            found.add("contrast")
        return found

    @staticmethod
    def _polarity(clause: str, negation: bool) -> str:
        words = _WORD.findall(clause.lower())
        for i, w in enumerate(words):
            pol = "positive" if w in POS else "negative" if w in NEG else None
            if pol is None:
                continue
            if negation and i > 0 and words[i - 1] in ("not", "never"):
                pol = "negative" if pol == "positive" else "positive"
            return pol
        return "neutral"

    def solve(self, text: str, covered: set[str]) -> dict[str, str]:
        found = self.phenomena(text)
        if "sarcasm" in found and "sarcasm" in covered:
            return {"sentiment": "negative"}
        clause = text
        if "contrast" in found and "contrast" in covered:
            clause = text.lower().split(", but ", 1)[1]
        return {"sentiment": self._polarity(clause, "negation" in covered)}

    def corrupt(self, text: str, answer: dict[str, str], rng: random.Random) -> dict[str, str]:
        return {"sentiment": rng.choice([lbl for lbl in LABELS if lbl != answer.get("sentiment")])}


def metric(example: Example, outputs: dict[str, str]) -> float:
    got = outputs.get("sentiment", "").strip().strip(".").lower()
    return float(got == example.outputs["sentiment"])


TASK = Task(
    name="sentiment",
    signature=Signature(
        inputs=(Field("text", "a short product review"),),
        outputs=(Field("sentiment", "one of: positive, negative, neutral"),),
        instructions="Classify the sentiment of the product review as positive, negative or neutral.",
        name="sentiment",
    ),
    metric=metric,
    skill=SentimentSkill(),
    generate=generate,
    input_field="text",
)
