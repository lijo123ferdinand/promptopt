"""Grade-school math word problems with a programmatic checker.

Signature: ``problem -> reasoning, answer``. The metric parses the first number
out of ``answer`` and compares it to the exact value computed when the problem
was generated, so there is no fuzzy string matching to argue with.

Phenomena and their naive mistakes:

* multi_step   - "has 12, buys 7, then gives 5 away"   naive: stops after the first step (19)
* percent      - "80 dollars, 25% off"                  naive: subtracts 25 (55, not 60)
* unit         - "150 minutes ... how many hours?"      naive: forgets to convert (150)
* distractor   - "is 9 years old and has 12 apples"     naive: adds every number it sees
"""

from __future__ import annotations

import random
import re
from fractions import Fraction

from ..signature import Example, Field, Signature
from .base import Skill, Task

NAMES = ("Ana", "Ben", "Chen", "Dev", "Ema", "Femi", "Gita", "Hugo", "Ines", "Jae", "Kofi", "Lena")
ITEMS = ("apples", "marbles", "stickers", "books", "pencils", "shells", "cards", "coins")
PRODUCTS = ("jacket", "lamp", "backpack", "kettle", "bicycle helmet", "desk chair")
ACTIVITIES = ("movie", "bus ride", "football match", "cooking class", "flight", "concert")


def fmt(x: Fraction | int | float) -> str:
    x = Fraction(x).limit_denominator(1000)
    if x.denominator == 1:
        return str(x.numerator)
    return f"{float(x):.2f}".rstrip("0").rstrip(".")


def generate(rng: random.Random) -> str:
    n, m = rng.sample(NAMES, 2)
    item = rng.choice(ITEMS)
    a, b = rng.randint(5, 60), rng.randint(2, 30)
    r = rng.random()
    if r < 0.30:  # plain one-step
        if rng.random() < 0.5:
            return f"{n} has {a} {item} and buys {b} more. How many {item} does {n} have now?"
        a = max(a, b + 1)
        return f"{n} has {a} {item} and gives away {b}. How many {item} does {n} have now?"
    if r < 0.50:  # multi_step
        c = rng.randint(1, a + b - 1)
        return (f"{n} has {a} {item}, buys {b} more, and then gives {c} to {m}. "
                f"How many {item} does {n} have now?")
    if r < 0.65:  # percent
        price = rng.choice(range(20, 401, 20))
        pct = rng.choice((10, 20, 25, 50, 75))
        return (f"A {rng.choice(PRODUCTS)} costs {price} dollars and is on sale for {pct}% off. "
                f"What is the sale price in dollars?")
    if r < 0.80:  # unit
        if rng.random() < 0.5:
            mins = rng.choice(range(30, 601, 30))
            return f"A {rng.choice(ACTIVITIES)} lasts {mins} minutes. How many hours is that?"
        cents, days = rng.choice((25, 50, 75, 150, 200)), rng.randint(2, 20)
        return f"{n} saves {cents} cents a day for {days} days. How many dollars does {n} save?"
    age = rng.randint(6, 15)  # distractor
    return (f"{n} is {age} years old and has {a} {item}. {n} buys {b} more. "
            f"How many {item} does {n} have now?")


_PLAIN_ADD = re.compile(r"has (\d+) \w+ and buys (\d+) more\.")
_PLAIN_SUB = re.compile(r"has (\d+) \w+ and gives away (\d+)\.")
_MULTI = re.compile(r"has (\d+) \w+, buys (\d+) more, and then gives (\d+) to")
_PCT = re.compile(r"costs (\d+) dollars and is on sale for (\d+)% off")
_MIN = re.compile(r"lasts (\d+) minutes\. How many hours")
_CENTS = re.compile(r"saves (\d+) cents a day for (\d+) days\. How many dollars")
_DISTRACT = re.compile(r"is (\d+) years old and has (\d+) \w+\. \w+ buys (\d+) more\.")


class MathSkill(Skill):
    output_fields = ("reasoning", "answer")
    keywords = {
        "multi_step": ("step by step", "step-by-step", "every step", "each operation"),
        "percent": ("percent", "percentage"),
        "unit": ("unit", "convert"),
        "distractor": ("irrelevant", "distractor"),
    }
    hints = {
        "multi_step": (
            "Solve the problem step by step and apply every operation in order before answering.",
            "Work step-by-step: track the quantity through each operation mentioned.",
        ),
        "percent": (
            "A discount given as a percent is a fraction of the price: 25% off 80 is 80 - 20 = 60.",
            "Turn every percentage into a fraction of the original amount before subtracting.",
        ),
        "unit": (
            "Check the unit the question asks for and convert (60 minutes = 1 hour, 100 cents = 1 dollar).",
            "Always convert the result into the unit requested by the question.",
        ),
        "distractor": (
            "Ignore irrelevant numbers such as ages that do not affect the quantity asked about.",
            "Some numbers are distractors; use only the quantities the question is about.",
        ),
    }
    decoys = (
        "Give the final answer as a number.",
        "Double-check your arithmetic.",
        "Read the problem carefully.",
        "Keep the reasoning short.",
    )

    def phenomena(self, text: str) -> set[str]:
        found = set()
        if _MULTI.search(text):
            found.add("multi_step")
        if _PCT.search(text):
            found.add("percent")
        if _MIN.search(text) or _CENTS.search(text):
            found.add("unit")
        if _DISTRACT.search(text):
            found.add("distractor")
        return found

    def solve(self, text: str, covered: set[str]) -> dict[str, str]:
        if m := _MULTI.search(text):
            a, b, c = map(int, m.groups())
            if "multi_step" in covered:
                return {"reasoning": f"{a} + {b} = {a + b}; {a + b} - {c} = {a + b - c}", "answer": fmt(a + b - c)}
            return {"reasoning": f"{a} + {b} = {a + b}", "answer": fmt(a + b)}
        if m := _PCT.search(text):
            price, pct = map(int, m.groups())
            if "percent" in covered:
                off = Fraction(price * pct, 100)
                return {"reasoning": f"{pct}% of {price} is {fmt(off)}; {price} - {fmt(off)} = {fmt(price - off)}",
                        "answer": fmt(price - off)}
            return {"reasoning": f"{price} - {pct} = {price - pct}", "answer": fmt(price - pct)}
        if m := _MIN.search(text):
            mins = int(m.group(1))
            if "unit" in covered:
                return {"reasoning": f"{mins} / 60 = {fmt(Fraction(mins, 60))}", "answer": fmt(Fraction(mins, 60))}
            return {"reasoning": f"it lasts {mins}", "answer": fmt(mins)}
        if m := _CENTS.search(text):
            cents, days = map(int, m.groups())
            total = cents * days
            if "unit" in covered:
                return {"reasoning": f"{cents} x {days} = {total} cents = {fmt(Fraction(total, 100))} dollars",
                        "answer": fmt(Fraction(total, 100))}
            return {"reasoning": f"{cents} x {days} = {total}", "answer": fmt(total)}
        if m := _DISTRACT.search(text):
            age, a, b = map(int, m.groups())
            if "distractor" in covered:
                return {"reasoning": f"{a} + {b} = {a + b}", "answer": fmt(a + b)}
            return {"reasoning": f"{age} + {a} + {b} = {age + a + b}", "answer": fmt(age + a + b)}
        if m := _PLAIN_ADD.search(text):
            a, b = map(int, m.groups())
            return {"reasoning": f"{a} + {b} = {a + b}", "answer": fmt(a + b)}
        if m := _PLAIN_SUB.search(text):
            a, b = map(int, m.groups())
            return {"reasoning": f"{a} - {b} = {a - b}", "answer": fmt(a - b)}
        return {"reasoning": "I could not parse the problem.", "answer": "0"}

    def corrupt(self, text: str, answer: dict[str, str], rng: random.Random) -> dict[str, str]:
        try:
            wrong = Fraction(answer.get("answer", "0")) + rng.choice((-2, -1, 1, 2, 10))
        except ValueError:
            wrong = Fraction(1)
        return {"reasoning": answer.get("reasoning", "") + " (arithmetic slip)", "answer": fmt(wrong)}


_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def parse_number(text: str) -> Fraction | None:
    m = _NUM.search(text.replace(",", "").replace("$", ""))
    if not m:
        return None
    try:
        return Fraction(m.group(0))
    except (ValueError, ZeroDivisionError):
        return None


def metric(example: Example, outputs: dict[str, str]) -> float:
    gold, got = parse_number(example.outputs["answer"]), parse_number(outputs.get("answer", ""))
    if gold is None or got is None:
        return 0.0
    return float(abs(gold - got) < Fraction(1, 100))


TASK = Task(
    name="math",
    signature=Signature(
        inputs=(Field("problem", "a short math word problem"),),
        outputs=(Field("reasoning", "brief working"), Field("answer", "the final number only")),
        instructions="Solve the math word problem.",
        name="math",
    ),
    metric=metric,
    skill=MathSkill(),
    generate=generate,
    input_field="problem",
)
