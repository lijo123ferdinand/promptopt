"""Named-entity extraction (PERSON / ORG / LOCATION) from one-sentence news snippets.

Output format: ``Alice Chen (PERSON); Pune (LOCATION)`` in order of appearance,
or ``none``. Scored by F1 over (name, type) pairs.

Phenomena and their naive mistakes:

* multiword          - "Maria Garcia" extracted as just "Maria"
* sentence_initial   - "Yesterday ..." : the capitalised first word is extracted as an ORG
* ambiguity          - "flew to Jordan" typed PERSON (the name's most common type), not LOCATION
"""

from __future__ import annotations

import random
import re

from ..signature import Example, Field, Signature
from .base import Skill, Task

PERSON_1 = ("Alice", "Ravi", "Priya", "Tom", "Kenji", "Amara", "Lukas", "Sofia", "Omar", "Mei")
PERSON_2 = ("Alice Chen", "Ravi Kumar", "Maria Garcia", "Tom Becker", "Kenji Sato", "Amara Okafor",
            "Lukas Novak", "Sofia Rossi", "Omar Haddad", "Mei Lin")
ORG_1 = ("Globex", "Initech", "Umbrella", "Hooli", "Vandelay", "Stark")
ORG_2 = ("Acme Corp", "Northwind Traders", "Blue Harbor Bank", "Pied Piper", "Wayne Enterprises",
         "Soylent Foods")
LOC_1 = ("Paris", "Pune", "Berlin", "Lagos", "Osaka", "Lima", "Oslo", "Austin")
LOC_2 = ("New York", "San Francisco", "Lake Como", "Cape Town", "Hong Kong", "Buenos Aires")
# Ambiguous names: (default type a naive reader assumes, the other possible type)
AMBIGUOUS = {"Jordan": "PERSON", "Georgia": "LOCATION", "Florence": "PERSON", "Victoria": "PERSON",
             "Chad": "PERSON", "Sydney": "LOCATION"}
OPENERS = ("Yesterday", "Today", "Later", "Meanwhile", "Afterwards", "Earlier")

GAZETTEER: dict[str, str] = {}
for _names, _type in ((PERSON_1, "PERSON"), (PERSON_2, "PERSON"), (ORG_1, "ORG"), (ORG_2, "ORG"),
                      (LOC_1, "LOCATION"), (LOC_2, "LOCATION")):
    for _n in _names:
        GAZETTEER[_n] = _type
MAX_SPAN = 3
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")
# Context that decides the type of an ambiguous name.
_LOC_BEFORE = {"in", "to", "from", "visited", "near"}
_PER_AFTER = {"said", "met", "joined", "called", "signed", "announced"}


def generate(rng: random.Random) -> str:
    def per(multi: bool) -> str:
        return rng.choice(PERSON_2 if multi else PERSON_1)

    def org(multi: bool) -> str:
        return rng.choice(ORG_2 if multi else ORG_1)

    def loc(multi: bool) -> str:
        return rng.choice(LOC_2 if multi else LOC_1)

    multi = rng.random() < 0.35
    r = rng.random()
    if r < 0.25:
        # ambiguity: put the ambiguous name in the context of its *non-default* type
        name, default = rng.choice(list(AMBIGUOUS.items()))
        if default == "PERSON":
            return f"{per(multi)} flew to {name} for talks with {org(multi)}."
        return f"{name} said {org(multi)} would open an office in {loc(multi)}."
    templates = (
        lambda: f"{per(multi)} met {per(multi)} in {loc(multi)}.",
        lambda: f"{per(multi)} joined {org(multi)} as a senior engineer.",
        lambda: f"{org(multi)} opened a new office in {loc(multi)}.",
        lambda: f"{per(multi)} said {org(multi)} will hire in {loc(multi)}.",
    )
    sent = rng.choice(templates)()
    if r < 0.50:  # sentence_initial: a capitalised common word starts the sentence
        # Every template starts with a proper name, so the name keeps its capital
        # and the opener is the only capitalised non-entity.
        sent = f"{rng.choice(OPENERS)} {sent}"
    return sent


def _spans(tokens: list[str]) -> list[tuple[int, int, str]]:
    """Longest-match gazetteer spans as (start, end, name)."""
    out, i = [], 0
    while i < len(tokens):
        for length in range(MAX_SPAN, 0, -1):
            name = " ".join(tokens[i : i + length])
            if len(tokens[i : i + length]) == length and (name in GAZETTEER or (length == 1 and name in AMBIGUOUS)):
                out.append((i, i + length, name))
                i += length
                break
        else:
            i += 1
    return out


def format_entities(ents: list[tuple[str, str]]) -> str:
    return "; ".join(f"{n} ({t})" for n, t in ents) if ents else "none"


def parse_entities(text: str) -> set[tuple[str, str]]:
    found = set()
    for m in re.finditer(r"([A-Za-z][A-Za-z' -]*?)\s*\((PERSON|ORG|LOCATION)\)", text, re.IGNORECASE):
        found.add((" ".join(m.group(1).split()), m.group(2).upper()))
    return found


class EntitySkill(Skill):
    output_fields = ("entities",)
    keywords = {
        "multiword": ("multi-word", "multiword", "full name", "complete span"),
        "sentence_initial": ("sentence-initial", "start of the sentence", "capitalized common", "capitalised common"),
        "ambiguity": ("ambiguous", "ambiguity", "surrounding context"),
    }
    hints = {
        "multiword": (
            "Extract the full name of every entity, including multi-word names such as 'New York'.",
            "Keep multiword entities together as one complete span.",
        ),
        "sentence_initial": (
            "A capitalized common word at the start of the sentence (e.g. 'Yesterday') is not an entity.",
            "Do not tag sentence-initial words like 'Meanwhile' just because they are capitalised.",
        ),
        "ambiguity": (
            "Some names are ambiguous ('Jordan' can be a person or a country): use the surrounding context to pick the type.",
            "Resolve ambiguous names by their surrounding context: 'flew to X' means a LOCATION, 'X said' means a PERSON.",
        ),
    }
    decoys = (
        "List the entities in the order they appear.",
        "Be careful and thorough.",
        "Use the labels PERSON, ORG and LOCATION.",
        "Do not add any explanation.",
    )

    def phenomena(self, text: str) -> set[str]:
        toks = _TOKEN.findall(text)
        spans = _spans(toks)
        found = set()
        if any(e - s > 1 for s, e, _ in spans):
            found.add("multiword")
        if toks and toks[0] in OPENERS:
            found.add("sentence_initial")
        if any(n in AMBIGUOUS for _, _, n in spans):
            found.add("ambiguity")
        return found

    def solve(self, text: str, covered: set[str]) -> dict[str, str]:
        toks = _TOKEN.findall(text)
        ents: list[tuple[str, str]] = []
        if toks and toks[0] in OPENERS and "sentence_initial" not in covered:
            ents.append((toks[0], "ORG"))  # naive: any capitalised word is a name
        for s, e, name in _spans(toks):
            if name in AMBIGUOUS:
                typ = AMBIGUOUS[name]
                if "ambiguity" in covered:
                    before = toks[s - 1].lower() if s > 0 else ""
                    after = toks[e].lower() if e < len(toks) else ""
                    if before in _LOC_BEFORE:
                        typ = "LOCATION"
                    elif after in _PER_AFTER:
                        typ = "PERSON"
                ents.append((name, typ))
                continue
            typ = GAZETTEER[name]
            if e - s > 1 and "multiword" not in covered:
                name = toks[s]  # naive: stop at the first token
            ents.append((name, typ))
        return {"entities": format_entities(ents)}

    def corrupt(self, text: str, answer: dict[str, str], rng: random.Random) -> dict[str, str]:
        ents = sorted(parse_entities(answer.get("entities", "")))  # sorted: set order varies per process
        if len(ents) > 1:
            ents.pop(rng.randrange(len(ents)))  # drop one entity
        else:
            ents = [(n, "ORG" if t != "ORG" else "PERSON") for n, t in ents]  # mistype
        return {"entities": format_entities(sorted(ents))}


def metric(example: Example, outputs: dict[str, str]) -> float:
    gold = parse_entities(example.outputs["entities"])
    got = parse_entities(outputs.get("entities", ""))
    if not gold and not got:
        return 1.0
    tp = len(gold & got)
    if tp == 0:
        return 0.0
    p, r = tp / len(got), tp / len(gold)
    return 2 * p * r / (p + r)


TASK = Task(
    name="entities",
    signature=Signature(
        inputs=(Field("sentence", "one news sentence"),),
        outputs=(Field("entities", "Name (TYPE); Name (TYPE) with TYPE in PERSON, ORG, LOCATION, or 'none'"),),
        instructions="Extract the named entities (people, organizations and locations) from the sentence.",
        name="entities",
    ),
    metric=metric,
    skill=EntitySkill(),
    generate=generate,
    input_field="sentence",
)
