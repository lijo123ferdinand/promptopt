"""Programs: a signature plus the learnable parameters (instructions, demos).

A compiled program serializes to canonical JSON (sorted keys, 2-space indent,
trailing newline) so that a checked-in ``program.json`` produces a clean,
reviewable ``git diff`` when re-optimized. ``program_hash`` covers only the
parts that change model behaviour (signature, instructions, demos), never the
metadata, so "did the prompt actually change?" is a one-field comparison.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .lm import LM
from .prompt import parse_completion, render_prompt
from .signature import Example, Signature

FORMAT = "promptopt.program"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Prediction:
    outputs: dict[str, str]
    completion: str = ""
    cached: bool = False


@dataclass(frozen=True)
class Predict:
    signature: Signature
    instructions: str | None = None  # None -> the signature's default instructions
    demos: tuple[Example, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)
    max_tokens: int = 1024

    @property
    def effective_instructions(self) -> str:
        return self.instructions if self.instructions is not None else self.signature.instructions

    def with_(self, **changes: Any) -> "Predict":
        if "demos" in changes:
            changes["demos"] = tuple(changes["demos"])
        return replace(self, **changes)

    def prompt(self, inputs: dict[str, str]) -> str:
        missing = [n for n in self.signature.input_names if n not in inputs]
        if missing:
            raise KeyError(f"missing input fields: {missing}")
        return render_prompt(self.signature, self.effective_instructions, self.demos, inputs)

    def __call__(self, lm: LM, **inputs: str) -> Prediction:
        c = lm.complete(self.prompt(inputs), max_tokens=self.max_tokens)
        return Prediction(parse_completion(self.signature, c.text), c.text, c.cached)

    # ------------------------------------------------------------------ serialization

    def program_hash(self) -> str:
        body = {
            "signature": self.signature.to_dict(),
            "instructions": self.effective_instructions,
            "demos": [d.to_dict() for d in self.demos],
        }
        blob = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "program_hash": self.program_hash(),
            "signature": self.signature.to_dict(),
            "instructions": self.effective_instructions,
            "demos": [d.to_dict() for d in self.demos],
            "max_tokens": self.max_tokens,
            "meta": self.meta,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def save(self, path: str | os.PathLike[str]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Predict":
        if d.get("format") != FORMAT:
            raise ValueError(f"not a promptopt program (format={d.get('format')!r})")
        version = d.get("schema_version")
        if version != SCHEMA_VERSION:
            # One place to add migrations when the schema evolves; refusing an
            # unknown version beats silently mis-loading a prompt into prod.
            raise ValueError(f"unsupported schema_version {version}; this build reads {SCHEMA_VERSION}")
        prog = cls(
            signature=Signature.from_dict(d["signature"]),
            instructions=d["instructions"],
            demos=tuple(Example.from_dict(x) for x in d.get("demos", [])),
            meta=dict(d.get("meta", {})),
            max_tokens=int(d.get("max_tokens", 1024)),
        )
        if "program_hash" in d and d["program_hash"] != prog.program_hash():
            raise ValueError("program_hash mismatch: the file was edited by hand or is corrupt")
        return prog

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Predict":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def diff_programs(a: Predict, b: Predict, *, a_name: str = "a", b_name: str = "b") -> str:
    """A reviewer-friendly diff: a semantic summary, then a unified JSON diff."""
    lines: list[str] = []
    if a.program_hash() == b.program_hash():
        lines.append(f"programs are behaviourally identical ({a.program_hash()})")
    else:
        lines.append(f"{a_name}: {a.program_hash()}  ->  {b_name}: {b.program_hash()}")
        if a.effective_instructions != b.effective_instructions:
            lines.append("instructions changed")
        a_keys = {d.key() for d in a.demos}
        b_keys = {d.key() for d in b.demos}
        added, removed = len(b_keys - a_keys), len(a_keys - b_keys)
        if added or removed:
            lines.append(f"demos: +{added} -{removed} (now {len(b.demos)})")
    for side, p in ((a_name, a), (b_name, b)):
        score = p.meta.get("dev_score")
        if score is not None:
            lines.append(f"{side} dev_score={score}")
    udiff = difflib.unified_diff(
        a.to_json().splitlines(), b.to_json().splitlines(), a_name, b_name, lineterm=""
    )
    body = "\n".join(udiff)
    return "\n".join(lines) + ("\n\n" + body if body else "") + "\n"
