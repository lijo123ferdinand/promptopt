"""Signatures: the declarative contract of a task.

A signature says *what* a step does (input fields -> output fields, plus a
plain-language instruction) without saying *how* to prompt for it. Keeping the
"what" separate from the "how" is the whole point: the optimizer is free to
rewrite instructions and pick demonstrations, while the inputs/outputs your
code depends on stay fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Field:
    """One named slot in a signature.

    ``prefix`` is the label rendered in the prompt ("Sentiment:"). It defaults
    to a title-cased version of ``name`` so prompts read naturally to a model
    while code keeps snake_case keys.
    """

    name: str
    desc: str = ""
    prefix: str = ""

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"field name must be an identifier, got {self.name!r}")
        if not self.prefix:
            # frozen dataclass: bypass __setattr__ to fill the derived default once.
            object.__setattr__(self, "prefix", self.name.replace("_", " ").title())

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "desc": self.desc, "prefix": self.prefix}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Field":
        return cls(name=d["name"], desc=d.get("desc", ""), prefix=d.get("prefix", ""))


@dataclass(frozen=True)
class Signature:
    """Inputs -> outputs, plus the default (un-optimized) instructions."""

    inputs: tuple[Field, ...]
    outputs: tuple[Field, ...]
    instructions: str = ""
    name: str = ""

    def __post_init__(self) -> None:
        if not self.inputs or not self.outputs:
            raise ValueError("a signature needs at least one input and one output")
        names = [f.name for f in (*self.inputs, *self.outputs)]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names in signature: {names}")
        prefixes = [f.prefix.lower() for f in (*self.inputs, *self.outputs)]
        if len(set(prefixes)) != len(prefixes):
            # Prefixes are how we parse completions back into fields; a clash
            # would make parsing ambiguous, so reject it up front.
            raise ValueError(f"duplicate field prefixes in signature: {prefixes}")

    @classmethod
    def parse(
        cls,
        spec: str,
        instructions: str = "",
        *,
        descs: dict[str, str] | None = None,
        name: str = "",
    ) -> "Signature":
        """Build from the compact form ``"question, context -> answer"``."""
        if "->" not in spec:
            raise ValueError(f"signature spec must contain '->': {spec!r}")
        left, right = spec.split("->", 1)
        descs = descs or {}

        def fields(part: str) -> tuple[Field, ...]:
            names = [n.strip() for n in part.split(",") if n.strip()]
            return tuple(Field(n, descs.get(n, "")) for n in names)

        return cls(fields(left), fields(right), instructions, name)

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.outputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": [f.to_dict() for f in self.inputs],
            "outputs": [f.to_dict() for f in self.outputs],
            "instructions": self.instructions,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Signature":
        return cls(
            inputs=tuple(Field.from_dict(x) for x in d["inputs"]),
            outputs=tuple(Field.from_dict(x) for x in d["outputs"]),
            instructions=d.get("instructions", ""),
            name=d.get("name", ""),
        )


@dataclass(frozen=True)
class Example:
    """A labelled example: input values plus the expected output values.

    Demonstrations inside a prompt are also Examples. For bootstrapped demos the
    ``outputs`` are the *model's own* passing outputs rather than gold labels,
    which is what makes them good demos: they are in the model's own voice.
    """

    inputs: dict[str, str]
    outputs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": dict(self.inputs), "outputs": dict(self.outputs)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Example":
        return cls(inputs=dict(d["inputs"]), outputs=dict(d.get("outputs", {})))

    def key(self) -> tuple[tuple[str, str], ...]:
        """Hashable identity of the inputs (used to dedupe and to exclude self-demos)."""
        return tuple(sorted(self.inputs.items()))
