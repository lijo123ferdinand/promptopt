"""The instruction-proposal meta-prompt: show an LLM where the current prompt
fails and ask it for better instructions.

Grounding proposals in concrete failures (rather than "rewrite this prompt to be
better") is what makes instruction search work in practice: the proposer can
see *which* inputs break and name the missing rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .lm import LM
from .signature import Example, Signature

MARKER = "### promptopt: propose-instructions"
_LINE = re.compile(r"^\s*(?:[-*]|\d+[.)])?\s*Instruction\s*:\s*(.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Failure:
    example: Example
    got: dict[str, str]


def render_meta_prompt(sig: Signature, current: str, failures: list[Failure], n: int) -> str:
    parts = [
        MARKER,
        "You are an expert prompt engineer improving the instructions given to a language model.",
        "",
        f"Task inputs: {', '.join(f.prefix for f in sig.inputs)}",
        f"Task outputs: {', '.join(f.prefix for f in sig.outputs)}",
        "Current instructions:",
        "<<<",
        current.strip(),
        ">>>",
        "",
        "The current instructions FAILED on these examples:",
    ]
    for i, fl in enumerate(failures, 1):
        parts.append(f"Example {i}")
        parts.append("Input:")
        parts += [f"  {f.prefix}: {' '.join(fl.example.inputs.get(f.name, '').split())}" for f in sig.inputs]
        parts.append("Expected: " + "; ".join(f"{f.prefix}: {fl.example.outputs.get(f.name, '')}" for f in sig.outputs))
        parts.append("Got: " + "; ".join(f"{f.prefix}: {fl.got.get(f.name, '')}" for f in sig.outputs))
    parts += [
        "",
        "Work out what the failures have in common, then write "
        f"{n} improved, self-contained instructions for the task. Keep what already works.",
        'Output each one on its own line, starting with "Instruction: ". Output nothing else.',
    ]
    return "\n".join(parts)


def parse_proposals(text: str) -> list[str]:
    """Extract ``Instruction: ...`` lines, deduplicated, order preserved."""
    seen: dict[str, None] = {}
    for line in text.splitlines():
        m = _LINE.match(line)
        if m:
            seen.setdefault(m.group(1).strip().strip('"'), None)
    return [s for s in seen if s]


def propose_instructions(
    lm: LM, sig: Signature, current: str, failures: list[Failure], n: int, *, max_tokens: int = 2048
) -> list[str]:
    prompt = render_meta_prompt(sig, current, failures, n)
    return parse_proposals(lm.complete(prompt, max_tokens=max_tokens).text)[:n]
