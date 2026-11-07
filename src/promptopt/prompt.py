"""Rendering a (signature, instructions, demos, input) tuple into one prompt string,
and parsing a completion back into output fields.

The layout is deliberately plain text with ``---`` separated blocks:

    <instructions>

    ---

    Follow the following format.

    Text: <desc>
    Sentiment: <desc>

    ---

    Text: the demo input
    Sentiment: the demo output

    ---

    Text: the real input
    Sentiment:

Why plain text and not chat messages / JSON mode: it works identically on every
provider (Anthropic, any OpenAI-compatible server, a local Ollama model), it is
trivially diffable in a saved program, and the cache key is just the string.
The prompt ends on the first output prefix so the model continues the pattern.
"""

from __future__ import annotations

import re

from .signature import Example, Signature

SEPARATOR = "\n\n---\n\n"
FORMAT_HEADER = "Follow the following format."


def _one_line(value: str) -> str:
    # Field values are rendered on a single line. Multi-line values would make
    # "Prefix: value" parsing ambiguous with the plain-text layout, and none of
    # the bundled tasks need them. Collapse whitespace instead of failing.
    return " ".join(str(value).split())


def render_fields(sig: Signature, example: Example, *, include_outputs: bool) -> str:
    lines = [f"{f.prefix}: {_one_line(example.inputs.get(f.name, ''))}" for f in sig.inputs]
    if include_outputs:
        lines += [f"{f.prefix}: {_one_line(example.outputs.get(f.name, ''))}" for f in sig.outputs]
    return "\n".join(lines)


def render_prompt(
    sig: Signature,
    instructions: str,
    demos: tuple[Example, ...] | list[Example],
    inputs: dict[str, str],
) -> str:
    fmt = [FORMAT_HEADER, ""]
    for f in (*sig.inputs, *sig.outputs):
        fmt.append(f"{f.prefix}: {f.desc or '${' + f.name + '}'}")
    blocks = [instructions.strip() or "Complete the task.", "\n".join(fmt)]
    blocks += [render_fields(sig, d, include_outputs=True) for d in demos]
    query = render_fields(sig, Example(inputs), include_outputs=False)
    blocks.append(f"{query}\n{sig.outputs[0].prefix}:")
    return SEPARATOR.join(blocks)


def parse_fields(text: str, prefixes: dict[str, str]) -> dict[str, str]:
    """Parse ``Prefix: value`` lines. ``prefixes`` maps prefix -> field name.

    A value runs until the next line that starts with a known prefix, so a
    model that wraps a long reasoning field across lines is still parsed.
    The first occurrence of a field wins: models sometimes keep going and
    invent a *next* example, and we must not let that overwrite the answer.
    """
    pattern = re.compile(
        r"^\s*(" + "|".join(re.escape(p) for p in sorted(prefixes, key=len, reverse=True)) + r")\s*:\s?(.*)$",
        re.IGNORECASE,
    )
    lower = {p.lower(): name for p, name in prefixes.items()}
    out: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.strip() == "---":
            break  # the model started a new block; everything after is noise
        m = pattern.match(line)
        if m:
            name = lower[m.group(1).lower()]
            if name in out:
                break
            out[name] = m.group(2).strip()
            current = name
        elif current is not None and line.strip():
            out[current] = (out[current] + " " + line.strip()).strip()
    return out


def parse_completion(sig: Signature, completion: str) -> dict[str, str]:
    """Turn the raw completion into ``{output_name: value}``.

    The prompt ends with ``"<FirstOutput>:"`` so most models answer with just the
    value; some repeat the prefix. Handle both by re-attaching the prefix when
    it is missing. Missing fields come back as empty strings (the metric
    decides how bad that is) rather than raising mid-evaluation.
    """
    first = sig.outputs[0].prefix
    text = completion.lstrip()
    if not text.lower().startswith(first.lower() + ":"):
        text = f"{first}: {text}"
    parsed = parse_fields(text, {f.prefix: f.name for f in sig.outputs})
    return {f.name: parsed.get(f.name, "") for f in sig.outputs}


def split_prompt(prompt: str) -> tuple[str, str, list[str], str]:
    """Inverse of :func:`render_prompt`: (instructions, format, demo blocks, query).

    Used by the mock provider to "read" a prompt the way a model would.
    """
    blocks = prompt.split(SEPARATOR)
    if len(blocks) < 3 or not blocks[1].startswith(FORMAT_HEADER):
        raise ValueError("not a promptopt prompt")
    return blocks[0], blocks[1], blocks[2:-1], blocks[-1]
