"""A deterministic, offline stand-in for an LLM that *responds to prompt quality*.

THIS IS A SIMULATION. It exists so that the optimizers can be developed,
tested and benchmarked with no API key and bit-for-bit reproducible results.
How it reacts to a prompt is documented in :mod:`promptopt.tasks.base`:

* It parses the prompt exactly like the renderer wrote it (instructions, format,
  demos, query) and recognises the task by its output fields.
* A phenomenon (negation, multi-step arithmetic, ...) is handled correctly only
  if the instructions mention one of its keywords or a *correct* demo shows it.
  Uncovered phenomena get the task's systematic naive mistake.
* On top of that, a hash-seeded noise rate flips a few answers. Noise grows
  when the prompt carries more demos or longer instructions than the model can
  usefully attend to, which makes "add everything" a losing strategy.
* For instruction-proposal meta-prompts it acts as a decent, imperfect prompt
  engineer: it reads the listed failures, names their phenomena with hint
  sentences most of the time, and sometimes proposes generic filler or
  misdiagnoses.

Every output is a pure function of (model, seed, prompt), so runs are
reproducible and the cache in :class:`~promptopt.lm.MeteredLM` is exact.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter

from ..lm import Completion, approx_tokens
from ..prompt import FORMAT_HEADER, parse_fields, split_prompt
from ..proposer import MARKER
from ..tasks import TASKS, Skill


def _norm(s: str) -> str:
    return " ".join(s.lower().split()).strip(".")


def _prefix_key(prefix: str) -> str:
    return prefix.strip().lower().replace(" ", "_")


class MockProvider:
    def __init__(
        self,
        model: str = "mock-1",
        *,
        skills: list[Skill] | None = None,
        noise: float = 0.04,
        demo_budget: int = 6,
        instruction_word_budget: int = 80,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.skills = skills if skills is not None else [t.skill for t in TASKS.values()]
        self.noise = noise
        self.demo_budget = demo_budget
        self.instruction_word_budget = instruction_word_budget
        self.seed = seed

    # ------------------------------------------------------------------ helpers

    def _u(self, *parts: str) -> float:
        h = hashlib.sha256("\x00".join((self.model, str(self.seed), *parts)).encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    def _skill_for(self, output_prefixes: list[str]) -> Skill | None:
        keys = tuple(_prefix_key(p) for p in output_prefixes)
        for skill in self.skills:
            n = len(skill.output_fields)
            if keys[-n:] == skill.output_fields:
                return skill
        return None

    def covered(self, skill: Skill, instructions: str, demos: list[tuple[str, dict[str, str]]]) -> set[str]:
        """Phenomena this prompt teaches. Public so tests can assert on it."""
        text = instructions.lower()
        out = {p for p, kws in skill.keywords.items() if any(k in text for k in kws)}
        last = skill.output_fields[-1]
        for demo_text, demo_out in demos:
            # Only a *correct* demo teaches anything; a mislabelled one is ignored.
            # (Judged on the final answer field, the way a reader would.)
            if _norm(demo_out.get(last, "")) == _norm(skill.gold(demo_text)[last]):
                out |= skill.phenomena(demo_text)
        return out

    # ------------------------------------------------------------------ task answers

    def _answer(self, prompt: str) -> str:
        instructions, fmt, demo_blocks, query = split_prompt(prompt)
        prefixes = [ln.split(":", 1)[0].strip() for ln in fmt.splitlines()[1:] if ":" in ln]
        skill = self._skill_for(prefixes)
        if skill is None:
            return self._generic(prompt, prefixes)
        n_out = len(skill.output_fields)
        in_prefixes, out_prefixes = prefixes[:-n_out], prefixes[-n_out:]
        pmap = {p: p for p in prefixes}

        demos = []
        for block in demo_blocks:
            vals = parse_fields(block, pmap)
            demos.append((
                " ".join(vals.get(p, "") for p in in_prefixes),
                {f: vals.get(p, "") for f, p in zip(skill.output_fields, out_prefixes)},
            ))
        qvals = parse_fields(query, pmap)
        text = " ".join(qvals.get(p, "") for p in in_prefixes)

        answer = skill.solve(text, self.covered(skill, instructions, demos))
        p_noise = (
            self.noise
            + 0.015 * max(0, len(demos) - self.demo_budget)
            + 0.001 * max(0, len(instructions.split()) - self.instruction_word_budget)
        )
        if self._u("noise", prompt) < p_noise:
            answer = skill.corrupt(text, answer, random.Random(self._u("corrupt", prompt)))
        lines = [f"{p}: {answer[f]}" for f, p in zip(skill.output_fields, out_prefixes)]
        # The prompt ends with "<FirstOutput>:", so continue after it like a model would.
        return "\n".join(lines)[len(out_prefixes[0]) + 1 :].lstrip()

    def _generic(self, prompt: str, prefixes: list[str]) -> str:
        # Unknown task: deterministic placeholder output so custom signatures
        # still run end-to-end offline (scores will be meaningless, by design).
        tag = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        outs = prefixes[-1:] if prefixes else ["Answer"]
        return f"[mock {tag}]" + "".join(f"\n{p}: [mock]" for p in outs[1:])

    # ------------------------------------------------------------------ proposals

    def _propose(self, prompt: str) -> str:
        m = re.search(r"^Task outputs: (.+)$", prompt, re.MULTILINE)
        skill = self._skill_for([s.strip() for s in m.group(1).split(",")]) if m else None
        cur = re.search(r"<<<\n(.*?)\n>>>", prompt, re.DOTALL)
        current = cur.group(1).strip() if cur else ""
        nm = re.search(r"write (\d+) improved", prompt)
        n = int(nm.group(1)) if nm else 4
        if skill is None:
            return "\n".join(f"Instruction: {current} Be precise. (variant {i})" for i in range(1, n + 1))

        failure_texts = re.findall(r"^Input:\n((?:  .+\n?)+)", prompt, re.MULTILINE)
        counts: Counter[str] = Counter()
        for block in failure_texts:
            text = " ".join(ln.split(":", 1)[1].strip() for ln in block.splitlines() if ":" in ln)
            counts.update(sorted(skill.phenomena(text)))

        rng = random.Random(self._u("propose", prompt))
        out = []
        for _ in range(n):
            r = rng.random()
            if r < 0.2 or not counts:
                picks: list[str] = []
            elif r < 0.3:
                picks = [rng.choice(sorted(skill.all_phenomena()))]  # misdiagnosis
            else:
                pool = sorted(counts)  # sorted: set/str-hash order must not leak into output
                weights = [counts[p] for p in pool]
                picks = [rng.choices(pool, weights)[0]]
                if len(pool) > 1 and rng.random() < 0.5:
                    rest = [p for p in pool if p != picks[0]]
                    picks.append(rng.choices(rest, [counts[p] for p in rest])[0])
            sentences = [rng.choice(skill.hints[p]) for p in picks] or [rng.choice(skill.decoys)]
            new = [s for s in sentences if s not in current]
            out.append(" ".join([current, *new]).strip())
        return "\n".join(f"Instruction: {s}" for s in out)

    # ------------------------------------------------------------------ LM protocol

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> Completion:
        if prompt.startswith(MARKER):
            text = self._propose(prompt)
        elif FORMAT_HEADER in prompt:
            try:
                text = self._answer(prompt)
            except ValueError:
                text = self._generic(prompt, [])
        else:
            text = self._generic(prompt, [])
        # Respect max_tokens roughly, as a real provider would truncate.
        if approx_tokens(text) > max_tokens:
            text = text[: max_tokens * 4]
        return Completion(text, approx_tokens(prompt), approx_tokens(text), self.model)
