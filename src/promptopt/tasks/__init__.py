"""Bundled tasks with synthetic data. See :mod:`promptopt.tasks.base` for the
simulation contract the MockProvider follows."""

from __future__ import annotations

from .base import Skill, Split, Task
from .entities import TASK as ENTITIES
from .math_word import TASK as MATH
from .sentiment import TASK as SENTIMENT

TASKS: dict[str, Task] = {t.name: t for t in (SENTIMENT, ENTITIES, MATH)}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; choose from {sorted(TASKS)}") from None


__all__ = ["ENTITIES", "MATH", "SENTIMENT", "TASKS", "Skill", "Split", "Task", "get_task"]
