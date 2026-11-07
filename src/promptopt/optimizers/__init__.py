"""Optimizers. Each has ``compile(program, *, trainset, devset, lm) -> OptimizationResult``
and never sees the test set."""

from .base import Ledger, OptimizationResult, Scorer
from .bootstrap import BootstrapFewShot
from .instructions import InstructionSearch
from .random_search import RandomSearch

OPTIMIZERS = {"bootstrap": BootstrapFewShot, "instructions": InstructionSearch, "random_search": RandomSearch}

__all__ = ["OPTIMIZERS", "BootstrapFewShot", "InstructionSearch", "Ledger", "OptimizationResult",
           "RandomSearch", "Scorer"]
