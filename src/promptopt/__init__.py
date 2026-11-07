"""promptopt: DSPy-style automatic prompt optimization without the framework.

Declare a task as a :class:`Signature`, give a metric and train/dev data, pick
an optimizer, get back a compiled :class:`Predict` program that serializes to
versioned, diffable JSON.
"""

from .evaluate import EvalResult, evaluate
from .lm import JsonlCache, LM, LMError, MemoryCache, MeteredLM, Price, Usage
from .optimizers import BootstrapFewShot, InstructionSearch, OptimizationResult, RandomSearch
from .program import Predict, Prediction, diff_programs
from .providers import AnthropicProvider, MockProvider, OpenAICompatProvider
from .report import HoldoutReport, holdout_report
from .signature import Example, Field, Signature

__version__ = "0.1.0"

__all__ = [
    "AnthropicProvider", "BootstrapFewShot", "EvalResult", "Example", "Field", "HoldoutReport",
    "InstructionSearch", "JsonlCache", "LM", "LMError", "MemoryCache", "MeteredLM", "MockProvider",
    "OpenAICompatProvider", "OptimizationResult", "Predict", "Prediction", "Price", "RandomSearch",
    "Signature", "Usage", "diff_programs", "evaluate", "holdout_report",
]
