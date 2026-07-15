"""Pydantic Evals integration exposing the `SqlEquivalence` evaluator and `close_all` for connection cleanup."""

from evaldata.platforms.registry import close_all
from evaldata.pydantic_evals.evaluator import SqlEquivalence

__all__ = ["SqlEquivalence", "close_all"]
