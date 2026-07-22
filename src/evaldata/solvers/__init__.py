"""Solvers: wrappers for the AI system under test (`EvalCase` -> `SolverOutput`)."""

from evaldata.solvers.base import Solver, SuccessfulSolver
from evaldata.solvers.callable import CallableSolver
from evaldata.solvers.prompt import SCHEMA_PROMPT_TEMPLATE, PromptSolver

__all__ = ["SCHEMA_PROMPT_TEMPLATE", "CallableSolver", "PromptSolver", "Solver", "SuccessfulSolver"]
