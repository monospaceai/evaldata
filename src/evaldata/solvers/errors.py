"""Translate an `LlmError` into the solver's `SolverError` vocabulary."""

from evaldata.types import LlmError, SolverError


def to_solver_error(error: LlmError) -> SolverError:
    """Translate an `LlmError` into a `SolverError`.

    `malformed_output` maps to `invalid_structured_output`; all other kinds pass through unchanged.

    Args:
        error: The `LlmError` to translate.

    Returns:
        The equivalent `SolverError`.
    """
    kind = "invalid_structured_output" if error.kind == "malformed_output" else error.kind
    return SolverError(kind=kind, message=error.message, provider=error.provider, cause=error.cause)
