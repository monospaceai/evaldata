"""Eval orchestration and the pytest-facing assertion.

``assert_eval`` chains the slice end-to-end: the Solver produces SQL, the platform
adapter executes it, each Scorer compares the result, and any failure is raised as an
``AssertionError`` carrying a readable diagnostic. The library stays errors-as-values
throughout (``ExecutionResult.error``, ``ScoreResult.passed``); only this thin wrapper
raises, because raising *is* pytest's failure protocol.

Adapter resolution (GE-style dual rule): an explicitly passed ``adapter`` always wins —
typically a pytest fixture that owns its own connection lifecycle. When ``adapter`` is
omitted, the live adapter is resolved from the case's ``PlatformRef`` via
``platforms.registry.resolve``, which caches it session-globally and closes it at session
end (the pytest plugin's ``pytest_sessionfinish`` hook). So ``assert_eval`` never closes a
resolved adapter mid-run — reuse across cases is the point — and never closes a
caller-supplied one (the caller owns it).

Message composition follows prevailing practice (GE/DeepEval/Inspect): the originating
input/SQL is *not* stored on ``ScoreResult`` — it is composed (in ``reporting.terminal``)
from the case, the solver output, and the execution result, alongside the structured diff.
"""

from collections.abc import Sequence

from data_eval.platforms.base import PlatformAdapter
from data_eval.platforms.registry import resolve
from data_eval.reporting.collector import CaseReport, record
from data_eval.reporting.terminal import render_failure, render_solver_error
from data_eval.scorers.base import Scorer
from data_eval.solvers.base import Solver
from data_eval.types import EvalCase


def assert_eval(
    case: EvalCase,
    solver: Solver,
    *,
    scorers: Sequence[Scorer],
    adapter: PlatformAdapter | None = None,
) -> None:
    """Run ``case`` through ``solver`` + a platform adapter + ``scorers``; raise on any failure.

    Solves the case, executes the produced SQL, scores the result with each scorer, and
    raises ``AssertionError`` with a composed diagnostic if any scorer fails. The adapter is
    the explicitly passed ``adapter`` if given, otherwise resolved (and session-cached) from
    ``case.platform``. Returns ``None`` on success (pytest-friendly).
    """
    output = solver.solve(case)
    if output.error is not None:
        error = output.error
        record(CaseReport(id=case.id, input=case.input, passed=False, error=f"solver error [{error.kind}]"))
        raise AssertionError(render_solver_error(case, error))
    sql = output.output
    if sql is None:  # invariant: error is None implies output is set (SolverOutput validator)
        raise AssertionError(f"data-eval case {case.id!r}: solver returned neither output nor error")
    live = adapter if adapter is not None else resolve(case.platform)
    result = live.execute(sql)
    scores = [scorer.score(case, output, result) for scorer in scorers]
    failures = [s for s in scores if not s.passed]
    record(CaseReport(id=case.id, input=case.input, passed=not failures, scores=list(scores)))
    if failures:
        raise AssertionError(render_failure(case, output, result, failures))
