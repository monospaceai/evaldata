"""Eval orchestration and the pytest-facing `assert_eval`."""

from collections.abc import Sequence

from dataeval.platforms.base import PlatformAdapter
from dataeval.platforms.registry import resolve
from dataeval.reporting.collector import CaseReport, record
from dataeval.reporting.terminal import render_failure, render_solver_error
from dataeval.scorers.base import Scorer
from dataeval.scorers.context import ScoreContext
from dataeval.scorers.query import QueryRunner
from dataeval.solvers.base import Solver
from dataeval.types import EvalCase


def assert_eval(
    case: EvalCase,
    solver: Solver,
    *,
    scorers: Sequence[Scorer],
    adapter: PlatformAdapter | None = None,
) -> None:
    """Run `case` through `solver` + a platform adapter + `scorers`; raise on any failure.

    Solves the case, executes the produced SQL, and scores the result with each scorer.
    The adapter is the explicitly passed `adapter` if given, otherwise resolved (and
    session-cached) from `case.platform`. Execution is bounded by `case.cost_budget`'s
    `max_seconds`: an overrunning query is cancelled and scored as an execution failure.

    Args:
        case: The eval case to run.
        solver: The solver that produces SQL for the case.
        scorers: Scorers applied to the execution result; all must pass.
        adapter: A platform adapter to execute against. If omitted, one is resolved and
            session-cached from `case.platform`.

    Raises:
        AssertionError: If the solver fails or any scorer fails, carrying a composed
            diagnostic. Raising is pytest's failure protocol.
    """
    output = solver.solve(case)
    if output.error is not None:
        error = output.error
        record(CaseReport(id=case.id, input=case.input, passed=False, error=f"solver error [{error.kind}]"))
        raise AssertionError(render_solver_error(case, error))
    sql = output.output
    if sql is None:  # pragma: no cover - unreachable: SolverOutput's validator guarantees output XOR error
        msg = f"dataeval case {case.id!r}: solver returned neither output nor error"
        raise AssertionError(msg)
    live = adapter if adapter is not None else resolve(case.platform)
    max_seconds = case.cost_budget.max_seconds if case.cost_budget is not None else None
    dialect = case.platform.dialect or case.platform.kind
    queries = QueryRunner(live, sql, dialect, max_seconds)
    result = queries.run(sql)
    context = ScoreContext(queries=queries)
    scores = [scorer.score(case, output, result, context=context) for scorer in scorers]
    failures = [s for s in scores if not s.passed]
    record(CaseReport(id=case.id, input=case.input, passed=not failures, scores=list(scores)))
    if failures:
        raise AssertionError(render_failure(case, output, result, failures))
