"""Eval orchestration: the per-case pipeline, the pytest-facing `assert_eval`, and `run_benchmark`."""

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from typing import cast

from pydantic import BaseModel, ConfigDict

from evaldata.platforms.base import PlatformAdapter
from evaldata.platforms.registry import acquired
from evaldata.reporting.collector import CaseReport, record
from evaldata.reporting.terminal import render_failure, render_solver_error
from evaldata.scorers.base import Scorer
from evaldata.scorers.context import ScoreContext
from evaldata.scorers.query import QueryRunner
from evaldata.solvers.base import Solver
from evaldata.types import EvalCase, ExecutionResult, ScoreResult, SolverOutput, Sql


@dataclass(frozen=True)
class CaseEvaluation:
    """The outcome of running one case through a solver + platform + scorers, without raising.

    Attributes:
        report: The case outcome (identity, pass/fail, per-scorer results or a solver error).
        output: The solver output, carrying either the SQL or a typed `SolverError`.
        result: The executed model result, or `None` when the solver itself failed.
        failures: The failing scorer results (empty when the case passed or the solver failed).
    """

    report: CaseReport
    output: SolverOutput
    result: ExecutionResult | None
    failures: list[ScoreResult]


def evaluate_case(
    case: EvalCase,
    solver: Solver,
    *,
    scorers: Sequence[Scorer],
    adapter: PlatformAdapter | None = None,
) -> CaseEvaluation:
    """Run `case` through `solver` + a platform adapter + `scorers`, returning the outcome.

    Solves the case, executes the produced SQL, and scores the result with each scorer. The
    adapter is the explicitly passed `adapter` if given, otherwise a session acquired from
    `case.platform`'s pool for the execute-and-score pipeline and released afterwards.
    Execution is bounded by `case.cost_budget`'s `max_seconds`: an overrunning query is
    cancelled and scored as an execution failure. Does not raise on failure and does not record
    to the run accumulator — callers decide how to surface the result.

    Args:
        case: The eval case to run.
        solver: The solver that produces SQL for the case.
        scorers: Scorers applied to the execution result; all must pass for the case to pass.
        adapter: A platform adapter to execute against. If omitted, a pool member is acquired
            from `case.platform`.

    Returns:
        A `CaseEvaluation` carrying the case report, the solver output, the execution result
        (or `None` on solver error), and any failing scorer results.
    """
    output = solver.solve(case)
    return _score_output(case, output, scorers=scorers, adapter=adapter)


def _score_output(
    case: EvalCase,
    output: SolverOutput,
    *,
    scorers: Sequence[Scorer],
    adapter: PlatformAdapter | None = None,
) -> CaseEvaluation:
    """Execute `output`'s SQL and score it, returning the outcome without raising.

    Args:
        case: The eval case the output belongs to.
        output: The solver output to execute and score.
        scorers: Scorers applied to the execution result; all must pass for the case to pass.
        adapter: A platform adapter to execute against. If omitted, a pool member is acquired
            from `case.platform` for this call and released afterward.

    Returns:
        A `CaseEvaluation` carrying the case report, the solver output, the execution result
        (or `None` on solver error), and any failing scorer results.

    Raises:
        AssertionError: If the solver returns neither output nor error (unreachable — the
            `SolverOutput` validator guarantees exactly one is set).
    """
    if output.error is not None:
        report = CaseReport(id=case.id, input=case.input, passed=False, error=output.error)
        return CaseEvaluation(report=report, output=output, result=None, failures=[])
    sql = output.output
    if sql is None:  # pragma: no cover - unreachable: SolverOutput's validator guarantees output XOR error
        msg = f"evaldata case {case.id!r}: solver returned neither output nor error"
        raise AssertionError(msg)
    if adapter is not None:
        return _execute_and_score(case, output, sql, scorers=scorers, adapter=adapter)
    with acquired(case.platform) as live:
        return _execute_and_score(case, output, sql, scorers=scorers, adapter=live)


def _execute_and_score(
    case: EvalCase,
    output: SolverOutput,
    sql: Sql,
    *,
    scorers: Sequence[Scorer],
    adapter: PlatformAdapter,
) -> CaseEvaluation:
    """Execute `sql` on `adapter` and score the result against every scorer.

    Args:
        case: The eval case the output belongs to.
        output: The solver output being scored.
        sql: The model SQL to execute.
        scorers: Scorers applied to the execution result; all must pass for the case to pass.
        adapter: The platform adapter to execute against.

    Returns:
        A `CaseEvaluation` carrying the case report, the solver output, the execution result,
        and any failing scorer results.
    """
    max_seconds = case.cost_budget.max_seconds if case.cost_budget is not None else None
    dialect = case.platform.dialect or case.platform.kind
    queries = QueryRunner(adapter, sql, dialect, max_seconds)
    result = queries.run(sql)
    context = ScoreContext(queries=queries)
    scores = [scorer.score(case, output, result, context=context) for scorer in scorers]
    failures = [s for s in scores if not s.passed]
    report = CaseReport(id=case.id, input=case.input, passed=not failures, scores=list(scores))
    return CaseEvaluation(report=report, output=output, result=result, failures=failures)


def assert_eval(
    case: EvalCase,
    solver: Solver,
    *,
    scorers: Sequence[Scorer],
    adapter: PlatformAdapter | None = None,
) -> None:
    """Run `case` through `solver` + a platform adapter + `scorers`; raise on any failure.

    Records the case outcome to the run accumulator, then raises if the solver failed or any
    scorer failed. Raising is pytest's failure protocol.

    Args:
        case: The eval case to run.
        solver: The solver that produces SQL for the case.
        scorers: Scorers applied to the execution result; all must pass.
        adapter: A platform adapter to execute against. If omitted, a pool member is acquired
            from `case.platform`.

    Raises:
        AssertionError: If the solver fails or any scorer fails, carrying a composed diagnostic.
    """
    evaluation = evaluate_case(case, solver, scorers=scorers, adapter=adapter)
    record(evaluation.report)
    if evaluation.output.error is not None:
        raise AssertionError(render_solver_error(case, evaluation.output.error))
    if evaluation.failures:
        # `result` is always set when scorers ran (the solver-error path returns before scoring).
        result = cast(ExecutionResult, evaluation.result)
        raise AssertionError(render_failure(case, evaluation.output, result, evaluation.failures))


class BenchmarkSummary(BaseModel):
    """The aggregate outcome of running a set of cases: pass count and accuracy."""

    model_config = ConfigDict(extra="forbid")

    total: int
    passed: int
    accuracy: float
    cases: list[CaseReport]


def run_benchmark(
    cases: Iterable[EvalCase],
    solver: Solver,
    *,
    scorers: Sequence[Scorer],
    limit: int | None = None,
    max_concurrency: int = 1,
) -> BenchmarkSummary:
    """Run `cases` through `solver` + `scorers` and return aggregate accuracy.

    Each case runs through `evaluate_case` (so a solver error or any failing scorer marks the
    case failed), and adapters are acquired per `case.platform` from its connection pool. Unlike
    `assert_eval`, this neither raises nor records to the run accumulator — it returns the
    aggregate for the caller to print or persist.

    With `max_concurrency` above 1, each case's solve-execute-score pipeline runs on a thread
    pool and acquires its own platform session from the per-name pool, so cases run in parallel
    up to the pool's per-engine size; excess cases block on acquire. Reports come back in case
    order regardless of which case finished first.

    Args:
        cases: The eval cases to run, in order.
        solver: The solver under test.
        scorers: Scorers applied to each case; all must pass for the case to count as passed.
        limit: Run at most this many cases, or `None` for all of them.
        max_concurrency: How many cases may run at once. `1` runs everything serially.

    Returns:
        A `BenchmarkSummary` with the total, the passed count, the accuracy
        (`passed / total`, or `0.0` when no cases ran), and every case report.
    """
    selected = list(islice(cases, limit))
    if max_concurrency > 1:
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            reports = [
                evaluation.report
                for evaluation in pool.map(lambda case: evaluate_case(case, solver, scorers=scorers), selected)
            ]
    else:
        reports = [evaluate_case(case, solver, scorers=scorers).report for case in selected]
    total = len(reports)
    passed = sum(1 for report in reports if report.passed)
    accuracy = passed / total if total else 0.0
    return BenchmarkSummary(total=total, passed=passed, accuracy=accuracy, cases=reports)
