"""Eval orchestration: the per-case pipeline, the pytest-facing `assert_eval`, and `run_benchmark`."""

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from typing import TypeAlias, overload

from pydantic import BaseModel, ConfigDict

from evaldata.platforms.base import PlatformAdapter
from evaldata.platforms.pool import PoolUnavailableError
from evaldata.platforms.registry import acquired
from evaldata.reporting.collector import (
    CaseReport,
    ExecutionFailureCaseReport,
    PassedCaseReport,
    ScoredFailureCaseReport,
    SolverFailureCaseReport,
    record,
)
from evaldata.reporting.terminal import render_failure, render_solver_error
from evaldata.scorers.base import Scorer
from evaldata.scorers.context import ScoreContext
from evaldata.scorers.query import QueryRunner
from evaldata.solvers.base import Solver, SuccessfulSolver
from evaldata.types import (
    EvalCase,
    ExecutionError,
    ExecutionFailure,
    ExecutionResult,
    ScoreResult,
    SolverFailure,
    SolverOutput,
    SolverSuccess,
    Sql,
)


@dataclass(frozen=True)
class SolverFailedCaseEvaluation:
    """A case evaluation that stopped at solver failure."""

    report: SolverFailureCaseReport
    output: SolverFailure
    result: None = None
    failures: list[ScoreResult] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutedCaseEvaluation:
    """A case evaluation that reached execution and scoring.

    Attributes:
        report: The case report.
        output: The successful solver output.
        result: The execution outcome.
        failures: The non-passing scorer results.
    """

    report: PassedCaseReport | ScoredFailureCaseReport | ExecutionFailureCaseReport
    output: SolverSuccess
    result: ExecutionResult
    failures: list[ScoreResult]


CaseEvaluation: TypeAlias = SolverFailedCaseEvaluation | ExecutedCaseEvaluation


@overload
def evaluate_case(
    case: EvalCase,
    solver: SuccessfulSolver,
    *,
    scorers: Sequence[Scorer],
    adapter: PlatformAdapter | None = None,
) -> ExecutedCaseEvaluation: ...


@overload
def evaluate_case(
    case: EvalCase,
    solver: Solver,
    *,
    scorers: Sequence[Scorer],
    adapter: PlatformAdapter | None = None,
) -> CaseEvaluation: ...


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
        A solver-failure evaluation if the solver fails; otherwise, an executed evaluation.
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
        A solver-failure evaluation if the solver fails; otherwise, an executed evaluation.
    """
    if isinstance(output, SolverFailure):
        report = SolverFailureCaseReport(id=case.id, input=case.input, error=output.error)
        return SolverFailedCaseEvaluation(report=report, output=output)
    sql = output.output
    if adapter is not None:
        return _execute_and_score(case, output, sql, scorers=scorers, adapter=adapter)
    try:
        with acquired(case.platform) as live:
            return _execute_and_score(case, output, sql, scorers=scorers, adapter=live)
    except PoolUnavailableError as error:
        return _platform_unavailable(case, output, sql, scorers, error)


def _platform_unavailable(
    case: EvalCase,
    output: SolverSuccess,
    sql: Sql,
    scorers: Sequence[Scorer],
    error: PoolUnavailableError,
) -> CaseEvaluation:
    """Score a bounded acquisition failure as an execution error rather than raising it.

    Returns:
        A failed case evaluation with a `platform_unavailable` execution result.
    """
    result = ExecutionFailure(
        latency_seconds=0.0,
        error=ExecutionError(kind="platform_unavailable", message=str(error)),
    )
    adapter = _UnavailableAdapter(result)
    dialect = case.platform.dialect or case.platform.kind
    context = ScoreContext(queries=QueryRunner(adapter, sql, dialect, None))
    scores = [scorer.score(case, output, result, context=context) for scorer in scorers]
    failures = [score for score in scores if not score.passed]
    report = ExecutionFailureCaseReport(
        id=case.id,
        input=case.input,
        error=result.error,
        scores=list(scores),
    )
    return ExecutedCaseEvaluation(report=report, output=output, result=result, failures=failures)


class _UnavailableAdapter:
    """An error-only adapter used to prevent derived scorer SQL after failed acquisition."""

    def __init__(self, result: ExecutionResult) -> None:
        """Store the unavailable-platform result returned for every execution.

        Args:
            result: The failure result to return.
        """
        self._result = result

    def execute(self, sql: str) -> ExecutionResult:
        """Return the stored acquisition failure.

        Returns:
            The unavailable-platform result.
        """
        return self._result

    def cancel(self) -> None:
        """Perform no cancellation because no query was started."""

    def close(self) -> None:
        """Perform no cleanup because no connection was acquired."""


def _execute_and_score(
    case: EvalCase,
    output: SolverSuccess,
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
    if isinstance(result, ExecutionFailure):
        report: CaseReport = ExecutionFailureCaseReport(
            id=case.id,
            input=case.input,
            error=result.error,
            scores=list(scores),
        )
    elif failures:
        report = ScoredFailureCaseReport(id=case.id, input=case.input, scores=list(scores))
    else:
        report = PassedCaseReport(id=case.id, input=case.input, scores=list(scores))
    return ExecutedCaseEvaluation(report=report, output=output, result=result, failures=failures)


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
    if isinstance(evaluation, SolverFailedCaseEvaluation):
        raise AssertionError(render_solver_error(case, evaluation.output.error))
    if not evaluation.report.passed:
        raise AssertionError(render_failure(case, evaluation.output, evaluation.result, evaluation.failures))


class BenchmarkSummary(BaseModel):
    """The aggregate outcome of running a set of cases: pass count and accuracy."""

    model_config = ConfigDict(extra="forbid")

    cases: list[CaseReport]

    @property
    def total(self) -> int:
        """Number of evaluated cases."""
        return len(self.cases)

    @property
    def passed(self) -> int:
        """Number of passing cases."""
        return sum(1 for report in self.cases if report.passed)

    @property
    def accuracy(self) -> float:
        """Fraction of evaluated cases that passed, or `0.0` when empty."""
        return self.passed / self.total if self.total else 0.0


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

    With `max_concurrency` above 1, the solver calls (the network-bound half) run on a thread
    pool while execution and scoring stay serial and case-ordered. Reports come back in case
    order regardless of which solver finished first.

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
            outputs = list(pool.map(solver.solve, selected))
        reports = [
            _score_output(case, output, scorers=scorers).report for case, output in zip(selected, outputs, strict=True)
        ]
    else:
        reports = [evaluate_case(case, solver, scorers=scorers).report for case in selected]
    return BenchmarkSummary(cases=reports)
