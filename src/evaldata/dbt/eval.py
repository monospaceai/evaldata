"""Run Semantic Layer cases through a solver and scorers: per-case, pytest, and batch runners."""

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

from evaldata.core import BenchmarkSummary
from evaldata.dbt.semantic_layer import MetricCase, MetricScorer, MetricSolver, MetricSolverOutput
from evaldata.reporting.collector import CaseReport, record


def evaluate_metric_case(case: MetricCase, solver: MetricSolver, *, scorers: Sequence[MetricScorer]) -> CaseReport:
    """Run `case` through `solver` and `scorers`, returning its outcome.

    Args:
        case: The eval case to run.
        solver: The solver that produces a metric query for the case.
        scorers: Scorers applied to the candidate query; all must pass for the case to pass.

    Returns:
        A `CaseReport` with per-scorer results, or a solver error when the solver failed.
    """
    return _score_metric_output(case, solver.solve(case), scorers=scorers)


def _score_metric_output(
    case: MetricCase, output: MetricSolverOutput, *, scorers: Sequence[MetricScorer]
) -> CaseReport:
    """Score a solved metric `output`, returning its outcome.

    Args:
        case: The eval case the output belongs to.
        output: The solver output to score.
        scorers: Scorers applied to the candidate query; all must pass for the case to pass.

    Returns:
        A `CaseReport` with per-scorer results, or a solver error when the solver failed.

    Raises:
        AssertionError: If the solver returns neither a query nor an error (unreachable).
    """
    if output.error is not None:
        return CaseReport(id=case.id, input=case.input, passed=False, error=output.error)
    query = output.query
    if query is None:  # pragma: no cover - the MetricSolverOutput validator guarantees query XOR error
        msg = f"evaldata metric case {case.id!r}: solver returned neither query nor error"
        raise AssertionError(msg)
    scores = [scorer.score(case, query) for scorer in scorers]
    return CaseReport(id=case.id, input=case.input, passed=all(s.passed for s in scores), scores=scores)


def assert_metric_eval(case: MetricCase, solver: MetricSolver, *, scorers: Sequence[MetricScorer]) -> None:
    """Run `case` through `solver` and `scorers`; record the outcome and raise on any failure.

    Args:
        case: The eval case to run.
        solver: The solver that produces a metric query for the case.
        scorers: Scorers applied to the candidate query; all must pass.

    Raises:
        AssertionError: If the solver failed or any scorer did not pass.
    """
    report = evaluate_metric_case(case, solver, scorers=scorers)
    record(report)
    if report.error is not None:
        msg = f"evaldata solver error for {case.id!r}: {report.error.message}"
        raise AssertionError(msg)
    failures = [s for s in report.scores if not s.passed]
    if failures:
        detail = "; ".join(f"{s.scorer}={s.verdict} ({s.explanation})" for s in failures)
        msg = f"evaldata metric case {case.id!r} failed: {detail}"
        raise AssertionError(msg)


def run_metric_benchmark(
    cases: Iterable[MetricCase],
    solver: MetricSolver,
    *,
    scorers: Sequence[MetricScorer],
    limit: int | None = None,
    max_concurrency: int = 1,
) -> BenchmarkSummary:
    """Run `cases` through `solver` and `scorers` and return aggregate accuracy.

    With `max_concurrency` above 1, the solver calls run on a thread pool while scoring stays
    serial. Reports come back in case order regardless of which solver finished first.

    Args:
        cases: The eval cases to run, in order.
        solver: The solver under test.
        scorers: Scorers applied to each case; all must pass for the case to count as passed.
        limit: Run at most this many cases, or `None` for all of them.
        max_concurrency: How many solver calls may run at once. `1` runs everything serially.

    Returns:
        A `BenchmarkSummary` with the total, the passed count, the accuracy
        (`passed / total`, or `0.0` when no cases ran), and every case report.
    """
    selected = list(islice(cases, limit))
    if max_concurrency > 1:
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            outputs = list(pool.map(solver.solve, selected))
        reports = [
            _score_metric_output(case, output, scorers=scorers) for case, output in zip(selected, outputs, strict=True)
        ]
    else:
        reports = [evaluate_metric_case(case, solver, scorers=scorers) for case in selected]
    total = len(reports)
    passed = sum(1 for report in reports if report.passed)
    accuracy = passed / total if total else 0.0
    return BenchmarkSummary(total=total, passed=passed, accuracy=accuracy, cases=reports)
