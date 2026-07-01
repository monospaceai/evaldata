"""Run Semantic Layer cases through a solver and scorers: per-case, pytest, and batch runners."""

from collections.abc import Iterable, Sequence

from evaldata.core import BenchmarkSummary
from evaldata.dbt.semantic_layer import MetricCase, MetricScorer, MetricSolver
from evaldata.reporting.collector import CaseReport, record


def evaluate_metric_case(case: MetricCase, solver: MetricSolver, *, scorers: Sequence[MetricScorer]) -> CaseReport:
    """Run `case` through `solver` and `scorers`, returning its outcome.

    Args:
        case: The eval case to run.
        solver: The solver that produces a metric query for the case.
        scorers: Scorers applied to the candidate query; all must pass for the case to pass.

    Returns:
        A `CaseReport` with per-scorer results, or a solver error when the solver failed.

    Raises:
        AssertionError: If the solver returns neither a query nor an error (unreachable).
    """
    output = solver.solve(case)
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
    cases: Iterable[MetricCase], solver: MetricSolver, *, scorers: Sequence[MetricScorer], limit: int | None = None
) -> BenchmarkSummary:
    """Run `cases` through `solver` and `scorers` and return aggregate accuracy.

    Args:
        cases: The eval cases to run, in order.
        solver: The solver under test.
        scorers: Scorers applied to each case; all must pass for the case to count as passed.
        limit: Run at most this many cases, or `None` for all of them.

    Returns:
        A `BenchmarkSummary` with the total, the passed count, the accuracy
        (`passed / total`, or `0.0` when no cases ran), and every case report.
    """
    reports: list[CaseReport] = []
    for index, case in enumerate(cases):
        if limit is not None and index >= limit:
            break
        reports.append(evaluate_metric_case(case, solver, scorers=scorers))
    total = len(reports)
    passed = sum(1 for report in reports if report.passed)
    accuracy = passed / total if total else 0.0
    return BenchmarkSummary(total=total, passed=passed, accuracy=accuracy, cases=reports)
