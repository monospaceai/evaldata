"""Tests for the Semantic Layer eval pipeline and the `MetricFirstDecisive` combinator."""

import time

import pytest

from evaldata.dbt import (
    MetricCase,
    MetricFirstDecisive,
    MetricQuery,
    MetricSolverOutput,
    assert_metric_eval,
    evaluate_metric_case,
    run_metric_benchmark,
)
from evaldata.types import PlatformRef, ScoreResult, SolverError

pytestmark = pytest.mark.unit

PLATFORM = PlatformRef(name="duck", kind="duckdb")
QUERY = MetricQuery(metrics=["revenue"])


def _case(id: str = "c") -> MetricCase:
    return MetricCase(id=id, input="q", gold=QUERY, platform=PLATFORM, target_dir="t")


class _Solver:
    def __init__(self, output: MetricSolverOutput) -> None:
        self._output = output

    def solve(self, case: MetricCase) -> MetricSolverOutput:
        return self._output


class _Scorer:
    def __init__(self, verdict: str, *, name: str = "s") -> None:
        self._result = _result(verdict, name)
        self.calls = 0

    def score(self, case: MetricCase, query: MetricQuery) -> ScoreResult:
        self.calls += 1
        return self._result


def _result(verdict: str, name: str = "s") -> ScoreResult:
    basis = None if verdict == "inconclusive" else "proven"
    return ScoreResult(scorer=name, verdict=verdict, basis=basis)  # type: ignore[arg-type]


def _ok_output() -> MetricSolverOutput:
    return MetricSolverOutput(query=QUERY)


def test_evaluate_reports_solver_error() -> None:
    output = MetricSolverOutput(error=SolverError(kind="auth", message="bad key"))
    report = evaluate_metric_case(_case(), _Solver(output), scorers=[_Scorer("pass")])
    assert report.passed is False
    assert report.error is not None


def test_evaluate_passes_when_all_scorers_pass() -> None:
    report = evaluate_metric_case(_case(), _Solver(_ok_output()), scorers=[_Scorer("pass")])
    assert report.passed is True


def test_evaluate_fails_when_a_scorer_does_not_pass() -> None:
    report = evaluate_metric_case(_case(), _Solver(_ok_output()), scorers=[_Scorer("pass"), _Scorer("fail")])
    assert report.passed is False


def test_assert_metric_eval_passes_silently() -> None:
    assert_metric_eval(_case(), _Solver(_ok_output()), scorers=[_Scorer("pass")])


def test_assert_metric_eval_raises_on_solver_error() -> None:
    output = MetricSolverOutput(error=SolverError(kind="auth", message="bad key"))
    with pytest.raises(AssertionError, match="solver error"):
        assert_metric_eval(_case(), _Solver(output), scorers=[_Scorer("pass")])


def test_assert_metric_eval_raises_on_scorer_failure() -> None:
    with pytest.raises(AssertionError, match="failed"):
        assert_metric_eval(_case(), _Solver(_ok_output()), scorers=[_Scorer("fail")])


def test_run_metric_benchmark_aggregates_and_limits() -> None:
    cases = [_case("a"), _case("b"), _case("c")]
    summary = run_metric_benchmark(cases, _Solver(_ok_output()), scorers=[_Scorer("pass")], limit=2)
    assert summary.total == 2
    assert summary.passed == 2
    assert summary.accuracy == 1.0


def test_run_metric_benchmark_empty_is_zero_accuracy() -> None:
    summary = run_metric_benchmark([], _Solver(_ok_output()), scorers=[_Scorer("pass")])
    assert summary.total == 0
    assert summary.accuracy == 0.0


def test_run_metric_benchmark_concurrency_preserves_order() -> None:
    class _SleepSolver:
        def solve(self, case: MetricCase) -> MetricSolverOutput:
            time.sleep(0.02 * (3 - int(case.id)))
            return _ok_output()

    cases = [_case(str(i)) for i in range(3)]
    summary = run_metric_benchmark(cases, _SleepSolver(), scorers=[_Scorer("pass")], max_concurrency=4)
    assert [c.id for c in summary.cases] == ["0", "1", "2"]
    assert summary.passed == 3


def test_run_metric_benchmark_concurrency_case_error_does_not_kill_run() -> None:
    class _MaybeErroring:
        def solve(self, case: MetricCase) -> MetricSolverOutput:
            if case.id == "b":
                return MetricSolverOutput(error=SolverError(kind="auth", message="boom"))
            return _ok_output()

    cases = [_case("a"), _case("b"), _case("c")]
    summary = run_metric_benchmark(cases, _MaybeErroring(), scorers=[_Scorer("pass")], max_concurrency=3)
    assert summary.total == 3
    assert summary.passed == 2
    reports = {r.id: r for r in summary.cases}
    assert reports["b"].error is not None


class TestMetricFirstDecisive:
    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            MetricFirstDecisive([])

    def test_returns_first_decisive_and_skips_later(self) -> None:
        second = _Scorer("fail")
        decided = MetricFirstDecisive([_Scorer("pass"), second]).score(_case(), QUERY)
        assert decided.verdict == "pass"
        assert second.calls == 0
        assert decided.metadata["first_decisive"] == [{"scorer": "s", "passed": True, "verdict": "pass"}]

    def test_returns_last_when_all_inconclusive(self) -> None:
        decided = MetricFirstDecisive([_Scorer("inconclusive"), _Scorer("inconclusive")]).score(_case(), QUERY)
        assert decided.verdict == "inconclusive"
        assert len(decided.metadata["first_decisive"]) == 2
