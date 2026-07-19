"""End-to-end tests for evaluation orchestration."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pytest

from evaldata import CallableSolver, EvalCase, PlatformRef, ResultSetEquivalence, assert_eval
from evaldata.platforms import DuckDBAdapter, duckdb_platform
from evaldata.platforms.pool import PoolUnavailableError
from evaldata.scorers import QueryRunner, ScoreContext
from evaldata.types import (
    CostBudget,
    ExecutionResult,
    ScoreResult,
    SolverError,
    SolverOutput,
    Sql,
    UntypedResultSet,
)

_ROCK_SQL = "SELECT count(*) AS count FROM tracks WHERE genre = 'Rock'"


@pytest.fixture
def duck() -> Iterator[DuckDBAdapter]:
    with DuckDBAdapter() as adapter:
        adapter.execute("CREATE TABLE tracks (id INTEGER, genre VARCHAR)")
        adapter.execute("INSERT INTO tracks VALUES (1, 'Rock'), (2, 'Rock'), (3, 'Jazz')")
        yield adapter


def _case(expected_rows: list[dict[str, object]]) -> EvalCase:
    return EvalCase(
        id="rock-count",
        input="How many tracks are in the 'Rock' genre?",
        expected=UntypedResultSet(rows=expected_rows),
        platform=PlatformRef(name="local", kind="duckdb"),
    )


@pytest.mark.unit
class TestAssertEvalEndToEnd:
    def test_passes_when_sql_is_correct(self, duck: DuckDBAdapter) -> None:
        case = _case([{"count": 2}])
        solver = CallableSolver(lambda c: _ROCK_SQL)
        assert_eval(case, solver, adapter=duck, scorers=[ResultSetEquivalence()])

    def test_fails_with_diff_and_sql_on_wrong_value(self, duck: DuckDBAdapter) -> None:
        case = _case([{"count": 99}])
        solver = CallableSolver(lambda c: _ROCK_SQL)
        with pytest.raises(AssertionError) as exc:
            assert_eval(case, solver, adapter=duck, scorers=[ResultSetEquivalence()])
        msg = str(exc.value)
        assert "rock-count" in msg
        assert _ROCK_SQL in msg
        assert "missing rows" in msg and "extra rows" in msg
        assert "99" in msg
        assert "2" in msg

    def test_fails_with_execution_error_on_bad_sql(self, duck: DuckDBAdapter) -> None:
        case = _case([{"count": 2}])
        solver = CallableSolver(lambda c: "SELECT * FROM does_not_exist_xyz")
        with pytest.raises(AssertionError) as exc:
            assert_eval(case, solver, adapter=duck, scorers=[ResultSetEquivalence()])
        assert "execution error" in str(exc.value)

    def test_column_alias_mismatch_surfaces_in_diff(self, duck: DuckDBAdapter) -> None:
        case = _case([{"count": 2}])
        solver = CallableSolver(lambda c: "SELECT count(*) AS n FROM tracks WHERE genre = 'Rock'")
        with pytest.raises(AssertionError) as exc:
            assert_eval(case, solver, adapter=duck, scorers=[ResultSetEquivalence()])
        msg = str(exc.value)
        assert "missing columns" in msg
        assert "unexpected columns" in msg


@pytest.mark.unit
class TestAssertEvalAdapterResolution:
    def test_resolves_adapter_from_platform_when_none_passed(self, tmp_path: Path) -> None:
        db = tmp_path / "t.duckdb"
        con = duckdb.connect(str(db))
        con.execute("CREATE TABLE t (genre VARCHAR)")
        con.execute("INSERT INTO t VALUES ('Rock'), ('Rock')")
        con.close()
        case = EvalCase(
            id="resolved",
            input="how many rock rows?",
            expected=UntypedResultSet(rows=[{"count": 2}]),
            platform=duckdb_platform(name="runner-resolution", path=str(db)),
        )
        solver = CallableSolver(lambda c: "SELECT count(*) AS count FROM t WHERE genre = 'Rock'")
        assert_eval(case, solver, scorers=[ResultSetEquivalence()])

    def test_pool_unavailable_becomes_an_execution_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from evaldata.core.runner import evaluate_case

        class DerivedQueryScorer:
            def score(
                self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
            ) -> ScoreResult:
                derived = context.queries.run(Sql("SELECT 1"))
                assert derived is result
                return ScoreResult(scorer="derived", verdict="fail")

        @contextmanager
        def unavailable(_: PlatformRef) -> Iterator[DuckDBAdapter]:
            platform_name = "local"
            message = "connection pool for platform 'local' is unavailable"
            raise PoolUnavailableError(platform_name, message)
            yield  # pragma: no cover - satisfies contextmanager's generator contract

        monkeypatch.setattr("evaldata.core.runner.acquired", unavailable)
        evaluation = evaluate_case(
            _case([{"count": 2}]), CallableSolver(lambda c: _ROCK_SQL), scorers=[DerivedQueryScorer()]
        )
        assert evaluation.result is not None
        assert evaluation.result.error is not None
        assert evaluation.result.error.kind == "platform_unavailable"
        assert evaluation.report.passed is False
        assert len(evaluation.failures) == 1

    def test_pool_unavailable_raises_even_when_a_custom_scorer_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PassingScorer:
            def score(
                self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
            ) -> ScoreResult:
                return ScoreResult(scorer="passing", verdict="pass")

        @contextmanager
        def unavailable(_: PlatformRef) -> Iterator[DuckDBAdapter]:
            platform_name = "local"
            message = "connection pool for platform 'local' is unavailable"
            raise PoolUnavailableError(platform_name, message)
            yield  # pragma: no cover - satisfies contextmanager's generator contract

        monkeypatch.setattr("evaldata.core.runner.acquired", unavailable)
        with pytest.raises(AssertionError, match="connection pool for platform 'local' is unavailable"):
            assert_eval(_case([{"count": 2}]), CallableSolver(lambda c: _ROCK_SQL), scorers=[PassingScorer()])


class _ErrorSolver:
    def __init__(self, error: SolverError) -> None:
        self._error = error

    def solve(self, case: EvalCase) -> SolverOutput:
        return SolverOutput(error=self._error)


class _ExplodingAdapter:
    def execute(self, sql: str) -> ExecutionResult:
        msg = "adapter.execute must not be called when the solver errors"
        raise AssertionError(msg)

    def cancel(self) -> None:
        return

    def close(self) -> None:
        return


@pytest.mark.unit
class TestAssertEvalSolverError:
    def test_solver_error_raises_without_executing(self) -> None:
        case = _case([{"count": 1}])
        solver = _ErrorSolver(SolverError(kind="auth", message="invalid api key", provider="openai"))
        with pytest.raises(AssertionError) as exc:
            assert_eval(case, solver, adapter=_ExplodingAdapter(), scorers=[ResultSetEquivalence()])
        msg = str(exc.value)
        assert "rock-count" in msg
        assert "auth" in msg
        assert "invalid api key" in msg


class _SpyScorer:
    def __init__(self) -> None:
        self.context: ScoreContext | None = None

    def score(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> ScoreResult:
        self.context = context
        return ScoreResult(scorer="spy", verdict="pass")


class _FixedVerdictScorer:
    def __init__(self, scorer: str, verdict: str) -> None:
        self._scorer = scorer
        self._verdict = verdict

    def score(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> ScoreResult:
        return ScoreResult(scorer=self._scorer, verdict=self._verdict)  # ty: ignore[invalid-argument-type]


@pytest.mark.unit
class TestAssertEvalInconclusive:
    def test_inconclusive_scorer_raises_and_renders_distinctly(self, duck: DuckDBAdapter) -> None:
        case = _case([{"count": 2}])
        solver = CallableSolver(lambda c: _ROCK_SQL)
        with pytest.raises(AssertionError) as exc:
            assert_eval(
                case,
                solver,
                adapter=duck,
                scorers=[_FixedVerdictScorer("semantic_equivalence", "inconclusive")],
            )
        msg = str(exc.value)
        assert "INCONCLUSIVE" in msg
        assert "FAIL" not in msg

    def test_fail_scorer_renders_as_fail_not_inconclusive(self, duck: DuckDBAdapter) -> None:
        case = _case([{"count": 2}])
        solver = CallableSolver(lambda c: _ROCK_SQL)
        with pytest.raises(AssertionError) as exc:
            assert_eval(
                case,
                solver,
                adapter=duck,
                scorers=[_FixedVerdictScorer("result_set_equivalence", "fail")],
            )
        msg = str(exc.value)
        assert "FAIL" in msg
        assert "INCONCLUSIVE" not in msg


@pytest.mark.unit
class TestAssertEvalContext:
    def test_injects_usable_query_runner(self, duck: DuckDBAdapter) -> None:
        case = _case([{"count": 2}])
        solver = CallableSolver(lambda c: _ROCK_SQL)
        spy = _SpyScorer()
        assert_eval(case, solver, adapter=duck, scorers=[spy])
        assert isinstance(spy.context, ScoreContext)
        assert isinstance(spy.context.queries, QueryRunner)
        derived = spy.context.queries.run(Sql("SELECT 1"))
        assert derived.error is None
        assert derived.rows == [{"1": 1}]

    def test_model_query_flows_through_runner_under_budget(self, duck: DuckDBAdapter) -> None:
        case = EvalCase(
            id="rock-count",
            input="How many tracks are in the 'Rock' genre?",
            expected=UntypedResultSet(rows=[{"count": 2}]),
            platform=PlatformRef(name="local", kind="duckdb"),
            cost_budget=CostBudget(max_seconds=30.0),
        )
        solver = CallableSolver(lambda c: _ROCK_SQL)
        assert_eval(case, solver, adapter=duck, scorers=[ResultSetEquivalence()])
