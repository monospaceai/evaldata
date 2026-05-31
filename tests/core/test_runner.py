"""End-to-end slice test: EvalCase -> CallableSolver -> DuckDB -> ResultSetEquivalence -> assert_eval."""

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from data_eval import CallableSolver, EvalCase, PlatformRef, ResultSetEquivalence, assert_eval
from data_eval.platforms import DuckDBAdapter, duckdb_platform
from data_eval.types import ExecutionResult, ExpectedResultSet, SolverError, SolverOutput

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
        expected=ExpectedResultSet(rows=expected_rows),
        platform=PlatformRef(name="local", kind="duckdb"),
    )


@pytest.mark.unit
class TestAssertEvalEndToEnd:
    def test_passes_when_sql_is_correct(self, duck: DuckDBAdapter) -> None:
        case = _case([{"count": 2}])
        solver = CallableSolver(lambda c: _ROCK_SQL)
        assert_eval(case, solver, adapter=duck, scorers=[ResultSetEquivalence()])  # no raise == pass

    def test_fails_with_diff_and_sql_on_wrong_value(self, duck: DuckDBAdapter) -> None:
        case = _case([{"count": 99}])
        solver = CallableSolver(lambda c: _ROCK_SQL)
        with pytest.raises(AssertionError) as exc:
            assert_eval(case, solver, adapter=duck, scorers=[ResultSetEquivalence()])
        msg = str(exc.value)
        assert "rock-count" in msg
        assert _ROCK_SQL in msg  # the generated SQL is surfaced for debugging
        assert "missing rows" in msg and "extra rows" in msg  # both diff directions rendered
        assert "99" in msg  # expected-row sample (missing from actual)
        assert "2" in msg  # actual-row sample (extra vs expected)

    def test_fails_with_execution_error_on_bad_sql(self, duck: DuckDBAdapter) -> None:
        case = _case([{"count": 2}])
        solver = CallableSolver(lambda c: "SELECT * FROM does_not_exist_xyz")
        with pytest.raises(AssertionError) as exc:
            assert_eval(case, solver, adapter=duck, scorers=[ResultSetEquivalence()])
        assert "execution error" in str(exc.value)

    def test_column_alias_mismatch_surfaces_in_diff(self, duck: DuckDBAdapter) -> None:
        # AI aliases the column 'n' but the case expects 'count' -> missing/extra columns
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
            expected=ExpectedResultSet(rows=[{"count": 2}]),
            platform=duckdb_platform(name="runner-resolution", path=str(db)),
        )
        solver = CallableSolver(lambda c: "SELECT count(*) AS count FROM t WHERE genre = 'Rock'")
        assert_eval(case, solver, scorers=[ResultSetEquivalence()])  # no adapter, no raise == pass

    # An unsupported platform kind is unrepresentable (PlatformRef validation rejects it),
    # so there is no runtime "no adapter" path to test here.


class _ErrorSolver:
    """A stub Solver that always returns a typed solver error."""

    def __init__(self, error: SolverError) -> None:
        self._error = error

    def solve(self, case: EvalCase) -> SolverOutput:
        return SolverOutput(error=self._error)


class _ExplodingAdapter:
    """A stub adapter whose execute fails the test if ever called."""

    def execute(self, sql: str) -> ExecutionResult:
        msg = "adapter.execute must not be called when the solver errors"
        raise AssertionError(msg)


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
