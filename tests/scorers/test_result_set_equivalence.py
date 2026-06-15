"""Tests for `ResultSetEquivalence` — the in-warehouse `EXCEPT ALL` scorer."""

import pytest

from dataeval.platforms.duckdb import DuckDBAdapter
from dataeval.scorers import QueryRunner, ResultSetEquivalence, ScoreContext, Scorer
from dataeval.scorers.result_set_equivalence import SCORER_NAME
from dataeval.types import (
    Column,
    ComparisonConfig,
    EvalCase,
    ExecutionResult,
    ExpectationSuite,
    Expected,
    GoldQuery,
    PlatformRef,
    RowCountExpectation,
    Schema,
    SolverOutput,
    Sql,
    SqlType,
    TypedResultSet,
    UntypedResultSet,
)

_OUTPUT = SolverOutput(output="SELECT ...")


class _ScriptedAdapter:
    """Returns a fixed sequence of `ExecutionResult`s, one per `execute` call."""

    def __init__(self, results: list[ExecutionResult]) -> None:
        self._results = list(results)

    def execute(self, sql: str) -> ExecutionResult:
        return self._results.pop(0)

    def cancel(self) -> None: ...

    def close(self) -> None: ...


def _count(value: int) -> ExecutionResult:
    return ExecutionResult(rows=[{"c": value}], latency_seconds=0.0)


def _err(message: str) -> ExecutionResult:
    return ExecutionResult(rows=[], latency_seconds=0.0, error=message)


def _scripted_score(case: EvalCase, result: ExecutionResult, results: list[ExecutionResult]) -> object:
    context = ScoreContext(queries=QueryRunner(_ScriptedAdapter(results), Sql("SELECT 1"), "duckdb", None))
    return ResultSetEquivalence().score(case, _OUTPUT, result, context=context)


def _case(expected: Expected, comparison: ComparisonConfig | None = None) -> EvalCase:
    return EvalCase(
        id="c",
        input="q",
        expected=expected,
        platform=PlatformRef(name="x", kind="duckdb"),
        comparison=comparison or ComparisonConfig(),
    )


def _context(model: str) -> ScoreContext:
    return ScoreContext(queries=QueryRunner(DuckDBAdapter(), Sql(model), "duckdb", None))


def _score(case: EvalCase, result: ExecutionResult, model: str) -> object:
    return ResultSetEquivalence().score(case, _OUTPUT, result, context=_context(model))


@pytest.mark.unit
class TestResultSetEquivalence:
    def test_passes_on_match_untyped(self) -> None:
        case = _case(UntypedResultSet(rows=[{"count": 1297}]))
        model = "SELECT 1297 AS count"
        result = ExecutionResult(rows=[{"count": 1297}], latency_seconds=0.0)
        score = _score(case, result, model)
        assert score.scorer == SCORER_NAME
        assert score.passed is True
        assert score.diff is None

    def test_fails_on_value_mismatch_and_carries_samples(self) -> None:
        case = _case(UntypedResultSet(rows=[{"count": 1297}]))
        model = "SELECT 1298 AS count"
        result = ExecutionResult(rows=[{"count": 1298}], latency_seconds=0.0)
        score = _score(case, result, model)
        assert score.passed is False
        assert score.diff is not None
        assert score.diff.missing_row_count == 1
        assert score.diff.extra_row_count == 1
        assert score.diff.sample_missing_rows == [{"count": 1297}]
        assert score.diff.sample_extra_rows == [{"count": 1298}]

    def test_execution_error_fails_with_explanation(self) -> None:
        case = _case(UntypedResultSet(rows=[{"count": 1297}]))
        result = ExecutionResult(rows=[], latency_seconds=0.0, error="relation does not exist")
        score = _score(case, result, "SELECT 1")
        assert score.passed is False
        assert score.diff is None
        assert score.explanation is not None
        assert "relation does not exist" in score.explanation

    def test_derived_query_error_fails_without_raise(self) -> None:
        # The model references a missing table; the derived EXCEPT ALL query errors.
        case = _case(UntypedResultSet(rows=[{"n": 1}]))
        result = ExecutionResult(rows=[{"n": 1}], latency_seconds=0.0)
        score = _score(case, result, "SELECT n FROM does_not_exist_xyz")
        assert score.passed is False
        assert score.diff is None
        assert score.explanation is not None

    def test_distinct_null_equality_without_key_is_rejected(self) -> None:
        case = _case(UntypedResultSet(rows=[{"n": None}]), ComparisonConfig(null_equality="distinct"))
        result = ExecutionResult(rows=[{"n": None}], latency_seconds=0.0)
        score = _score(case, result, "SELECT NULL AS n")
        assert score.passed is False
        assert score.diff is None
        assert score.explanation is not None
        assert "requires a match_key" in score.explanation

    def test_typed_path_detects_type_mismatch(self) -> None:
        case = _case(TypedResultSet(rows=[{"n": 1}], schema=[Column(name="n", type="INTEGER")]))
        result = ExecutionResult(
            rows=[{"n": 1}],
            schema=[Column(name="n", type=SqlType.parse("BIGINT", "duckdb"))],
            latency_seconds=0.0,
        )
        score = _score(case, result, "SELECT CAST(1 AS BIGINT) AS n")
        assert score.passed is False
        assert score.diff is not None
        assert len(score.diff.type_mismatches) == 1
        assert score.diff.type_mismatches[0].column == "n"

    def test_typed_path_treats_aliased_types_as_equal(self) -> None:
        case = _case(TypedResultSet(rows=[{"n": 1}], schema=[Column(name="n", type="INT8")]))
        result = ExecutionResult(
            rows=[{"n": 1}],
            schema=[Column(name="n", type=SqlType.parse("BIGINT", "duckdb"))],
            latency_seconds=0.0,
        )
        score = _score(case, result, "SELECT CAST(1 AS BIGINT) AS n")
        assert score.passed is True

    def test_unparseable_expected_type_does_not_raise(self) -> None:
        # An expected schema with a type SQLGlot cannot parse must not raise; the cell
        # degrades to a bare literal and scoring returns a result (errors-as-values).
        case = _case(
            TypedResultSet(rows=[{"n": 1}], schema=[Column(name="n", type=SqlType.parse("MY_CUSTOM_TYPE", "duckdb"))])
        )
        result = ExecutionResult(rows=[{"n": 1}], latency_seconds=0.0)
        score = _score(case, result, "SELECT 1 AS n")
        assert score.scorer == SCORER_NAME
        assert score.passed is True

    def test_missing_column_carries_signal(self) -> None:
        case = _case(UntypedResultSet(rows=[{"a": 1, "b": 2}]))
        result = ExecutionResult(rows=[{"a": 1}], latency_seconds=0.0)
        score = _score(case, result, "SELECT 1 AS a")
        assert score.passed is False
        assert score.diff is not None
        assert score.diff.missing_columns == ["b"]

    def test_raises_on_non_result_set_expected(self) -> None:
        case = _case(ExpectationSuite(expectations=[RowCountExpectation(exact=1)]))
        result = ExecutionResult(rows=[{"n": 1}], latency_seconds=0.0)
        with pytest.raises(TypeError, match="UntypedResultSet, TypedResultSet, or GoldQuery"):
            _score(case, result, "SELECT 1")

    def test_empty_vs_empty_without_schema_passes(self) -> None:
        # No shared columns and no rows: the diff is empty and no derived query runs.
        case = _case(UntypedResultSet(rows=[]))
        result = ExecutionResult(rows=[], latency_seconds=0.0)
        score = _scripted_score(case, result, [])
        assert score.passed is True
        assert score.diff is None

    def test_second_count_query_error_fails(self) -> None:
        # First (missing) count succeeds, second (extra) count errors -> failing result.
        case = _case(UntypedResultSet(rows=[{"n": 1}]))
        result = ExecutionResult(rows=[{"n": 1}], latency_seconds=0.0)
        score = _scripted_score(case, result, [_count(0), _err("boom")])
        assert score.passed is False
        assert score.diff is None
        assert score.explanation is not None
        assert "boom" in score.explanation

    def test_missing_sample_query_error_fails(self) -> None:
        # Missing count > 0 but the missing-sample query errors -> failing result.
        case = _case(UntypedResultSet(rows=[{"n": 1}]))
        result = ExecutionResult(rows=[{"n": 2}], latency_seconds=0.0)
        score = _scripted_score(case, result, [_count(1), _count(0), _err("sample boom")])
        assert score.passed is False
        assert score.diff is None
        assert "sample boom" in (score.explanation or "")

    def test_extra_sample_query_error_fails(self) -> None:
        # Extra count > 0 but the extra-sample query errors -> failing result.
        case = _case(UntypedResultSet(rows=[{"n": 1}]))
        result = ExecutionResult(rows=[{"n": 2}], latency_seconds=0.0)
        score = _scripted_score(case, result, [_count(0), _count(1), _err("extra boom")])
        assert score.passed is False
        assert score.diff is None
        assert "extra boom" in (score.explanation or "")

    def test_satisfies_scorer_protocol(self) -> None:
        assert isinstance(ResultSetEquivalence(), Scorer)


@pytest.mark.unit
class TestGoldQueryPath:
    """The `GoldQuery` path over a real DuckDB adapter: the gold query is run in-warehouse."""

    def test_matching_gold_passes(self) -> None:
        case = _case(GoldQuery(sql="SELECT 1 AS n UNION ALL SELECT 2 AS n"))
        model = "SELECT 2 AS n UNION ALL SELECT 1 AS n"
        result = ExecutionResult(rows=[{"n": 2}, {"n": 1}], latency_seconds=0.0)
        score = _score(case, result, model)
        assert score.passed is True
        assert score.diff is None

    def test_mismatching_gold_fails_with_diff(self) -> None:
        case = _case(GoldQuery(sql="SELECT 1 AS n"))
        model = "SELECT 2 AS n"
        result = ExecutionResult(rows=[{"n": 2}], latency_seconds=0.0)
        score = _score(case, result, model)
        assert score.passed is False
        assert score.diff is not None
        assert score.diff.missing_row_count == 1
        assert score.diff.extra_row_count == 1
        assert score.diff.sample_missing_rows == [{"n": 1}]
        assert score.diff.sample_extra_rows == [{"n": 2}]
        assert score.diff.expected_row_count == 1
        assert score.diff.actual_row_count == 1

    def test_gold_query_error_is_attributed_failure(self) -> None:
        case = _case(GoldQuery(sql="SELECT n FROM does_not_exist_xyz"))
        result = ExecutionResult(rows=[{"n": 1}], latency_seconds=0.0)
        score = _score(case, result, "SELECT 1 AS n")
        assert score.passed is False
        assert score.diff is None
        assert score.explanation is not None
        assert score.explanation.startswith("gold query failed:")
        assert score.metadata.get("gold_query_failed") is True

    def test_gold_count_error_is_attributed_failure(self) -> None:
        # The schema probe (LIMIT 0) succeeds; the COUNT(*) over the gold query errors.
        case = _case(GoldQuery(sql="SELECT 1 AS n"))
        result = ExecutionResult(rows=[{"n": 1}], latency_seconds=0.0)
        schema = Schema(root=[Column(name="n", type=SqlType.parse("INTEGER", "duckdb"))])
        probe = ExecutionResult(rows=[], schema=schema, latency_seconds=0.0)
        score = _scripted_score(case, result, [probe, _err("count boom")])
        assert score.passed is False
        assert score.diff is None
        assert score.explanation is not None
        assert "gold query failed: count boom" in score.explanation
        assert score.metadata.get("gold_query_failed") is True

    def test_gold_typed_type_mismatch_detected(self) -> None:
        # Gold yields BIGINT; the model yields INTEGER — a type mismatch over the gold schema.
        case = _case(GoldQuery(sql="SELECT CAST(1 AS BIGINT) AS n"))
        result = ExecutionResult(
            rows=[{"n": 1}],
            schema=Schema(root=[Column(name="n", type=SqlType.parse("INTEGER", "duckdb"))]),
            latency_seconds=0.0,
        )
        score = _score(case, result, "SELECT CAST(1 AS INTEGER) AS n")
        assert score.passed is False
        assert score.diff is not None
        assert len(score.diff.type_mismatches) == 1
        assert score.diff.type_mismatches[0].column == "n"

    def test_keyed_gold_passes(self) -> None:
        case = _case(
            GoldQuery(sql="SELECT 1 AS id, 10 AS v UNION ALL SELECT 2, 20"),
            ComparisonConfig(match_key=["id"]),
        )
        model = "SELECT 2 AS id, 20 AS v UNION ALL SELECT 1, 10"
        result = ExecutionResult(rows=[{"id": 2, "v": 20}, {"id": 1, "v": 10}], latency_seconds=0.0)
        score = _score(case, result, model)
        assert score.passed is True
        assert score.diff is None

    def test_keyed_gold_per_column_mismatch(self) -> None:
        case = _case(
            GoldQuery(sql="SELECT 1 AS id, 10 AS v"),
            ComparisonConfig(match_key=["id"]),
        )
        result = ExecutionResult(rows=[{"id": 1, "v": 99}], latency_seconds=0.0)
        score = _score(case, result, "SELECT 1 AS id, 99 AS v")
        assert score.passed is False
        assert score.diff is not None
        assert len(score.diff.column_mismatches) == 1
        assert score.diff.column_mismatches[0].column == "v"

    def test_gold_float_tolerance_passes(self) -> None:
        case = _case(
            GoldQuery(sql="SELECT CAST(1.0 AS DOUBLE) AS v"),
            ComparisonConfig(float_tolerance=0.5),
        )
        result = ExecutionResult(rows=[{"v": 1.4}], latency_seconds=0.0)
        score = _score(case, result, "SELECT CAST(1.4 AS DOUBLE) AS v")
        assert score.passed is True

    def test_keyed_gold_null_equality_distinct(self) -> None:
        case = _case(
            GoldQuery(sql="SELECT 1 AS id, CAST(NULL AS INTEGER) AS v"),
            ComparisonConfig(match_key=["id"], null_equality="distinct"),
        )
        result = ExecutionResult(rows=[{"id": 1, "v": None}], latency_seconds=0.0)
        score = _score(case, result, "SELECT 1 AS id, CAST(NULL AS INTEGER) AS v")
        assert score.passed is False
        assert score.diff is not None
        assert len(score.diff.column_mismatches) == 1
        assert score.diff.column_mismatches[0].column == "v"


def _stats(missing: int, extra: int, *mismatches: int) -> ExecutionResult:
    """One-row keyed-diff stats: `missing`, `extra`, then a count per value column."""
    row: dict[str, object] = {"missing": missing, "extra": extra}
    for index, count in enumerate(mismatches):
        row[f"m{index}"] = count
    return ExecutionResult(rows=[row], latency_seconds=0.0)


@pytest.mark.unit
class TestKeyedPath:
    """Error-injection branches of the keyed `FULL OUTER JOIN` path (via the scripted adapter)."""

    @staticmethod
    def _keyed_case(key: list[str]) -> EvalCase:
        return _case(UntypedResultSet(rows=[{"id": 1, "v": 10}]), ComparisonConfig(match_key=key))

    def test_dupes_probe_error_fails(self) -> None:
        # The expected-side duplicate-key probe (the first derived query) errors.
        case = self._keyed_case(["id"])
        result = ExecutionResult(rows=[{"id": 1, "v": 10}], latency_seconds=0.0)
        score = _scripted_score(case, result, [_err("dupe boom")])
        assert score.passed is False
        assert score.diff is None
        assert "dupe boom" in (score.explanation or "")

    def test_stats_query_error_fails(self) -> None:
        # Both dup probes return 0; the stats aggregate errors.
        case = self._keyed_case(["id"])
        result = ExecutionResult(rows=[{"id": 1, "v": 10}], latency_seconds=0.0)
        score = _scripted_score(case, result, [_count(0), _count(0), _err("stats boom")])
        assert score.passed is False
        assert score.diff is None
        assert "stats boom" in (score.explanation or "")

    def test_missing_sample_error_fails(self) -> None:
        # missing > 0 but the missing-sample query errors.
        case = self._keyed_case(["id"])
        result = ExecutionResult(rows=[{"id": 1, "v": 10}], latency_seconds=0.0)
        score = _scripted_score(case, result, [_count(0), _count(0), _stats(1, 0, 0), _err("missing boom")])
        assert score.passed is False
        assert score.diff is None
        assert "missing boom" in (score.explanation or "")

    def test_extra_sample_error_fails(self) -> None:
        # extra > 0 but the extra-sample query errors.
        case = self._keyed_case(["id"])
        result = ExecutionResult(rows=[{"id": 1, "v": 10}], latency_seconds=0.0)
        score = _scripted_score(case, result, [_count(0), _count(0), _stats(0, 1, 0), _err("extra boom")])
        assert score.passed is False
        assert score.diff is None
        assert "extra boom" in (score.explanation or "")

    def test_counts_read_by_name_not_position(self) -> None:
        # The stats row is keyed by alias in a scrambled order: a positional reader would
        # swap missing/extra and misattribute the per-column count.
        case = self._keyed_case(["id"])
        result = ExecutionResult(rows=[{"id": 1, "v": 10}], latency_seconds=0.0)
        scrambled = ExecutionResult(rows=[{"m0": 7, "extra": 5, "missing": 3}], latency_seconds=0.0)
        missing_sample = ExecutionResult(rows=[{"id": 1, "v": 10}], latency_seconds=0.0)
        extra_sample = ExecutionResult(rows=[{"id": 2, "v": 20}], latency_seconds=0.0)
        score = _scripted_score(case, result, [_count(0), _count(0), scrambled, missing_sample, extra_sample])
        assert score.diff is not None
        assert score.diff.missing_row_count == 3
        assert score.diff.extra_row_count == 5
        assert len(score.diff.column_mismatches) == 1
        assert score.diff.column_mismatches[0].column == "v"
        assert score.diff.column_mismatches[0].unexpected_count == 7
