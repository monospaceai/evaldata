"""Unit tests for `SqlEquivalence` — the pydantic-evals SQL-equivalence evaluator."""

from collections.abc import Iterator
from typing import Any

import anyio
import pytest
from pydantic_evals.evaluators import EvaluatorContext
from pydantic_evals.otel import SpanTreeRecordingError

from evaldata.platforms.registry import close_all, duckdb_platform, resolve
from evaldata.pydantic_evals import SqlEquivalence
from evaldata.types import (
    ExpectationSuite,
    GoldQuery,
    PlatformRef,
    RowCountExpectation,
    TypedResultSet,
    UntypedResultSet,
)
from pydantic_evals import Case, Dataset

pytestmark = pytest.mark.unit

_SPAN = SpanTreeRecordingError("no otel recorded")


@pytest.fixture
def platforms() -> Iterator[None]:
    """Tear down every adapter opened during the test (each test uses a unique platform name)."""
    yield
    close_all()


def _seed(name: str) -> PlatformRef:
    """Build a uniquely-named DuckDB platform seeded with a small `t(id, region)` table."""
    platform = duckdb_platform(name)
    resolve(platform).execute(
        "CREATE TABLE t (id INTEGER, region VARCHAR); INSERT INTO t VALUES (1, 'east'), (2, 'west'), (3, 'east')"
    )
    return platform


def _ctx(
    output: Any,
    expected: Any,
    *,
    name: str | None = "case-1",
    inputs: Any = "the question",
) -> EvaluatorContext[Any, str, Any]:
    """Construct an `EvaluatorContext` the way pydantic-evals does, with inert telemetry."""
    return EvaluatorContext(
        name=name,
        inputs=inputs,
        metadata=None,
        expected_output=expected,
        output=output,
        duration=0.0,
        _span_tree=_SPAN,
        attributes={},
        metrics={},
    )


@pytest.mark.usefixtures("platforms")
class TestEvaluate:
    def test_correct_sql_vs_gold_str_passes(self) -> None:
        platform = _seed("pe_correct")
        result = SqlEquivalence(platform=platform).evaluate(
            _ctx("SELECT id, region FROM t", "SELECT id, region FROM t")
        )
        assert result.value is True
        assert result.reason

    def test_wrong_sql_vs_gold_str_fails_with_mismatch_reason(self) -> None:
        platform = _seed("pe_wrong")
        result = SqlEquivalence(platform=platform).evaluate(
            _ctx("SELECT id, region FROM t WHERE id = 1", "SELECT id, region FROM t")
        )
        assert result.value is False
        assert result.reason
        assert "missing" in result.reason
        assert result.reason != "queries are not equivalent"

    def test_gold_query_instance_passes(self) -> None:
        platform = _seed("pe_goldquery")
        result = SqlEquivalence(platform=platform).evaluate(_ctx("SELECT id FROM t", GoldQuery(sql="SELECT id FROM t")))
        assert result.value is True
        assert result.reason

    def test_untyped_result_set_expected(self) -> None:
        platform = _seed("pe_untyped")
        expected = UntypedResultSet(rows=[{"id": 1, "region": "east"}])
        result = SqlEquivalence(platform=platform).evaluate(_ctx("SELECT id, region FROM t WHERE id = 1", expected))
        assert result.value is True
        assert result.reason

    def test_typed_result_set_expected(self) -> None:
        platform = _seed("pe_typed")
        expected = TypedResultSet.model_validate({"rows": [{"id": 1}], "schema": [{"name": "id", "type": "INTEGER"}]})
        result = SqlEquivalence(platform=platform).evaluate(_ctx("SELECT id FROM t WHERE id = 1", expected))
        assert result.value is True
        assert result.reason

    def test_invalid_expected_kind_raises(self) -> None:
        platform = _seed("pe_badexpected")
        suite = ExpectationSuite(expectations=[RowCountExpectation(exact=3)])
        with pytest.raises(ValueError, match="expected_output"):
            SqlEquivalence(platform=platform).evaluate(_ctx("SELECT id FROM t", suite))

    def test_non_str_expected_raises(self) -> None:
        platform = _seed("pe_intexpected")
        with pytest.raises(ValueError, match="expected_output"):
            SqlEquivalence(platform=platform).evaluate(_ctx("SELECT id FROM t", 42))

    def test_empty_output_raises(self) -> None:
        platform = _seed("pe_emptyoutput")
        with pytest.raises(ValueError, match="non-empty SQL string"):
            SqlEquivalence(platform=platform).evaluate(_ctx("   ", "SELECT id FROM t"))

    def test_non_str_output_raises(self) -> None:
        platform = _seed("pe_nonstroutput")
        with pytest.raises(ValueError, match="non-empty SQL string"):
            SqlEquivalence(platform=platform).evaluate(_ctx(None, "SELECT id FROM t"))

    def test_invalid_sql_fails_without_raising(self) -> None:
        platform = _seed("pe_syntax")
        result = SqlEquivalence(platform=platform).evaluate(_ctx("SELCT bogus", "SELECT id FROM t"))
        assert result.value is False
        assert "query execution failed" in result.reason

    def test_question_falls_back_to_name_when_inputs_blank(self) -> None:
        platform = duckdb_platform("pe_blank_inputs")
        result = SqlEquivalence(platform=platform).evaluate(
            _ctx("SELECT 1 AS x", "SELECT 1 AS x", name="named-case", inputs="   ")
        )
        assert result.value is True

    def test_question_and_id_default_when_inputs_and_name_absent(self) -> None:
        platform = duckdb_platform("pe_no_ids")
        result = SqlEquivalence(platform=platform).evaluate(
            _ctx("SELECT 1 AS x", "SELECT 1 AS x", name=None, inputs=None)
        )
        assert result.value is True


@pytest.mark.usefixtures("platforms")
def test_end_to_end_dataset_evaluate_sync() -> None:
    platform = _seed("pe_e2e")
    dataset: Dataset[str, str, Any] = Dataset(
        name="sql-equivalence",
        cases=[
            Case(name="correct", inputs="SELECT id, region FROM t", expected_output="SELECT id, region FROM t"),
            Case(
                name="wrong",
                inputs="SELECT id, region FROM t WHERE id = 1",
                expected_output="SELECT id, region FROM t",
            ),
        ],
        evaluators=[SqlEquivalence(platform=platform)],
    )
    report = dataset.evaluate_sync(lambda sql: sql)
    verdicts = {case.name: case.assertions["SqlEquivalence"].value for case in report.cases}
    assert verdicts == {"correct": True, "wrong": False}


@pytest.mark.usefixtures("platforms")
def test_evaluate_async_offloads_to_a_worker_thread() -> None:
    platform = _seed("pe_async")
    result = anyio.run(
        SqlEquivalence(platform=platform).evaluate_async, _ctx("SELECT id, region FROM t", "SELECT id, region FROM t")
    )
    assert result.value is True
    assert result.reason


@pytest.mark.usefixtures("platforms")
def test_concurrent_dataset_scores_each_case_correctly() -> None:
    # Several DuckDB cases run under Pydantic Evals concurrency: the offloaded `evaluate_async`
    # acquires a session per case, so verdicts stay correct without a serialization lock.
    platform = _seed("pe_concurrent")
    cases = [
        Case(name=f"ok-{i}", inputs="SELECT id, region FROM t", expected_output="SELECT id, region FROM t")
        for i in range(4)
    ] + [
        Case(
            name=f"bad-{i}",
            inputs="SELECT id, region FROM t WHERE id = 1",
            expected_output="SELECT id, region FROM t",
        )
        for i in range(4)
    ]
    dataset: Dataset[str, str, Any] = Dataset(
        name="concurrent", cases=cases, evaluators=[SqlEquivalence(platform=platform)]
    )
    report = dataset.evaluate_sync(lambda sql: sql, max_concurrency=4)
    verdicts = {case.name: case.assertions["SqlEquivalence"].value for case in report.cases}
    assert all(verdicts[f"ok-{i}"] is True for i in range(4))
    assert all(verdicts[f"bad-{i}"] is False for i in range(4))
