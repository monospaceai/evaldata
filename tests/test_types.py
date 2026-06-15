"""Tests for the core public Pydantic types."""

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from dataeval.types import (
    Column,
    ColumnMismatch,
    ColumnPresenceExpectation,
    ColumnTypeExpectation,
    ComparisonConfig,
    CostBudget,
    EvalCase,
    ExecutionResult,
    Expectation,
    ExpectationOutcome,
    ExpectationSuite,
    Expected,
    GoldQuery,
    NotNullExpectation,
    PlatformRef,
    ResultSetDiff,
    RowCountExpectation,
    Schema,
    ScoreResult,
    SolverError,
    SolverOutput,
    SqlType,
    TypedResultSet,
    TypeMismatch,
    UniqueExpectation,
    UntypedResultSet,
)

ExpectedAdapter: TypeAdapter[Expected] = TypeAdapter(Expected)
ExpectationAdapter: TypeAdapter[Expectation] = TypeAdapter(Expectation)


@pytest.mark.unit
class TestPlatformRef:
    def test_minimal_construction(self) -> None:
        ref = PlatformRef(name="local", kind="duckdb")
        assert ref.name == "local"
        assert ref.kind == "duckdb"
        assert ref.dialect is None
        assert ref.config == {}

    def test_full_construction(self) -> None:
        ref = PlatformRef(
            name="prod-pg",
            kind="postgres",
            dialect="postgres",
            config={"host": "db.example.com", "dbname": "analytics"},
        )
        assert ref.config["dbname"] == "analytics"

    def test_json_schema_round_trip(self) -> None:
        ref = PlatformRef(name="local", kind="duckdb", config={"path": ":memory:"})
        restored = PlatformRef.model_validate_json(ref.model_dump_json())
        assert restored == ref

    # "snowflake" is included deliberately: it has no shipped adapter, so it is not a
    # PlatformKind and must be rejected at the boundary just like a never-supported "oracle".
    @pytest.mark.parametrize("kind", ["oracle", "snowflake"])
    def test_rejects_unknown_kind(self, kind: str) -> None:
        with pytest.raises(ValidationError):
            PlatformRef.model_validate({"name": "x", "kind": kind})

    def test_rejects_unknown_dialect(self) -> None:
        with pytest.raises(ValidationError):
            PlatformRef.model_validate({"name": "x", "kind": "duckdb", "dialect": "mysql"})

    def test_accepts_cross_platform_dialect_override(self) -> None:
        ref = PlatformRef(name="local", kind="duckdb", dialect="snowflake")
        assert ref.dialect == "snowflake"

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            PlatformRef(name="", kind="duckdb")

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            PlatformRef.model_validate({"name": "x", "kind": "duckdb", "nmae": "typo"})


@pytest.mark.unit
class TestSqlType:
    def test_string_shorthand_leaves_canonical_unset(self) -> None:
        t = SqlType.model_validate("BIGINT")
        assert t.raw == "BIGINT"
        assert t.canonical is None

    def test_parse_populates_canonical(self) -> None:
        t = SqlType.parse("BIGINT", "duckdb")
        assert t.raw == "BIGINT"
        assert t.canonical == "BIGINT"

    def test_canonical_equality_bigint_int8(self) -> None:
        assert SqlType.parse("BIGINT", "duckdb") == SqlType.parse("INT8", "duckdb")

    def test_case_insensitive(self) -> None:
        assert SqlType.parse("bigint", "duckdb") == SqlType.parse("BIGINT", "duckdb")

    def test_whitespace_insensitive(self) -> None:
        assert SqlType.parse("DECIMAL(10,2)", "duckdb") == SqlType.parse("DECIMAL(10, 2)", "duckdb")

    def test_exact_parameters_distinct(self) -> None:
        assert SqlType.parse("DECIMAL(10,2)", "duckdb") != SqlType.parse("DECIMAL", "duckdb")

    def test_cross_dialect_canonical_equality(self) -> None:
        assert SqlType.parse("BIGINT", "duckdb") == SqlType.parse("LONG", "databricks")

    def test_unparseable_falls_back_to_raw(self) -> None:
        a = SqlType.parse("UNKNOWN_EXOTIC_TYPE", "duckdb")
        b = SqlType.parse("UNKNOWN_EXOTIC_TYPE", "duckdb")
        assert a.canonical is None
        assert a == b

    def test_unparseable_different_raw_distinct(self) -> None:
        assert SqlType.parse("EXOTIC_A", "duckdb") != SqlType.parse("EXOTIC_B", "duckdb")

    def test_not_equal_to_non_sqltype(self) -> None:
        # __eq__ returns NotImplemented for a non-SqlType, so Python falls back to inequality.
        assert SqlType.parse("BIGINT", "duckdb") != 5

    def test_hashable_by_canonical(self) -> None:
        a = SqlType.parse("BIGINT", "duckdb")
        b = SqlType.parse("INT8", "duckdb")
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_hashable_by_raw_when_unparseable(self) -> None:
        a = SqlType.parse("EXOTIC_X", "duckdb")
        assert a.canonical is None
        assert hash(a) == hash(SqlType.parse("EXOTIC_X", "duckdb"))

    def test_json_round_trip(self) -> None:
        t = SqlType.parse("BIGINT", "duckdb")
        dumped = t.model_dump_json()
        assert '"raw":"BIGINT"' in dumped
        assert '"canonical":"BIGINT"' in dumped
        restored = SqlType.model_validate_json(dumped)
        assert restored == t
        assert restored.raw == t.raw and restored.canonical == t.canonical


@pytest.mark.unit
class TestColumn:
    def test_construction(self) -> None:
        col = Column(name="revenue", type="DECIMAL(10, 2)")
        assert col.name == "revenue"
        assert col.type.raw == "DECIMAL(10, 2)"
        assert col.nullable is None

    def test_nullable_tri_state(self) -> None:
        assert Column(name="a", type="INT", nullable=True).nullable is True
        assert Column(name="a", type="INT", nullable=False).nullable is False
        assert Column(name="a", type="INT").nullable is None

    def test_nested_type_string(self) -> None:
        col = Column(name="payload", type="ARRAY<STRUCT<a: INT, b: STRING>>")
        assert col.type.raw == "ARRAY<STRUCT<a: INT, b: STRING>>"

    def test_json_round_trip(self) -> None:
        col = Column(name="id", type="BIGINT")
        restored = Column.model_validate_json(col.model_dump_json())
        assert restored == col

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            Column(name="", type="INTEGER")

    def test_rejects_empty_type(self) -> None:
        with pytest.raises(ValidationError):
            Column(name="id", type="")

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Column.model_validate({"name": "id", "type": "INTEGER", "scale": 2})


@pytest.mark.unit
class TestSchema:
    def test_preserves_duplicate_names(self) -> None:
        s = Schema([Column(name="a", type="INT"), Column(name="a", type="VARCHAR")])
        assert s.names == ["a", "a"]
        assert [t.raw for t in s.types] == ["INT", "VARCHAR"]

    def test_len_index_iter(self) -> None:
        s = Schema([Column(name="a", type="INT"), Column(name="b", type="VARCHAR")])
        assert len(s) == 2
        assert s[1].name == "b"
        assert [c.name for c in s] == ["a", "b"]

    def test_names_and_types_positionally_aligned(self) -> None:
        s = Schema([Column(name="x", type="INT"), Column(name="y", type="DOUBLE")])
        assert s.names == ["x", "y"]
        assert [t.raw for t in s.types] == ["INT", "DOUBLE"]

    def test_json_array_round_trip(self) -> None:
        s = Schema([Column(name="id", type="INTEGER")])
        dumped = s.model_dump_json()
        assert dumped.startswith("[")
        assert '"schema_"' not in dumped
        restored = Schema.model_validate_json(dumped)
        assert restored == s

    def test_coerces_from_plain_list(self) -> None:
        result = ExecutionResult(
            rows=[{"id": 1}],
            schema=[Column(name="id", type="INTEGER")],
            latency_seconds=0.1,
        )
        assert isinstance(result.schema_, Schema)


@pytest.mark.unit
class TestUntypedResultSet:
    def test_minimal_construction(self) -> None:
        exp = UntypedResultSet(rows=[{"count": 1297}])
        assert exp.kind == "untyped_result_set"
        assert exp.rows == [{"count": 1297}]

    def test_empty_rows_allowed(self) -> None:
        exp = UntypedResultSet(rows=[])
        assert exp.rows == []

    def test_json_round_trip(self) -> None:
        exp = UntypedResultSet(rows=[{"x": 1}])
        restored = UntypedResultSet.model_validate_json(exp.model_dump_json())
        assert restored == exp

    def test_rejects_schema_field(self) -> None:
        with pytest.raises(ValidationError):
            UntypedResultSet.model_validate({"rows": [], "schema": []})


@pytest.mark.unit
class TestTypedResultSet:
    def test_minimal_construction(self) -> None:
        exp = TypedResultSet(rows=[], schema=[Column(name="id", type="INTEGER")])
        assert exp.kind == "typed_result_set"
        assert exp.rows == []
        assert exp.schema_.names == ["id"]
        assert exp.schema_[0].type.raw == "INTEGER"

    def test_with_rows_and_schema(self) -> None:
        exp = TypedResultSet(
            rows=[{"id": 1, "name": "rock"}],
            schema=[Column(name="id", type="BIGINT"), Column(name="name", type="VARCHAR")],
        )
        assert len(exp.schema_) == 2

    def test_schema_required(self) -> None:
        with pytest.raises(ValidationError):
            TypedResultSet.model_validate({"rows": []})

    def test_json_round_trip_uses_external_alias(self) -> None:
        exp = TypedResultSet(rows=[{"x": 1}], schema=[Column(name="x", type="INTEGER")])
        dumped = exp.model_dump_json()
        assert '"schema"' in dumped
        assert '"schema_"' not in dumped
        restored = TypedResultSet.model_validate_json(dumped)
        assert restored == exp

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            TypedResultSet.model_validate(
                {"rows": [], "schema": [{"name": "x", "type": "INT"}], "dialect": "duckdb"},
            )


@pytest.mark.unit
class TestGoldQuery:
    def test_construction(self) -> None:
        exp = GoldQuery(sql="SELECT COUNT(*) FROM tracks WHERE genre = 'Rock'")
        assert exp.kind == "gold_query"

    def test_json_round_trip(self) -> None:
        exp = GoldQuery(sql="SELECT 1")
        restored = GoldQuery.model_validate_json(exp.model_dump_json())
        assert restored == exp

    def test_rejects_empty_sql(self) -> None:
        with pytest.raises(ValidationError):
            GoldQuery(sql="")


@pytest.mark.unit
class TestExpectations:
    def test_row_count_construction(self) -> None:
        e = RowCountExpectation(exact=42)
        assert e.kind == "row_count"
        assert e.exact == 42

    def test_row_count_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            RowCountExpectation(exact=-1)

    def test_column_presence_construction(self) -> None:
        e = ColumnPresenceExpectation(columns=["id", "revenue"])
        assert e.columns == ["id", "revenue"]

    def test_column_presence_rejects_empty_list(self) -> None:
        with pytest.raises(ValidationError):
            ColumnPresenceExpectation(columns=[])

    def test_column_type_construction(self) -> None:
        e = ColumnTypeExpectation(column="id", expected_type="INTEGER")
        assert e.column == "id"
        assert e.expected_type == SqlType(raw="INTEGER")
        assert e.expected_type.raw == "INTEGER"

    def test_not_null_construction(self) -> None:
        e = NotNullExpectation(column="email")
        assert e.column == "email"

    def test_unique_construction(self) -> None:
        e = UniqueExpectation(column="user_id")
        assert e.column == "user_id"

    def test_expectation_discriminator_dispatches(self) -> None:
        data = {"kind": "row_count", "exact": 7}
        e = ExpectationAdapter.validate_python(data)
        assert isinstance(e, RowCountExpectation)

    def test_expectation_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValidationError):
            ExpectationAdapter.validate_python({"kind": "fictional", "x": 1})


@pytest.mark.unit
class TestExpectationSuite:
    def test_construction(self) -> None:
        suite = ExpectationSuite(
            expectations=[
                RowCountExpectation(exact=10),
                UniqueExpectation(column="id"),
            ]
        )
        assert suite.kind == "expectation_suite"
        assert len(suite.expectations) == 2

    def test_rejects_empty_expectations(self) -> None:
        with pytest.raises(ValidationError):
            ExpectationSuite(expectations=[])

    def test_json_round_trip(self) -> None:
        suite = ExpectationSuite(
            expectations=[
                RowCountExpectation(exact=10),
                ColumnPresenceExpectation(columns=["id"]),
            ]
        )
        restored = ExpectationSuite.model_validate_json(suite.model_dump_json())
        assert restored == suite


@pytest.mark.unit
class TestExpected:
    def test_dispatches_to_untyped_result_set(self) -> None:
        e = ExpectedAdapter.validate_python({"kind": "untyped_result_set", "rows": [{"count": 1}]})
        assert isinstance(e, UntypedResultSet)

    def test_dispatches_to_typed_result_set(self) -> None:
        e = ExpectedAdapter.validate_python(
            {"kind": "typed_result_set", "rows": [{"n": 1}], "schema": [{"name": "n", "type": "INTEGER"}]}
        )
        assert isinstance(e, TypedResultSet)

    def test_infers_untyped_from_bare_rows(self) -> None:
        e = ExpectedAdapter.validate_python({"rows": [{"count": 1}]})
        assert isinstance(e, UntypedResultSet)

    def test_infers_typed_from_schema_key(self) -> None:
        e = ExpectedAdapter.validate_python({"rows": [{"n": 1}], "schema": [{"name": "n", "type": "INTEGER"}]})
        assert isinstance(e, TypedResultSet)

    def test_infers_typed_from_schema_underscore_key(self) -> None:
        e = ExpectedAdapter.validate_python({"rows": [{"n": 1}], "schema_": [{"name": "n", "type": "INTEGER"}]})
        assert isinstance(e, TypedResultSet)

    def test_does_not_infer_when_kind_present(self) -> None:
        e = ExpectedAdapter.validate_python({"kind": "gold_query", "sql": "SELECT 1"})
        assert isinstance(e, GoldQuery)

    def test_leaves_non_dict_untouched(self) -> None:
        e = ExpectedAdapter.validate_python(UntypedResultSet(rows=[{"n": 1}]))
        assert isinstance(e, UntypedResultSet)

    def test_dispatches_to_gold_query(self) -> None:
        e = ExpectedAdapter.validate_python({"kind": "gold_query", "sql": "SELECT 1"})
        assert isinstance(e, GoldQuery)

    def test_dispatches_to_expectation_suite(self) -> None:
        e = ExpectedAdapter.validate_python(
            {
                "kind": "expectation_suite",
                "expectations": [{"kind": "row_count", "exact": 5}],
            }
        )
        assert isinstance(e, ExpectationSuite)

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValidationError):
            ExpectedAdapter.validate_python({"kind": "something_else", "rows": []})

    def test_json_round_trip_through_adapter(self) -> None:
        original = ExpectationSuite(expectations=[RowCountExpectation(exact=3)])
        restored = ExpectedAdapter.validate_json(ExpectedAdapter.dump_json(original))
        assert restored == original


@pytest.mark.unit
class TestComparisonConfig:
    def test_defaults(self) -> None:
        cfg = ComparisonConfig()
        assert cfg.column_order == "ignore"
        assert cfg.null_equality == "equal"
        assert cfg.float_tolerance == 1e-9

    def test_strict_construction(self) -> None:
        cfg = ComparisonConfig(
            column_order="strict",
            null_equality="distinct",
            float_tolerance=0.0,
        )
        assert cfg.column_order == "strict"
        assert cfg.null_equality == "distinct"

    def test_json_round_trip(self) -> None:
        cfg = ComparisonConfig(column_order="strict", float_tolerance=1e-6)
        restored = ComparisonConfig.model_validate_json(cfg.model_dump_json())
        assert restored == cfg

    def test_rejects_unknown_column_order(self) -> None:
        with pytest.raises(ValidationError):
            ComparisonConfig.model_validate({"column_order": "maybe"})

    def test_rejects_unknown_null_equality(self) -> None:
        with pytest.raises(ValidationError):
            ComparisonConfig.model_validate({"null_equality": "python"})

    def test_rejects_negative_float_tolerance(self) -> None:
        with pytest.raises(ValidationError):
            ComparisonConfig(float_tolerance=-1e-9)

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ComparisonConfig.model_validate({"duplicates": "multiset"})


@pytest.mark.unit
class TestCostBudget:
    def test_default_no_limit(self) -> None:
        budget = CostBudget()
        assert budget.max_seconds is None

    def test_with_seconds(self) -> None:
        budget = CostBudget(max_seconds=30.0)
        assert budget.max_seconds == 30.0

    def test_subsecond_budget(self) -> None:
        budget = CostBudget(max_seconds=0.5)
        assert budget.max_seconds == 0.5

    def test_json_round_trip(self) -> None:
        budget = CostBudget(max_seconds=12.5)
        restored = CostBudget.model_validate_json(budget.model_dump_json())
        assert restored == budget

    def test_rejects_zero_seconds(self) -> None:
        with pytest.raises(ValidationError):
            CostBudget(max_seconds=0)

    def test_rejects_negative_seconds(self) -> None:
        with pytest.raises(ValidationError):
            CostBudget(max_seconds=-1.0)

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            CostBudget.model_validate({"max_seconds": 30.0, "max_usd": 50.0})


def _make_case(**overrides: Any) -> EvalCase:
    defaults: dict[str, Any] = {
        "id": "rock_track_count",
        "input": "How many tracks are in the 'Rock' genre?",
        "expected": UntypedResultSet(rows=[{"count": 1297}]),
        "platform": PlatformRef(name="local", kind="duckdb"),
    }
    return EvalCase(**(defaults | overrides))


@pytest.mark.unit
class TestEvalCase:
    def test_minimal_construction(self) -> None:
        case = _make_case()
        assert case.id == "rock_track_count"
        assert case.comparison == ComparisonConfig()
        assert case.cost_budget is None
        assert case.metadata == {}

    def test_full_construction(self) -> None:
        case = EvalCase(
            id="case-1",
            input="List active users",
            expected=GoldQuery(sql="SELECT * FROM users WHERE active"),
            platform=PlatformRef(name="warehouse", kind="postgres"),
            comparison=ComparisonConfig(column_order="strict"),
            cost_budget=CostBudget(max_seconds=30.0),
            metadata={"owner": "analytics", "ticket": "ANL-42"},
        )
        assert case.comparison.column_order == "strict"
        assert case.metadata["owner"] == "analytics"

    def test_accepts_expectation_suite(self) -> None:
        case = _make_case(expected=ExpectationSuite(expectations=[RowCountExpectation(exact=10)]))
        assert isinstance(case.expected, ExpectationSuite)

    def test_canonicalizes_column_type_expectation(self) -> None:
        case = _make_case(
            expected=ExpectationSuite(
                expectations=[
                    ColumnTypeExpectation(column="id", expected_type="INT"),
                    RowCountExpectation(exact=1),
                ]
            ),
        )
        suite = case.expected
        assert isinstance(suite, ExpectationSuite)
        column_type = suite.expectations[0]
        assert isinstance(column_type, ColumnTypeExpectation)
        assert column_type.expected_type.canonical is not None
        assert column_type.expected_type == SqlType.parse("INTEGER", "duckdb")
        assert suite.expectations[1] == RowCountExpectation(exact=1)

    def test_json_round_trip(self) -> None:
        case = _make_case()
        restored = EvalCase.model_validate_json(case.model_dump_json())
        assert restored == case

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValidationError):
            _make_case(id="")

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValidationError):
            _make_case(input="")

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            _make_case(typo_field="anything")

    def test_rejects_duplicate_expected_schema_columns(self) -> None:
        with pytest.raises(ValidationError, match="duplicate column name"):
            _make_case(
                expected=TypedResultSet(
                    rows=[{"x": 1}],
                    schema=[Column(name="x", type="INTEGER"), Column(name="x", type="BIGINT")],
                ),
            )

    def test_default_metadata_not_shared(self) -> None:
        case_a = _make_case()
        case_b = _make_case(id="another")
        case_a.metadata["touched"] = True
        assert case_b.metadata == {}


@pytest.mark.unit
class TestSolverOutput:
    def test_minimal_construction(self) -> None:
        out = SolverOutput(output="SELECT 1")
        assert out.output == "SELECT 1"
        assert out.prompt_tokens is None
        assert out.completion_tokens is None
        assert out.latency_seconds is None
        assert out.cost_usd is None
        assert out.metadata == {}

    def test_full_construction(self) -> None:
        out = SolverOutput(
            output="SELECT COUNT(*) FROM tracks WHERE genre = 'Rock'",
            prompt_tokens=120,
            completion_tokens=18,
            latency_seconds=0.92,
            cost_usd=0.00034,
            metadata={"model": "gpt-4o", "prompt_version": "v3"},
        )
        assert out.prompt_tokens == 120
        assert out.completion_tokens == 18
        assert out.cost_usd == 0.00034
        assert out.metadata["model"] == "gpt-4o"

    def test_callable_solver_can_omit_usage(self) -> None:
        out = SolverOutput(output="SELECT 1", metadata={"source": "custom_pipeline"})
        assert out.prompt_tokens is None
        assert out.metadata["source"] == "custom_pipeline"

    def test_json_round_trip(self) -> None:
        out = SolverOutput(output="SELECT 1", prompt_tokens=10, latency_seconds=0.5)
        restored = SolverOutput.model_validate_json(out.model_dump_json())
        assert restored == out

    def test_rejects_empty_output(self) -> None:
        with pytest.raises(ValidationError):
            SolverOutput(output="")

    def test_rejects_negative_tokens(self) -> None:
        with pytest.raises(ValidationError):
            SolverOutput.model_validate({"output": "SELECT 1", "prompt_tokens": -1})

    def test_rejects_negative_latency(self) -> None:
        with pytest.raises(ValidationError):
            SolverOutput.model_validate({"output": "SELECT 1", "latency_seconds": -0.1})

    def test_rejects_negative_cost(self) -> None:
        with pytest.raises(ValidationError):
            SolverOutput.model_validate({"output": "SELECT 1", "cost_usd": -0.01})

    def test_accepts_zero_completion_tokens(self) -> None:
        out = SolverOutput(output="SELECT 1", completion_tokens=0)
        assert out.completion_tokens == 0

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            SolverOutput.model_validate({"output": "SELECT 1", "sql": "SELECT 1"})

    def test_default_metadata_not_shared(self) -> None:
        a = SolverOutput(output="SELECT 1")
        b = SolverOutput(output="SELECT 2")
        a.metadata["touched"] = True
        assert b.metadata == {}

    def test_error_construction(self) -> None:
        out = SolverOutput(error=SolverError(kind="timeout", message="timed out"))
        assert out.output is None
        assert out.error is not None
        assert out.error.kind == "timeout"

    def test_rejects_neither_output_nor_error(self) -> None:
        with pytest.raises(ValidationError):
            SolverOutput()

    def test_rejects_both_output_and_error(self) -> None:
        with pytest.raises(ValidationError):
            SolverOutput(output="SELECT 1", error=SolverError(kind="auth", message="bad key"))

    def test_error_json_round_trip(self) -> None:
        out = SolverOutput(error=SolverError(kind="rate_limit", message="429", provider="openai"))
        restored = SolverOutput.model_validate_json(out.model_dump_json())
        assert restored == out


@pytest.mark.unit
class TestSolverError:
    def test_construction(self) -> None:
        err = SolverError(kind="auth", message="invalid api key", provider="openai")
        assert err.kind == "auth"
        assert err.message == "invalid api key"
        assert err.provider == "openai"

    def test_provider_optional(self) -> None:
        err = SolverError(kind="empty_response", message="model returned no SQL")
        assert err.provider is None

    def test_rejects_empty_message(self) -> None:
        with pytest.raises(ValidationError):
            SolverError(kind="timeout", message="")

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValidationError):
            SolverError.model_validate({"kind": "explosion", "message": "boom"})

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            SolverError.model_validate({"kind": "auth", "message": "x", "status_code": 401})


@pytest.mark.unit
class TestExecutionResult:
    def test_minimal_construction(self) -> None:
        result = ExecutionResult(rows=[{"count": 1297}], latency_seconds=0.042)
        assert result.rows == [{"count": 1297}]
        assert result.schema_ is None
        assert result.latency_seconds == 0.042
        assert result.error is None
        assert result.metadata == {}

    def test_empty_rows_allowed(self) -> None:
        result = ExecutionResult(rows=[], latency_seconds=0.01)
        assert result.rows == []

    def test_with_schema(self) -> None:
        result = ExecutionResult(
            rows=[{"id": 1, "revenue": 12.5}],
            schema=[Column(name="id", type="INTEGER"), Column(name="revenue", type="DOUBLE")],
            latency_seconds=0.1,
        )
        assert result.schema_ is not None
        assert result.schema_.names == ["id", "revenue"]
        assert [t.raw for t in result.schema_.types] == ["INTEGER", "DOUBLE"]

    def test_with_error(self) -> None:
        result = ExecutionResult(
            rows=[],
            latency_seconds=0.005,
            error="syntax error at or near 'FROMM'",
        )
        assert result.error == "syntax error at or near 'FROMM'"

    def test_with_metadata(self) -> None:
        result = ExecutionResult(
            rows=[{"x": 1}],
            latency_seconds=0.2,
            metadata={"warehouse": "EVAL_WH", "query_id": "abc-123"},
        )
        assert result.metadata["warehouse"] == "EVAL_WH"

    def test_json_round_trip_minimal(self) -> None:
        result = ExecutionResult(rows=[{"x": 1}], latency_seconds=0.1)
        restored = ExecutionResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_json_round_trip_schema_uses_external_alias(self) -> None:
        result = ExecutionResult(
            rows=[{"x": 1}],
            schema=[Column(name="x", type="INTEGER")],
            latency_seconds=0.1,
        )
        dumped = result.model_dump_json()
        assert '"schema"' in dumped
        assert '"schema_"' not in dumped
        restored = ExecutionResult.model_validate_json(dumped)
        assert restored == result

    def test_rejects_negative_latency(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionResult.model_validate({"rows": [], "latency_seconds": -0.001})

    def test_rejects_empty_error_string(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionResult.model_validate({"rows": [], "latency_seconds": 0.0, "error": ""})

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionResult.model_validate(
                {"rows": [], "latency_seconds": 0.0, "bytes_scanned": 1000},
            )

    def test_default_metadata_not_shared(self) -> None:
        a = ExecutionResult(rows=[], latency_seconds=0.0)
        b = ExecutionResult(rows=[], latency_seconds=0.0)
        a.metadata["touched"] = True
        assert b.metadata == {}


@pytest.mark.unit
class TestTypeMismatch:
    def test_construction(self) -> None:
        m = TypeMismatch(column="revenue", expected="DOUBLE", actual="VARCHAR")
        assert m.column == "revenue"
        assert m.expected == "DOUBLE"
        assert m.actual == "VARCHAR"

    def test_json_round_trip(self) -> None:
        m = TypeMismatch(column="id", expected="INTEGER", actual="BIGINT")
        restored = TypeMismatch.model_validate_json(m.model_dump_json())
        assert restored == m

    def test_rejects_empty_column(self) -> None:
        with pytest.raises(ValidationError):
            TypeMismatch(column="", expected="INT", actual="VARCHAR")

    def test_rejects_empty_expected(self) -> None:
        with pytest.raises(ValidationError):
            TypeMismatch(column="x", expected="", actual="VARCHAR")

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            TypeMismatch.model_validate({"column": "x", "expected": "INT", "actual": "VARCHAR", "reason": "..."})


@pytest.mark.unit
class TestColumnMismatch:
    def test_construction(self) -> None:
        m = ColumnMismatch(column="revenue", unexpected_count=7)
        assert m.column == "revenue"
        assert m.unexpected_count == 7

    def test_zero_unexpected_allowed(self) -> None:
        m = ColumnMismatch(column="id", unexpected_count=0)
        assert m.unexpected_count == 0

    def test_json_round_trip(self) -> None:
        m = ColumnMismatch(column="name", unexpected_count=3)
        restored = ColumnMismatch.model_validate_json(m.model_dump_json())
        assert restored == m

    def test_rejects_empty_column(self) -> None:
        with pytest.raises(ValidationError):
            ColumnMismatch(column="", unexpected_count=1)

    def test_rejects_negative_count(self) -> None:
        with pytest.raises(ValidationError):
            ColumnMismatch(column="x", unexpected_count=-1)

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ColumnMismatch.model_validate(
                {"column": "x", "unexpected_count": 1, "mismatch_count": 1},
            )


@pytest.mark.unit
class TestExpectationOutcome:
    def test_minimal_pass(self) -> None:
        o = ExpectationOutcome(kind="unique", passed=True)
        assert o.kind == "unique"
        assert o.passed is True
        assert o.column is None
        assert o.expected is None
        assert o.actual is None
        assert o.count is None
        assert o.detail is None

    def test_full_construction(self) -> None:
        o = ExpectationOutcome(
            kind="column_type",
            passed=False,
            column="n",
            expected="INTEGER",
            actual="BIGINT",
            count=None,
            detail="column_type: column 'n' expected type 'INTEGER', got 'BIGINT'",
        )
        assert o.column == "n"
        assert o.expected == "INTEGER"
        assert o.actual == "BIGINT"
        assert o.detail is not None

    def test_json_round_trip(self) -> None:
        o = ExpectationOutcome(kind="not_null", passed=False, column="email", count=2, detail="2 NULLs")
        restored = ExpectationOutcome.model_validate_json(o.model_dump_json())
        assert restored == o

    def test_rejects_empty_kind(self) -> None:
        with pytest.raises(ValidationError):
            ExpectationOutcome(kind="", passed=True)

    def test_rejects_empty_detail(self) -> None:
        with pytest.raises(ValidationError):
            ExpectationOutcome(kind="row_count", passed=False, detail="")

    def test_rejects_negative_count(self) -> None:
        with pytest.raises(ValidationError):
            ExpectationOutcome(kind="not_null", passed=False, count=-1)

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ExpectationOutcome.model_validate({"kind": "row_count", "passed": True, "sample": []})


@pytest.mark.unit
class TestResultSetDiff:
    def test_minimal_construction(self) -> None:
        diff = ResultSetDiff(expected_row_count=10, actual_row_count=10)
        assert diff.expected_row_count == 10
        assert diff.actual_row_count == 10
        assert diff.missing_row_count == 0
        assert diff.extra_row_count == 0
        assert diff.missing_columns == []
        assert diff.unexpected_columns == []
        assert diff.type_mismatches == []
        assert diff.column_mismatches == []
        assert diff.column_order_mismatch is False

    def test_full_construction(self) -> None:
        diff = ResultSetDiff(
            expected_row_count=10,
            actual_row_count=8,
            missing_row_count=3,
            extra_row_count=1,
            missing_columns=["revenue"],
            unexpected_columns=["unused"],
            type_mismatches=[TypeMismatch(column="id", expected="INTEGER", actual="BIGINT")],
            column_mismatches=[ColumnMismatch(column="name", unexpected_count=2)],
        )
        assert diff.missing_row_count == 3
        assert diff.extra_row_count == 1
        assert diff.missing_columns == ["revenue"]
        assert diff.type_mismatches[0].column == "id"
        assert diff.column_mismatches[0].unexpected_count == 2

    def test_zero_rows_both_sides(self) -> None:
        diff = ResultSetDiff(expected_row_count=0, actual_row_count=0)
        assert diff.expected_row_count == 0
        assert diff.actual_row_count == 0

    def test_json_round_trip(self) -> None:
        diff = ResultSetDiff(
            expected_row_count=5,
            actual_row_count=5,
            missing_columns=["a"],
            type_mismatches=[TypeMismatch(column="x", expected="INT", actual="VARCHAR")],
            column_mismatches=[ColumnMismatch(column="y", unexpected_count=3)],
        )
        restored = ResultSetDiff.model_validate_json(diff.model_dump_json())
        assert restored == diff

    def test_rejects_negative_expected_row_count(self) -> None:
        with pytest.raises(ValidationError):
            ResultSetDiff.model_validate({"expected_row_count": -1, "actual_row_count": 0})

    def test_rejects_negative_missing_row_count(self) -> None:
        with pytest.raises(ValidationError):
            ResultSetDiff(expected_row_count=10, actual_row_count=10, missing_row_count=-1)

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ResultSetDiff.model_validate(
                {
                    "expected_row_count": 1,
                    "actual_row_count": 1,
                    "null_mismatches": 0,
                },
            )

    def test_default_collections_not_shared(self) -> None:
        a = ResultSetDiff(expected_row_count=0, actual_row_count=0)
        b = ResultSetDiff(expected_row_count=0, actual_row_count=0)
        a.missing_columns.append("touched")
        a.type_mismatches.append(TypeMismatch(column="x", expected="INT", actual="VARCHAR"))
        assert b.missing_columns == []
        assert b.type_mismatches == []


@pytest.mark.unit
class TestScoreResult:
    def test_minimal_passed(self) -> None:
        result = ScoreResult(scorer="result_set_equivalence", passed=True)
        assert result.scorer == "result_set_equivalence"
        assert result.passed is True
        assert result.diff is None
        assert result.outcomes == []
        assert result.explanation is None
        assert result.metadata == {}

    def test_minimal_failed(self) -> None:
        result = ScoreResult(scorer="result_set_equivalence", passed=False)
        assert result.passed is False

    def test_full_construction(self) -> None:
        diff = ResultSetDiff(
            expected_row_count=10,
            actual_row_count=8,
            missing_row_count=2,
            column_mismatches=[ColumnMismatch(column="revenue", unexpected_count=3)],
        )
        result = ScoreResult(
            scorer="result_set_equivalence",
            passed=False,
            diff=diff,
            explanation="2 missing rows; 3 value mismatches in revenue",
            metadata={"engine_version": "0.1.0"},
        )
        assert result.diff == diff
        assert result.explanation is not None
        assert result.metadata["engine_version"] == "0.1.0"

    def test_passed_can_carry_diff(self) -> None:
        diff = ResultSetDiff(expected_row_count=5, actual_row_count=5)
        result = ScoreResult(scorer="result_set_equivalence", passed=True, diff=diff)
        assert result.passed is True
        assert result.diff is not None
        assert result.diff.expected_row_count == 5

    def test_json_round_trip_minimal(self) -> None:
        result = ScoreResult(scorer="result_set_equivalence", passed=True)
        restored = ScoreResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_json_round_trip_with_diff(self) -> None:
        result = ScoreResult(
            scorer="result_set_equivalence",
            passed=False,
            diff=ResultSetDiff(
                expected_row_count=3,
                actual_row_count=2,
                missing_row_count=1,
            ),
            explanation="1 row missing",
        )
        restored = ScoreResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_json_round_trip_with_outcomes(self) -> None:
        result = ScoreResult(
            scorer="expectation_suite",
            passed=False,
            outcomes=[
                ExpectationOutcome(kind="row_count", passed=True, expected="2", actual="2"),
                ExpectationOutcome(
                    kind="not_null",
                    passed=False,
                    column="email",
                    count=2,
                    detail="not_null: column 'email' has 2 NULL value(s)",
                ),
            ],
            explanation="1 expectation(s) failed:\n  - not_null: column 'email' has 2 NULL value(s)",
        )
        restored = ScoreResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_default_outcomes_not_shared(self) -> None:
        a = ScoreResult(scorer="x", passed=True)
        b = ScoreResult(scorer="y", passed=True)
        a.outcomes.append(ExpectationOutcome(kind="row_count", passed=True))
        assert b.outcomes == []

    def test_rejects_empty_scorer(self) -> None:
        with pytest.raises(ValidationError):
            ScoreResult(scorer="", passed=True)

    def test_rejects_empty_explanation(self) -> None:
        with pytest.raises(ValidationError):
            ScoreResult.model_validate(
                {"scorer": "x", "passed": True, "explanation": ""},
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ScoreResult.model_validate(
                {"scorer": "x", "passed": True, "score": 0.5},
            )

    def test_default_metadata_not_shared(self) -> None:
        a = ScoreResult(scorer="x", passed=True)
        b = ScoreResult(scorer="x", passed=True)
        a.metadata["touched"] = True
        assert b.metadata == {}
