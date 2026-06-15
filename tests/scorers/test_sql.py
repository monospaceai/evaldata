"""Tests for `dataeval.scorers.sql` — dialect-correct, identifier-safe check SQL."""

import datetime
from decimal import Decimal

import pytest

from dataeval.scorers import sql
from dataeval.types import Column, Schema, Sql, SqlType

_MODEL = Sql("SELECT email FROM users")


def _schema(*pairs: tuple[str, str], dialect: str = "duckdb") -> Schema:
    return Schema(root=[Column(name=n, type=SqlType.parse(t, dialect)) for n, t in pairs])


@pytest.mark.unit
class TestWrapModel:
    def test_duckdb_aliases_derived_table(self) -> None:
        assert sql.wrap_model(_MODEL, "count(*)", "duckdb") == "SELECT COUNT(*) FROM (SELECT email FROM users) AS t"

    def test_postgres_aliases_derived_table(self) -> None:
        assert sql.wrap_model(_MODEL, "count(*)", "postgres") == "SELECT COUNT(*) FROM (SELECT email FROM users) AS t"

    def test_quotes_reserved_column_name(self) -> None:
        # A column named after a keyword must be quoted, not emitted bare.
        out = sql.wrap_model(Sql('SELECT "order" FROM t'), '"order"', "postgres")
        assert '"order"' in out


@pytest.mark.unit
class TestCheckBuilders:
    def test_row_count(self) -> None:
        assert sql.row_count(_MODEL, "duckdb") == "SELECT COUNT(*) FROM (SELECT email FROM users) AS t"

    def test_not_null_count(self) -> None:
        out = sql.not_null_count(_MODEL, "email", "postgres")
        assert out == 'SELECT COUNT(*) FROM (SELECT email FROM users) AS t WHERE "email" IS NULL'

    def test_not_null_sample_is_limited(self) -> None:
        out = sql.not_null_sample(_MODEL, "email", "duckdb")
        assert out == 'SELECT * FROM (SELECT email FROM users) AS t WHERE "email" IS NULL LIMIT 20'

    def test_unique_count_excludes_nulls(self) -> None:
        out = sql.unique_count(_MODEL, "email", "duckdb")
        assert 'WHERE NOT "email" IS NULL' in out
        assert "GROUP BY" in out
        assert "HAVING COUNT(*) > 1" in out

    def test_unique_sample_carries_counts_and_limit(self) -> None:
        out = sql.unique_sample(_MODEL, "email", "postgres")
        assert "COUNT(*) AS n" in out
        assert "LIMIT 20" in out

    def test_quoted_identifier_column(self) -> None:
        # A column named `order` is quoted in every clause it appears in.
        out = sql.unique_count(Sql('SELECT "order" FROM t'), "order", "duckdb")
        assert '"order"' in out
        assert " order " not in out


@pytest.mark.unit
class TestLiteral:
    def test_none_is_null(self) -> None:
        assert sql._literal(None).sql() == "NULL"

    def test_bool_renders_as_boolean(self) -> None:
        assert sql._literal(value=True).sql(dialect="duckdb") == "TRUE"

    def test_number(self) -> None:
        assert sql._literal(Decimal("2.50")).sql() == "2.50"

    def test_string_is_quoted(self) -> None:
        assert sql._literal("a").sql() == "'a'"

    def test_other_value_falls_through_to_convert(self) -> None:
        # A date is neither None/bool/number/str: it goes through the generic converter.
        out = sql._literal(datetime.date(2020, 1, 2)).sql(dialect="duckdb")
        assert "2020-01-02" in out


@pytest.mark.unit
class TestIsNumericType:
    def test_numeric_type(self) -> None:
        assert sql.is_numeric_type("NUMERIC", "duckdb") is True

    def test_non_numeric_type(self) -> None:
        assert sql.is_numeric_type("VARCHAR", "duckdb") is False

    def test_unparseable_type_is_not_numeric(self) -> None:
        assert sql.is_numeric_type("not a real type !!!", "duckdb") is False


@pytest.mark.unit
class TestRoundScale:
    def test_default_tolerance(self) -> None:
        assert sql.round_scale(1e-9) == 9

    def test_coarser_tolerance(self) -> None:
        assert sql.round_scale(1e-6) == 6

    def test_tolerance_above_one_clamps_to_zero(self) -> None:
        assert sql.round_scale(10.0) == 0


@pytest.mark.unit
class TestExpectedRelation:
    def test_typed_cast_literals(self) -> None:
        schema = _schema(("n", "INTEGER"), ("s", "VARCHAR"))
        out = sql.expected_relation([{"n": 1, "s": "a"}], schema, ["n", "s"], "duckdb", None).sql(dialect="duckdb")
        assert out == 'SELECT CAST(1 AS INT) AS "n", CAST(\'a\' AS TEXT) AS "s"'

    def test_null_cell_casts_null(self) -> None:
        schema = _schema(("n", "INTEGER"))
        out = sql.expected_relation([{"n": None}], schema, ["n"], "duckdb", None).sql(dialect="duckdb")
        assert out == 'SELECT CAST(NULL AS INT) AS "n"'

    def test_numeric_literal_not_string(self) -> None:
        # A NUMERIC column casts the Decimal as a number literal, not a quoted string.
        schema = _schema(("price", "NUMERIC"))
        out = sql.expected_relation([{"price": Decimal("2.50")}], schema, ["price"], "postgres", None).sql(
            dialect="postgres"
        )
        assert "CAST(2.50 AS" in out
        assert "'2.50'" not in out

    def test_multiple_rows_union_all(self) -> None:
        schema = _schema(("n", "INTEGER"))
        out = sql.expected_relation([{"n": 1}, {"n": 2}], schema, ["n"], "duckdb", None).sql(dialect="duckdb")
        assert out == 'SELECT CAST(1 AS INT) AS "n" UNION ALL SELECT CAST(2 AS INT) AS "n"'

    def test_empty_rows_yield_typed_empty_relation(self) -> None:
        schema = _schema(("n", "INTEGER"))
        out = sql.expected_relation([], schema, ["n"], "duckdb", None).sql(dialect="duckdb")
        assert out == 'SELECT CAST(NULL AS INT) AS "n" WHERE 1 = 0'

    def test_untyped_emits_bare_literals(self) -> None:
        out = sql.expected_relation([{"n": 1}], None, ["n"], "duckdb", None).sql(dialect="duckdb")
        assert out == 'SELECT 1 AS "n"'

    def test_quotes_reserved_column(self) -> None:
        schema = _schema(("order", "INTEGER"))
        out = sql.expected_relation([{"order": 1}], schema, ["order"], "postgres", None).sql(dialect="postgres")
        assert 'AS "order"' in out

    def test_numeric_column_rounded_when_scale_given(self) -> None:
        schema = _schema(("v", "DOUBLE"))
        out = sql.expected_relation([{"v": 1.0}], schema, ["v"], "duckdb", 9).sql(dialect="duckdb")
        assert "ROUND(CAST(CAST(1.0 AS DOUBLE) AS DECIMAL(38, 18)), 9)" in out

    def test_non_numeric_column_not_rounded(self) -> None:
        schema = _schema(("s", "VARCHAR"))
        out = sql.expected_relation([{"s": "a"}], schema, ["s"], "duckdb", 9).sql(dialect="duckdb")
        assert "ROUND" not in out

    def test_unparseable_type_falls_back_to_bare_literal(self) -> None:
        # A type SQLGlot cannot parse keeps `raw` with `canonical=None`; the cell degrades
        # to a bare literal rather than raising.
        schema = Schema(root=[Column(name="c", type=SqlType.parse("MY_CUSTOM_TYPE", "duckdb"))])
        out = sql.expected_relation([{"c": 1}], schema, ["c"], "duckdb", None).sql(dialect="duckdb")
        assert out == 'SELECT 1 AS "c"'
        assert "CAST" not in out


@pytest.mark.unit
class TestAlignedActual:
    def test_projects_columns_from_model(self) -> None:
        out = sql.aligned_actual(_MODEL, ["email"], set(), "duckdb", None).sql(dialect="duckdb")
        assert out == 'SELECT "email" AS "email" FROM (SELECT email FROM users) AS t'

    def test_rounds_named_numeric_columns(self) -> None:
        out = sql.aligned_actual(Sql("SELECT v FROM t"), ["v"], {"v"}, "duckdb", 9).sql(dialect="duckdb")
        assert 'ROUND(CAST("v" AS DECIMAL(38, 18)), 9) AS "v"' in out

    def test_quotes_reserved_column(self) -> None:
        out = sql.aligned_actual(Sql('SELECT "order" FROM t'), ["order"], set(), "postgres", None).sql(
            dialect="postgres"
        )
        assert '"order"' in out


@pytest.mark.unit
class TestGoldSchemaProbe:
    def test_wraps_gold_in_zero_row_subquery(self) -> None:
        out = sql.gold_schema_probe(Sql("SELECT id, v FROM t"), "duckdb")
        assert out == "SELECT * FROM (SELECT id, v FROM t) AS __gold__ LIMIT 0"

    def test_postgres_dialect(self) -> None:
        out = sql.gold_schema_probe(Sql("SELECT 1 AS n"), "postgres")
        assert out == "SELECT * FROM (SELECT 1 AS n) AS __gold__ LIMIT 0"


@pytest.mark.unit
class TestGoldExpected:
    def test_projects_columns_from_gold_subquery(self) -> None:
        out = sql.gold_expected(Sql("SELECT email FROM users"), ["email"], set(), "duckdb", None).sql(dialect="duckdb")
        assert out == 'SELECT "email" AS "email" FROM (SELECT email FROM users) AS __gold__'

    def test_rounds_named_numeric_columns(self) -> None:
        out = sql.gold_expected(Sql("SELECT v FROM t"), ["v"], {"v"}, "duckdb", 9).sql(dialect="duckdb")
        assert 'ROUND(CAST("v" AS DECIMAL(38, 18)), 9) AS "v"' in out
        assert "AS __gold__" in out

    def test_quotes_reserved_column(self) -> None:
        out = sql.gold_expected(Sql('SELECT "order" FROM t'), ["order"], set(), "postgres", None).sql(
            dialect="postgres"
        )
        assert '"order"' in out


@pytest.mark.unit
class TestExceptAllWithGold:
    def test_keyless_count_with_gold_subquery(self) -> None:
        left = sql.gold_expected(Sql("SELECT 1 AS n"), ["n"], set(), "duckdb", None)
        right = sql.aligned_actual(Sql("SELECT 2 AS n"), ["n"], set(), "duckdb", None)
        out = sql.except_all_count(left, right, "duckdb")
        assert out == (
            'SELECT COUNT(*) FROM (SELECT * FROM (SELECT "n" AS "n" FROM (SELECT 1 AS n) AS __gold__) AS l '
            'EXCEPT ALL SELECT * FROM (SELECT "n" AS "n" FROM (SELECT 2 AS n) AS t) AS r) AS d'
        )

    def test_keyed_stats_with_gold_subquery(self) -> None:
        left = sql.gold_expected(Sql("SELECT 1 AS id, 10 AS v"), ["id", "v"], set(), "duckdb", None)
        right = sql.aligned_actual(Sql("SELECT 1 AS id, 10 AS v"), ["id", "v"], set(), "duckdb", None)
        out = sql.keyed_diff_stats(left, right, ["id"], ["v"], set(), "equal", 1e-9, ["id", "v"], "duckdb")
        assert "FULL OUTER JOIN" in out
        assert "__gold__" in out


def _keyed_rels(
    expected_rows: list[tuple[int, int]] | None = None,
    actual_rows: list[tuple[int, int]] | None = None,
) -> tuple[object, object]:
    expected = " UNION ALL ".join(f"SELECT {i} AS id, {v} AS v" for i, v in expected_rows or [(1, 10)])
    actual = " UNION ALL ".join(f"SELECT {i} AS id, {v} AS v" for i, v in actual_rows or [(1, 10)])
    left = sql.aligned_actual(Sql(expected), ["id", "v"], set(), "duckdb", None)
    right = sql.aligned_actual(Sql(actual), ["id", "v"], set(), "duckdb", None)
    return left, right


@pytest.mark.unit
class TestKeyedDiffStats:
    def test_renames_and_presence_markers(self) -> None:
        left, right = _keyed_rels()
        out = sql.keyed_diff_stats(left, right, ["id"], ["v"], set(), "equal", 1e-9, ["id", "v"], "duckdb")
        assert '"e__id"' in out and '"e__v"' in out
        assert '"a__id"' in out and '"a__v"' in out
        assert "e_present" in out and "a_present" in out

    def test_aliases_missing_extra_and_per_column_counts(self) -> None:
        left, right = _keyed_rels()
        out = sql.keyed_diff_stats(left, right, ["id"], ["v"], set(), "equal", 1e-9, ["id", "v"], "duckdb")
        assert "AS missing" in out
        assert "AS extra" in out
        assert "AS m0" in out

    def test_full_outer_join_on_key(self) -> None:
        left, right = _keyed_rels()
        out = sql.keyed_diff_stats(left, right, ["id"], ["v"], set(), "equal", 1e-9, ["id", "v"], "duckdb")
        assert "FULL OUTER JOIN" in out
        assert '"e"."e__id" = "a"."a__id"' in out

    def test_numeric_column_uses_decimal_tolerance_band(self) -> None:
        left, right = _keyed_rels()
        out = sql.keyed_diff_stats(left, right, ["id"], ["v"], {"v"}, "equal", 0.5, ["id", "v"], "duckdb")
        assert "ABS(CAST(" in out
        assert "DECIMAL(38, 18)" in out
        assert "<= CAST(0.5 AS DECIMAL(38, 18))" in out

    def test_distinct_null_equality_uses_plain_equality(self) -> None:
        left, right = _keyed_rels()
        equal = sql.keyed_diff_stats(left, right, ["id"], ["v"], set(), "equal", 1e-9, ["id", "v"], "duckdb")
        distinct = sql.keyed_diff_stats(left, right, ["id"], ["v"], set(), "distinct", 1e-9, ["id", "v"], "duckdb")
        assert "IS NOT DISTINCT FROM" in equal
        assert "IS NOT DISTINCT FROM" not in distinct


@pytest.mark.unit
class TestKeyedSample:
    def test_missing_projects_expected_side_and_filters_absent_actual(self) -> None:
        left, right = _keyed_rels()
        out = sql.keyed_sample(left, right, ["id"], ["id", "v"], "missing", "duckdb")
        assert '"e__id" AS "id"' in out
        assert '"e__v" AS "v"' in out
        assert "j.a_present IS NULL" in out
        assert "LIMIT 20" in out

    def test_extra_projects_actual_side_and_filters_absent_expected(self) -> None:
        left, right = _keyed_rels()
        out = sql.keyed_sample(left, right, ["id"], ["id", "v"], "extra", "duckdb")
        assert '"a__id" AS "id"' in out
        assert '"a__v" AS "v"' in out
        assert "j.e_present IS NULL" in out


@pytest.mark.unit
class TestKeyedDupesCount:
    def test_groups_by_key_excluding_nulls(self) -> None:
        relation = sql.aligned_actual(Sql("SELECT 1 AS id, 10 AS v"), ["id", "v"], set(), "duckdb", None)
        out = sql.keyed_dupes_count(relation, ["id"], "duckdb")
        assert out.startswith("SELECT COUNT(*) FROM (")
        assert 'WHERE NOT "id" IS NULL' in out
        assert 'GROUP BY "id"' in out
        assert "HAVING COUNT(*) > 1" in out

    def test_composite_key_groups_and_filters_each_column(self) -> None:
        relation = sql.aligned_actual(Sql("SELECT 1 AS id, 2 AS k, 10 AS v"), ["id", "k", "v"], set(), "duckdb", None)
        out = sql.keyed_dupes_count(relation, ["id", "k"], "duckdb")
        assert 'NOT "id" IS NULL AND NOT "k" IS NULL' in out
        assert 'GROUP BY "id", "k"' in out


@pytest.mark.unit
class TestExceptAll:
    def test_count_shape(self) -> None:
        left = sql.expected_relation([{"n": 1}], _schema(("n", "INTEGER")), ["n"], "duckdb", None)
        right = sql.aligned_actual(Sql("SELECT 2 AS n"), ["n"], set(), "duckdb", None)
        out = sql.except_all_count(left, right, "duckdb")
        assert out == (
            'SELECT COUNT(*) FROM (SELECT * FROM (SELECT CAST(1 AS INT) AS "n") AS l '
            'EXCEPT ALL SELECT * FROM (SELECT "n" AS "n" FROM (SELECT 2 AS n) AS t) AS r) AS d'
        )

    def test_count_wraps_union_operand_for_precedence(self) -> None:
        # A multi-row (UNION ALL) operand must be parenthesised under EXCEPT ALL.
        left = sql.expected_relation([{"n": 1}, {"n": 2}], _schema(("n", "INTEGER")), ["n"], "duckdb", None)
        right = sql.aligned_actual(Sql("SELECT 1 AS n"), ["n"], set(), "duckdb", None)
        out = sql.except_all_count(left, right, "duckdb")
        assert "AS l EXCEPT ALL" in out
        assert out.index("UNION ALL") < out.index("EXCEPT ALL")

    def test_sample_shape_and_limit(self) -> None:
        left = sql.aligned_actual(Sql("SELECT 2 AS n"), ["n"], set(), "postgres", None)
        right = sql.expected_relation(
            [{"n": 1}], _schema(("n", "INTEGER"), dialect="postgres"), ["n"], "postgres", None
        )
        out = sql.except_all_sample(left, right, "postgres")
        assert out.startswith("SELECT * FROM (")
        assert "EXCEPT ALL" in out
        assert out.endswith(") AS d LIMIT 20")
