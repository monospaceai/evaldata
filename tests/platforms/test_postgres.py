"""Postgres-specific tests: native-type-string fidelity via psycopg `type_display`."""

import pytest
from sqlglot import exp

from data_eval.platforms.base import PlatformAdapter

from .conftest import connect_postgres_or_skip


@pytest.mark.e2e
class TestPostgresNativeTypes:
    """Postgres emits the native type strings (psycopg `type_display`) SQLGlot's `postgres` dialect parses."""

    @pytest.fixture
    def adapter(self) -> PlatformAdapter:
        return connect_postgres_or_skip()

    @pytest.mark.parametrize(
        ("sql", "expected_type"),
        [
            ("SELECT 1::bigint AS x", "int8"),
            ("SELECT 1::integer AS x", "int4"),
            ("SELECT 1::smallint AS x", "int2"),
            ("SELECT 'a'::text AS x", "text"),
            ("SELECT 'a'::varchar(10) AS x", "varchar(10)"),
            ("SELECT 1.5::numeric(10,2) AS x", "numeric(10,2)"),
            ("SELECT 1.5::double precision AS x", "float8"),
            ("SELECT true AS x", "bool"),
            ("SELECT ARRAY['a', 'b'] AS x", "text[]"),
            ("SELECT '{}'::jsonb AS x", "jsonb"),
            ("SELECT '2020-01-01'::date AS x", "date"),
        ],
    )
    def test_native_type_string_is_exact(self, adapter: PlatformAdapter, sql: str, expected_type: str) -> None:
        result = adapter.execute(sql)
        assert result.error is None
        assert result.schema_ is not None
        assert result.schema_[0].type.raw == expected_type

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1::bigint AS x",
            "SELECT 'a'::varchar(10) AS x",
            "SELECT 1.5::numeric(10,2) AS x",
            "SELECT true AS x",
            "SELECT ARRAY['a', 'b'] AS x",
            "SELECT '{}'::jsonb AS x",
        ],
    )
    def test_emitted_type_parses_under_postgres_dialect(self, adapter: PlatformAdapter, sql: str) -> None:
        """The type strings PostgresAdapter emits must round-trip through SQLGlot."""
        result = adapter.execute(sql)
        assert result.error is None
        assert result.schema_ is not None
        parsed = exp.DataType.build(result.schema_[0].type.raw, dialect="postgres")
        assert isinstance(parsed, exp.DataType)
