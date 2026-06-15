"""DuckDB-specific tests: native-type-string fidelity and file-backed lifecycle."""

from pathlib import Path

import pytest
from sqlglot import exp

from dataeval.platforms.duckdb import DuckDBAdapter


@pytest.mark.unit
class TestDuckDBNativeTypes:
    """DuckDB emits the native SQL type strings SQLGlot's `duckdb` dialect parses."""

    @pytest.fixture
    def adapter(self) -> DuckDBAdapter:
        return DuckDBAdapter()

    @pytest.mark.parametrize(
        ("sql", "expected_type"),
        [
            ("SELECT CAST(1 AS BIGINT) AS x", "BIGINT"),
            ("SELECT CAST(1 AS INTEGER) AS x", "INTEGER"),
            ("SELECT CAST('a' AS VARCHAR) AS x", "VARCHAR"),
            ("SELECT CAST(1.5 AS DOUBLE) AS x", "DOUBLE"),
            ("SELECT CAST(true AS BOOLEAN) AS x", "BOOLEAN"),
            ("SELECT [1, 2, 3] AS x", "INTEGER[]"),
            ("SELECT {'a': 1, 'b': 'x'} AS x", "STRUCT(a INTEGER, b VARCHAR)"),
        ],
    )
    def test_native_type_string_is_exact(self, adapter: DuckDBAdapter, sql: str, expected_type: str) -> None:
        result = adapter.execute(sql)
        assert result.error is None
        assert result.schema_ is not None
        assert result.schema_[0].type.raw == expected_type

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT CAST(1 AS BIGINT) AS x",
            "SELECT CAST('a' AS VARCHAR) AS x",
            "SELECT CAST(1.5 AS DOUBLE) AS x",
            "SELECT CAST(true AS BOOLEAN) AS x",
            "SELECT [1, 2, 3] AS x",
            "SELECT {'a': 1, 'b': 'x'} AS x",
        ],
    )
    def test_emitted_type_parses_under_duckdb_dialect(self, adapter: DuckDBAdapter, sql: str) -> None:
        """The type strings DuckDBAdapter emits must round-trip through SQLGlot."""
        result = adapter.execute(sql)
        assert result.error is None
        assert result.schema_ is not None
        parsed = exp.DataType.build(result.schema_[0].type.raw, dialect="duckdb")
        assert isinstance(parsed, exp.DataType)


@pytest.mark.unit
class TestDuckDBFilePath:
    """`database=` argument is honoured — a file-backed DB persists across reopens."""

    def test_file_backed_database_persists(self, tmp_path: Path) -> None:
        # Context-manager exit calls close() for deterministic release of the OS file
        # handle / WAL lock — implicit cleanup is unreliable on Windows.
        db_path = str(tmp_path / "test.duckdb")
        with DuckDBAdapter(database=db_path) as first:
            first.execute("CREATE TABLE t (x INTEGER)")
            first.execute("INSERT INTO t VALUES (42)")
        with DuckDBAdapter(database=db_path) as second:
            result = second.execute("SELECT x FROM t")
            assert result.error is None
            assert result.rows == [{"x": 42}]
