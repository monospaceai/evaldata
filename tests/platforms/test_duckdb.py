"""DuckDB-specific tests: native-type-string fidelity, file-backed lifecycle, and shared cursors."""

import threading
import time
from pathlib import Path

import duckdb
import pytest
from sqlglot import exp

from evaldata.platforms.duckdb import DuckDBAdapter


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


@pytest.mark.unit
class TestDuckDBSharedCursor:
    """`from_connection` adapters share one parent's in-process database and isolate interrupts."""

    def test_members_share_the_parent_database(self) -> None:
        parent = duckdb.connect(":memory:")
        try:
            seeder = DuckDBAdapter.from_connection(parent.cursor())
            reader = DuckDBAdapter.from_connection(parent.cursor())
            seeder.execute("CREATE TABLE t (n INTEGER); INSERT INTO t VALUES (1), (2), (3)")
            result = reader.execute("SELECT count(*) AS c FROM t")
            assert result.error is None
            assert result.rows == [{"c": 3}]
        finally:
            parent.close()

    def test_closing_a_member_does_not_close_the_parent(self) -> None:
        parent = duckdb.connect(":memory:")
        try:
            member = DuckDBAdapter.from_connection(parent.cursor())
            member.close()
            # The parent (and any sibling) still works after a member cursor is closed.
            assert parent.execute("SELECT 1 AS n").fetchall() == [(1,)]
            sibling = DuckDBAdapter.from_connection(parent.cursor())
            assert sibling.execute("SELECT 1 AS n").error is None
        finally:
            parent.close()

    def test_interrupt_isolation_between_sibling_cursors(self) -> None:
        # Canary: interrupting one cursor must not cancel a concurrent query on a sibling
        # cursor of the same parent. The pool's shared-cursor design (and DuckDB budgets)
        # relies on this; a future DuckDB bump that changes it should fail here.
        parent = duckdb.connect(":memory:")
        slow = (
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 1000000000) "
            "SELECT count(*) AS n FROM r"
        )
        a = parent.cursor()
        b = parent.cursor()
        results: dict[str, object] = {}

        def run(cursor: duckdb.DuckDBPyConnection, key: str, sql: str) -> None:
            try:
                results[key] = cursor.execute(sql).fetchall()
            except duckdb.InterruptException:
                results[key] = "interrupted"

        interrupted = threading.Thread(target=run, args=(a, "a", slow))
        sibling = threading.Thread(target=run, args=(b, "b", "SELECT count(*) AS n FROM range(3000000)"))
        try:
            interrupted.start()
            sibling.start()
            # Interrupt A repeatedly until its thread ends, so the interrupt reliably lands
            # regardless of scheduling (no fragile single sleep).
            while interrupted.is_alive():
                a.interrupt()
                time.sleep(0.01)
            interrupted.join(timeout=5)
            sibling.join(timeout=5)
        finally:
            parent.close()
        assert results["a"] == "interrupted"
        assert results["b"] == [(3000000,)]  # sibling completed; it was not cancelled
