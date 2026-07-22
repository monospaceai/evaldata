"""SQLite-specific tests: untyped result schemas and file-backed lifecycle."""

from pathlib import Path

import pytest

from evaldata.platforms.sqlite import SqliteAdapter
from evaldata.types import ExecutionSuccess, UntypedSchema


@pytest.mark.unit
class TestSqliteTypes:
    """SQLite is dynamically typed and the driver reports no result-column types."""

    @pytest.fixture
    def adapter(self) -> SqliteAdapter:
        return SqliteAdapter()

    @pytest.mark.parametrize(
        ("sql", "names"),
        [
            ("SELECT 1 AS x", ["x"]),
            ("SELECT 'hello' AS x", ["x"]),
            ("SELECT a * b AS x FROM (SELECT 2 AS a, 3 AS b)", ["x"]),
            ("SELECT 1 AS a, 2 AS b", ["a", "b"]),
            ("SELECT NULL AS x", ["x"]),
        ],
    )
    def test_schema_is_untyped_with_names(self, adapter: SqliteAdapter, sql: str, names: list[str]) -> None:
        # The schema carries the column names; types are absent rather than guessed.
        result = adapter.execute(sql)
        assert isinstance(result, ExecutionSuccess)
        assert isinstance(result.schema_, UntypedSchema)
        assert result.schema_.names == names


@pytest.mark.unit
class TestSqliteFilePath:
    """`database=` is honoured — a file-backed DB persists across reopens."""

    def test_file_backed_database_persists(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.sqlite")
        with SqliteAdapter(database=db_path) as first:
            first.execute("CREATE TABLE t (x INTEGER)")
            first.execute("INSERT INTO t VALUES (42)")
        with SqliteAdapter(database=db_path) as second:
            result = second.execute("SELECT x FROM t")
            assert isinstance(result, ExecutionSuccess)
            assert result.rows == [{"x": 42}]
