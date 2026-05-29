"""Shared conformance battery: every ``PlatformAdapter`` must pass these tests identically.

The contract is *behavioural*, not syntactic: each test names what a SQL fragment
must express ("returns three rows", "triggers a parse error"), and the subclass
supplies the dialect-specific phrasing as ``ClassVar`` attributes. This keeps the
battery honest across platforms where SQL syntax genuinely diverges (BigQuery's
``INT64`` vs ``BIGINT``; Snowflake/BQ row-constructor differences; ``CAST(NULL AS …)``
being needed where bare ``NULL`` is untyped).

Subclasses subclass ``PlatformAdapterConformance``, provide an ``adapter`` fixture,
and override any ``SQL_*`` attributes that don't work in the platform's dialect.
The base class is intentionally NOT prefixed ``Test``, so pytest does not collect
it standalone — only its subclasses execute, inheriting every ``test_*`` method.
"""

from typing import ClassVar

import pytest

from data_eval.platforms.base import PlatformAdapter


class PlatformAdapterConformance:
    """Contract tests every ``PlatformAdapter`` must satisfy.

    Each ``SQL_*`` ClassVar names what its fragment must express. Subclasses
    override any that don't parse / execute in their platform's dialect.
    """

    # SQL fragments — subclasses override per dialect.
    # The defaults are written in the most ANSI-portable form we can manage;
    # adapters whose dialect needs different phrasing override.

    #: A query returning one row with one column named ``n``.
    SQL_ONE_ROW_ONE_COLUMN: ClassVar[str] = "SELECT 1 AS n"
    #: A query whose result set has a known schema but zero rows.
    SQL_EMPTY_RESULT: ClassVar[str] = "SELECT 1 AS n WHERE 1=0"
    #: A query returning three rows with one column named ``n``, values 1, 2, 3.
    SQL_THREE_ROWS: ClassVar[str] = "SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3"
    #: A query returning one row containing a single NULL value in column ``x``.
    SQL_NULL_VALUE: ClassVar[str] = "SELECT NULL AS x"
    #: A query referencing a table that does not exist (catalog / binder error).
    SQL_REFERENCES_MISSING_TABLE: ClassVar[str] = "SELECT * FROM does_not_exist_xyz"
    #: A query that fails to parse (syntactic error).
    SQL_PARSE_ERROR: ClassVar[str] = "SELECT FROM nope"

    @pytest.fixture
    def adapter(self) -> PlatformAdapter:
        """Return a fresh ``PlatformAdapter`` instance. Subclasses must override."""
        raise NotImplementedError

    def test_execute_returns_rows_and_schema(self, adapter: PlatformAdapter) -> None:
        result = adapter.execute(self.SQL_ONE_ROW_ONE_COLUMN)
        assert result.error is None
        assert result.rows == [{"n": 1}]
        assert result.schema_ is not None
        assert len(result.schema_) == 1
        assert result.schema_[0].name == "n"
        assert result.schema_[0].type  # non-empty native type string

    def test_empty_result_set_keeps_schema(self, adapter: PlatformAdapter) -> None:
        result = adapter.execute(self.SQL_EMPTY_RESULT)
        assert result.error is None
        assert result.rows == []
        assert result.schema_ is not None
        assert len(result.schema_) == 1
        assert result.schema_[0].name == "n"

    def test_multiple_rows_returned(self, adapter: PlatformAdapter) -> None:
        result = adapter.execute(self.SQL_THREE_ROWS)
        assert result.error is None
        assert len(result.rows) == 3
        assert sorted(r["n"] for r in result.rows) == [1, 2, 3]

    def test_null_values_round_trip(self, adapter: PlatformAdapter) -> None:
        result = adapter.execute(self.SQL_NULL_VALUE)
        assert result.error is None
        assert result.rows == [{"x": None}]

    def test_missing_table_returns_error_not_exception(self, adapter: PlatformAdapter) -> None:
        result = adapter.execute(self.SQL_REFERENCES_MISSING_TABLE)
        assert result.error is not None
        assert result.error  # non-empty
        assert result.rows == []
        assert result.schema_ is None

    def test_parse_error_returns_error_not_exception(self, adapter: PlatformAdapter) -> None:
        result = adapter.execute(self.SQL_PARSE_ERROR)
        assert result.error is not None
        assert result.error  # non-empty
        assert result.rows == []
        assert result.schema_ is None

    def test_latency_is_measured_on_success(self, adapter: PlatformAdapter) -> None:
        result = adapter.execute(self.SQL_ONE_ROW_ONE_COLUMN)
        assert result.error is None
        assert result.latency_seconds >= 0

    def test_latency_is_measured_on_failure(self, adapter: PlatformAdapter) -> None:
        result = adapter.execute(self.SQL_PARSE_ERROR)
        assert result.error is not None
        assert result.latency_seconds >= 0
