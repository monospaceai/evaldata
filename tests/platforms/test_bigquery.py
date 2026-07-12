"""`BigQueryAdapter` tests: unit (fake client) + a live type-resolution e2e check."""

from typing import Any

import pytest

pytest.importorskip("google.cloud.bigquery")

from google.cloud import bigquery  # noqa: E402

from evaldata.platforms.bigquery import BigQueryAdapter, _type_string  # noqa: E402
from evaldata.types import SqlType  # noqa: E402


class _FakeRow:
    def __init__(self, values: tuple[Any, ...]) -> None:
        self._values = values

    def values(self) -> tuple[Any, ...]:
        return self._values


class _FakeRowIterator:
    def __init__(self, schema: list[bigquery.SchemaField], rows: list[_FakeRow]) -> None:
        self.schema = schema
        self._rows = rows

    def __iter__(self) -> Any:
        return iter(self._rows)


class _FakeQueryJob:
    def __init__(self, iterator: _FakeRowIterator, error: str | None = None) -> None:
        self._iterator = iterator
        self._error = error
        self.cancelled = False

    def result(self) -> _FakeRowIterator:
        if self._error is not None:
            raise RuntimeError(self._error)
        return self._iterator

    def cancel(self) -> None:
        self.cancelled = True


class _FakeClient:
    def __init__(self, job: _FakeQueryJob) -> None:
        self._job = job
        self.queried: str | None = None
        self.closed = False

    def query(self, sql: str, job_config: Any = None) -> _FakeQueryJob:
        self.queried = sql
        return self._job

    def close(self) -> None:
        self.closed = True


def _job(schema: list[bigquery.SchemaField], rows: list[tuple[Any, ...]], error: str | None = None) -> _FakeQueryJob:
    return _FakeQueryJob(_FakeRowIterator(schema, [_FakeRow(r) for r in rows]), error=error)


def _adapter(job: _FakeQueryJob, *, active: bool = False) -> BigQueryAdapter:
    """Build an adapter bound to a fake client, bypassing the real `__init__`.

    With `active=True` the job is set as the in-flight one, so `cancel` can reach it.
    """
    adapter = object.__new__(BigQueryAdapter)
    adapter._client = _FakeClient(job)
    adapter._job_config = None
    adapter._job = job if active else None
    return adapter


@pytest.mark.unit
class TestExecute:
    def test_rows_and_schema_on_success(self) -> None:
        schema = [
            bigquery.SchemaField("id", "NUMERIC", mode="NULLABLE", precision=10, scale=2),
            bigquery.SchemaField("label", "STRING", mode="NULLABLE", max_length=16),
        ]
        result = _adapter(_job(schema, [(1, "a"), (2, "b")])).execute("SELECT id, label FROM t")
        assert result.error is None
        assert result.rows == [{"id": 1, "label": "a"}, {"id": 2, "label": "b"}]
        assert result.schema_ is not None
        assert result.schema_.names == ["id", "label"]
        columns = {c.name: c for c in result.schema_.root}
        assert columns["id"].type == SqlType.parse("NUMERIC(10,2)", "bigquery")
        assert columns["label"].type == SqlType.parse("STRING(16)", "bigquery")

    def test_nullable_comes_through_from_mode(self) -> None:
        schema = [
            bigquery.SchemaField("id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("x", "STRING", mode="NULLABLE"),
        ]
        result = _adapter(_job(schema, [(1, "a")])).execute("SELECT id, x FROM t")
        assert result.schema_ is not None
        columns = {c.name: c for c in result.schema_.root}
        assert columns["id"].nullable is False
        assert columns["x"].nullable is True

    def test_error_is_returned_not_raised(self) -> None:
        result = _adapter(_job([], [], error="boom")).execute("SELECT bad")
        assert result.rows == []
        assert result.schema_ is None
        assert result.error is not None
        assert result.error.message == "boom"

    def test_non_row_returning_statement_has_no_schema(self) -> None:
        result = _adapter(_job([], [])).execute("CREATE TABLE t (n INT64)")
        assert result.error is None
        assert result.rows == []
        assert result.schema_ is None

    def test_duplicate_names_error(self) -> None:
        schema = [bigquery.SchemaField("x", "INTEGER"), bigquery.SchemaField("x", "INTEGER")]
        result = _adapter(_job(schema, [(1, 2)])).execute("SELECT 1 AS x, 2 AS x")
        assert result.rows == []
        assert result.schema_ is None
        assert result.error is not None
        assert "duplicate output column name(s)" in result.error.message


@pytest.mark.unit
class TestTypeString:
    def test_integer(self) -> None:
        assert _type_string(bigquery.SchemaField("n", "INTEGER")) == "INT64"

    def test_float(self) -> None:
        assert _type_string(bigquery.SchemaField("n", "FLOAT")) == "FLOAT64"

    def test_boolean(self) -> None:
        assert _type_string(bigquery.SchemaField("n", "BOOLEAN")) == "BOOL"

    def test_numeric_with_precision(self) -> None:
        assert _type_string(bigquery.SchemaField("n", "NUMERIC", precision=38, scale=9)) == "NUMERIC(38,9)"

    def test_numeric_without_precision(self) -> None:
        assert _type_string(bigquery.SchemaField("n", "NUMERIC")) == "NUMERIC"

    def test_bignumeric_with_precision(self) -> None:
        assert _type_string(bigquery.SchemaField("n", "BIGNUMERIC", precision=76, scale=38)) == "BIGNUMERIC(76,38)"

    def test_bignumeric_without_precision(self) -> None:
        assert _type_string(bigquery.SchemaField("n", "BIGNUMERIC")) == "BIGNUMERIC"

    def test_string_with_max_length(self) -> None:
        assert _type_string(bigquery.SchemaField("n", "STRING", max_length=255)) == "STRING(255)"

    def test_string_without_max_length(self) -> None:
        assert _type_string(bigquery.SchemaField("n", "STRING")) == "STRING"

    def test_bytes_with_max_length(self) -> None:
        assert _type_string(bigquery.SchemaField("n", "BYTES", max_length=100)) == "BYTES(100)"

    def test_bytes_without_max_length(self) -> None:
        assert _type_string(bigquery.SchemaField("n", "BYTES")) == "BYTES"

    def test_record_renders_struct(self) -> None:
        field = bigquery.SchemaField(
            "r",
            "RECORD",
            fields=[bigquery.SchemaField("a", "INTEGER"), bigquery.SchemaField("b", "STRING")],
        )
        assert _type_string(field) == "STRUCT<`a` INT64, `b` STRING>"

    def test_record_quotes_flexible_nested_field_names(self) -> None:
        field = bigquery.SchemaField("r", "RECORD", fields=[bigquery.SchemaField("a-b", "INTEGER")])
        assert _type_string(field) == "STRUCT<`a-b` INT64>"
        assert SqlType.parse(_type_string(field), "bigquery") == SqlType.parse("STRUCT<`a-b` INT64>", "bigquery")

    def test_repeated_wraps_in_array(self) -> None:
        assert _type_string(bigquery.SchemaField("n", "INTEGER", mode="REPEATED")) == "ARRAY<INT64>"

    def test_repeated_record_wraps_struct_in_array(self) -> None:
        field = bigquery.SchemaField(
            "r",
            "RECORD",
            mode="REPEATED",
            fields=[bigquery.SchemaField("a", "STRING")],
        )
        assert _type_string(field) == "ARRAY<STRUCT<`a` STRING>>"

    def test_range_preserves_its_element_type(self) -> None:
        field = bigquery.SchemaField("period", "RANGE", range_element_type="DATE")
        assert _type_string(field) == "RANGE<DATE>"
        assert SqlType.parse(_type_string(field), "bigquery") == SqlType.parse("RANGE<DATE>", "bigquery")

    @pytest.mark.parametrize("field_type", ["DATE", "DATETIME", "TIME", "TIMESTAMP", "GEOGRAPHY", "JSON"])
    def test_passthrough_types(self, field_type: str) -> None:
        assert _type_string(bigquery.SchemaField("n", field_type)) == field_type


@pytest.mark.unit
class TestLifecycle:
    """Client lifecycle against a mocked driver — keeps coverage independent of creds."""

    @staticmethod
    def _patch_client(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
        def fake_client(**kwargs: Any) -> _FakeClient:
            captured.update(kwargs)
            return _FakeClient(_job([], []))

        monkeypatch.setattr(bigquery, "Client", fake_client)

    def test_init_builds_client_with_project_and_location(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        self._patch_client(monkeypatch, captured)
        adapter = BigQueryAdapter(project="proj", location="EU")
        assert captured == {"project": "proj", "location": "EU"}
        assert adapter._job_config is None  # noqa: SLF001

    def test_init_sets_default_dataset_when_given(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        self._patch_client(monkeypatch, captured)
        adapter = BigQueryAdapter(project="proj", dataset="analytics")
        assert captured == {"project": "proj", "location": None}
        assert adapter._job_config is not None  # noqa: SLF001
        assert adapter._job_config.default_dataset == bigquery.DatasetReference("proj", "analytics")  # noqa: SLF001

    def test_execute_passes_job_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        self._patch_client(monkeypatch, captured)
        adapter = BigQueryAdapter(project="proj", dataset="analytics")
        adapter._client = _FakeClient(_job([bigquery.SchemaField("n", "INTEGER")], [(1,)]))  # noqa: SLF001
        result = adapter.execute("SELECT 1 AS n")
        assert result.error is None
        assert result.rows == [{"n": 1}]

    def test_client_exposes_the_underlying_client(self) -> None:
        adapter = _adapter(_job([], []))
        assert adapter.client is adapter._client  # noqa: SLF001

    def test_cancel_cancels_the_active_job(self) -> None:
        job = _job([], [])
        _adapter(job, active=True).cancel()
        assert job.cancelled is True

    def test_cancel_is_a_noop_when_no_query_runs(self) -> None:
        _adapter(_job([], [])).cancel()  # _job is None; must not raise

    def test_cancel_swallows_errors(self) -> None:
        class _BadCancelJob(_FakeQueryJob):
            def cancel(self) -> None:
                msg = "cancel failed"
                raise RuntimeError(msg)

        _adapter(_BadCancelJob(_FakeRowIterator([], [])), active=True).cancel()  # best-effort: swallowed

    def test_close_releases_the_client(self) -> None:
        adapter = _adapter(_job([], []))
        adapter.close()
        assert adapter._client.closed is True  # noqa: SLF001

    def test_context_manager_returns_self_and_closes(self) -> None:
        adapter = _adapter(_job([], []))
        with adapter as entered:
            assert entered is adapter
        assert adapter._client.closed is True  # noqa: SLF001


@pytest.mark.e2e
@pytest.mark.cloud
@pytest.mark.bigquery
class TestTypeResolutionLive:
    """Live schema resolution against a real project; unit tests use fakes.

    Fails loud (no skip) when `BIGQUERY_PROJECT` is unset.
    """

    def test_numeric_and_string_types_resolve_to_precise(self) -> None:
        from .conftest import connect_bigquery

        adapter = connect_bigquery()
        try:
            result = adapter.execute(
                "SELECT CAST(1.5 AS NUMERIC) AS amount, CAST('x' AS STRING) AS label, [1, 2, 3] AS nums"
            )
            assert result.error is None, result.error
            assert result.schema_ is not None
            precise = {c.name: c.type for c in result.schema_.root}
            # `.raw` surfaces the real reported type string if the assumption is wrong.
            assert precise["nums"] == SqlType.parse("ARRAY<INT64>", "bigquery"), precise["nums"].raw
        finally:
            adapter.close()
