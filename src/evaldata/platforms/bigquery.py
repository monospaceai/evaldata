"""`BigQueryAdapter`: BigQuery execution backend over `google-cloud-bigquery`."""

import contextlib
import math
import time
from types import TracebackType
from typing import Self, cast

from google.cloud import bigquery
from sqlglot import exp

from evaldata.platforms.base import execution_error, rows_or_error
from evaldata.types import Column, ExecutionFailure, ExecutionResult, ExecutionSuccess, SqlType


def _type_string(field: bigquery.SchemaField) -> str:
    """Render a BigQuery result-column field as a SQL type string.

    Recurses into `RECORD` fields to build a `STRUCT<...>`, and wraps a `REPEATED` field as an
    `ARRAY<...>`.

    Args:
        field: The result-column schema field reported by the driver.

    Returns:
        A SQL type string parseable by SQLGlot's bigquery dialect, e.g. `"INT64"`,
        `"NUMERIC(38,9)"`, `"ARRAY<STRING>"`, `"STRUCT<a INT64, b STRING>"`.
    """
    match field.field_type:
        case "INTEGER":
            base = "INT64"
        case "FLOAT":
            base = "FLOAT64"
        case "BOOLEAN":
            base = "BOOL"
        case "NUMERIC":
            base = f"NUMERIC({field.precision},{field.scale})" if field.precision is not None else "NUMERIC"
        case "BIGNUMERIC":
            base = f"BIGNUMERIC({field.precision},{field.scale})" if field.precision is not None else "BIGNUMERIC"
        case "STRING":
            base = f"STRING({field.max_length})" if field.max_length is not None else "STRING"
        case "BYTES":
            base = f"BYTES({field.max_length})" if field.max_length is not None else "BYTES"
        case "RECORD":
            fields = ", ".join(
                f"{exp.to_identifier(f.name, quoted=True).sql(dialect='bigquery')} {_type_string(f)}"
                for f in field.fields
            )
            base = f"STRUCT<{fields}>"
        case "RANGE":
            element = field.range_element_type
            base = f"RANGE<{element.element_type}>" if element is not None else "RANGE"
        case _:
            base = field.field_type
    return f"ARRAY<{base}>" if field.mode == "REPEATED" else base


def _column_from_field(field: bigquery.SchemaField) -> Column:
    """Build a `Column` from one result-column schema field.

    Args:
        field: The result-column schema field reported by the driver.

    Returns:
        A `Column` with a `SqlType` parsed in the bigquery dialect; `nullable` is `False` only
        for a `REQUIRED` field.
    """
    raw = _type_string(field)
    return Column(name=field.name, type=SqlType.parse(raw, "bigquery"), nullable=field.mode != "REQUIRED")


class BigQueryAdapter:
    """Executes SQL against BigQuery via google-cloud-bigquery."""

    def __init__(self, *, project: str, dataset: str | None = None, location: str | None = None) -> None:
        """Open a BigQuery client.

        Credentials are not passed here; they resolve through Application Default Credentials.

        Args:
            project: The Google Cloud project to run jobs and bill against.
            dataset: The default dataset for unqualified table names, or `None` to leave none.
            location: The location to run jobs in (e.g. `"US"`, `"EU"`), or `None` for the
                client default.
        """
        self._client = bigquery.Client(project=project, location=location)
        self._job_config = (
            bigquery.QueryJobConfig(default_dataset=f"{project}.{dataset}") if dataset is not None else None
        )
        self._job: bigquery.QueryJob | None = None

    @property
    def client(self) -> bigquery.Client:
        """The live BigQuery client backing this adapter."""
        return self._client

    def cancel(self) -> None:
        """Cancel the job currently executing on this client, if any.

        Safe to call from another thread while `execute` is blocked; best-effort, so all
        failures are swallowed.
        """
        job = self._job
        if job is not None:
            with contextlib.suppress(Exception):
                job.cancel()

    def close(self) -> None:
        """Release the underlying BigQuery client."""
        self._client.close()

    def __enter__(self) -> Self:
        """Return self; the client is already open from `__init__`."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the underlying client on context-manager exit."""
        self.close()

    def execute(self, sql: str) -> ExecutionResult:
        """Execute one SQL statement against BigQuery.

        Args:
            sql: The SQL statement to execute.

        Returns:
            An `ExecutionResult` with the returned rows, schema, and latency. Query
            failures are returned as an `ExecutionFailure` rather than raised.
        """
        return self._execute(sql)

    def execute_with_timeout(self, sql: str, timeout_seconds: float) -> ExecutionResult:
        """Submit one job with BigQuery's native job deadline.

        Args:
            sql: The SQL statement to execute.
            timeout_seconds: Positive job deadline in seconds.

        Returns:
            The structured job result.
        """
        config = self._copy_job_config()
        config.job_timeout_ms = math.ceil(timeout_seconds * 1000)
        return self._execute(sql, job_config=config)

    def _copy_job_config(self) -> bigquery.QueryJobConfig:
        """Return an independent copy of this adapter's base query configuration."""
        if self._job_config is None:
            return bigquery.QueryJobConfig()
        return cast(bigquery.QueryJobConfig, bigquery.QueryJobConfig.from_api_repr(self._job_config.to_api_repr()))

    def _execute(self, sql: str, *, job_config: bigquery.QueryJobConfig | None = None) -> ExecutionResult:
        """Submit user SQL once with the supplied immutable-per-call configuration.

        Returns:
            The structured job result.
        """
        start = time.perf_counter()
        try:
            job = self._client.query(sql, job_config=self._job_config if job_config is None else job_config)
            self._job = job
            iterator = job.result()
            schema = iterator.schema
            rows_raw = [tuple(row.values()) for row in iterator]
        except Exception as e:  # noqa: BLE001 - execute must never raise; failures return as ExecutionFailure
            elapsed = time.perf_counter() - start
            return ExecutionFailure(latency_seconds=elapsed, error=execution_error(e))
        finally:
            self._job = None
        elapsed = time.perf_counter() - start
        if not schema:
            return ExecutionSuccess(rows=[], schema=None, latency_seconds=elapsed)
        columns = [_column_from_field(field) for field in schema]
        return rows_or_error(columns, rows_raw, elapsed)
