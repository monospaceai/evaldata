"""`PlatformAdapter` Protocol: the contract every platform integration implements."""

import threading
import time
from collections import Counter
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from evaldata.types import (
    Column,
    ExecutionError,
    ExecutionErrorKind,
    ExecutionResult,
    Schema,
    SqlType,
    TypedSchema,
    UntypedSchema,
)


@runtime_checkable
class PlatformAdapter(Protocol):
    """Executes SQL against a data platform; returns rows + schema + latency.

    Required behavior:
        * On success: return `ExecutionResult` with `rows` populated and `schema_`
          populated — a `TypedSchema` (each `Column.type` a `SqlType` built from the
          driver's native type string in the adapter's static dialect), or an
          `UntypedSchema` (names only) when the driver reports no result-column types —
          with non-negative `latency_seconds` and `error is None`.
        * On query failure: return `ExecutionResult` with `rows=[]`,
          `schema_=None`, an `error` describing the failure, and non-negative
          `latency_seconds`. **Do NOT raise.**
        * `latency_seconds` measures wall-clock time spent inside `execute()`.
    """

    def execute(self, sql: str) -> ExecutionResult:
        """Execute one SQL statement and return its structured result."""
        ...

    def cancel(self) -> None:
        """Abort the query currently executing on this connection, if any.

        Must be safe to call from another thread while `execute` is blocked. A no-op when no
        query is running, and best-effort: it must not raise — a cancellation that fails or
        arrives late is non-fatal.
        """
        ...

    def close(self) -> None:
        """Release the underlying connection/resources."""
        ...


@runtime_checkable
class NativeTimeoutAdapter(Protocol):
    """Capability for executing one statement with a backend-native deadline."""

    def execute_with_timeout(self, sql: str, timeout_seconds: float) -> ExecutionResult:
        """Execute `sql` once with a backend-native timeout.

        Args:
            sql: The SQL statement to execute.
            timeout_seconds: The positive statement deadline in seconds.

        Returns:
            The structured execution result.
        """
        ...


@runtime_checkable
class ReusableStateAdapter(Protocol):
    """Capability for reporting whether local session state remains reusable."""

    def is_reusable(self) -> bool:
        """Return local, non-I/O session reuse state."""
        ...


@runtime_checkable
class TypeResolvingAdapter(Protocol):
    """Capability for backends whose `execute().schema_` reports types with unresolved parameters.

    Implemented only by adapters whose driver drops type parameters (e.g. a `DECIMAL` scale
    or an `ARRAY` element type); precise backends do not implement it. Both methods must be
    pure (no I/O): one produces the SQL that probes for precise types, the other interprets
    that probe's result rows. The caller runs the probe.
    """

    def type_probe_sql(self, sql: str) -> str:
        """Build the probe statement that recovers precise types for `sql`'s projection.

        Args:
            sql: The statement whose projected column types to resolve.

        Returns:
            A probe statement for the caller to execute; its rows feed `types_from_probe`.
        """
        ...

    def types_from_probe(self, rows: list[dict[str, Any]]) -> list[SqlType] | ExecutionError:
        """Parse the probe's result rows into precise `SqlType`s, one per projected column.

        They must be returned in the order the query projects its columns.

        Args:
            rows: The rows returned by executing `type_probe_sql`, in projection order.

        Returns:
            One precise `SqlType` per projected column, in order, or an `ExecutionError` when
            the rows cannot yield types.
        """
        ...


@runtime_checkable
class BudgetedExecutionAdapter(Protocol):
    """Capability for a pool lease that owns bounded execution lifecycle state."""

    def execute_within_budget(
        self, sql: str, max_seconds: float, *, cancel_grace_seconds: float | None
    ) -> ExecutionResult:
        """Execute `sql` with the lease's timeout and quarantine handling."""
        ...


def execute_within_budget(
    adapter: PlatformAdapter,
    sql: str,
    max_seconds: float | None,
    *,
    cancel_grace_seconds: float | None = None,
) -> ExecutionResult:
    """Execute `sql` on `adapter`, cancelling it if it exceeds `max_seconds`.

    With `max_seconds` unset, runs `adapter.execute` directly. Otherwise runs it on a daemon
    worker thread and waits up to `max_seconds`. At expiry, cancellation runs independently,
    then the caller waits at most `cancel_grace_seconds` before returning a budget error. Pool
    leases own the corresponding quarantine and deferred-close lifecycle.

    Args:
        adapter: The platform adapter to execute against.
        sql: The SQL statement to execute.
        max_seconds: Wall-clock ceiling for the query, or `None` for no limit.
        cancel_grace_seconds: Extra time to allow cancellation to finish after expiry. Defaults
            to one second for direct adapters; pool leases use their configured policy.

    Returns:
        The adapter's `ExecutionResult` if the query finishes within budget, otherwise an
        `ExecutionResult` with `error` set and `latency_seconds` measuring the elapsed time.
    """
    if max_seconds is None:
        return adapter.execute(sql)
    if max_seconds <= 0:
        return _budget_exceeded(time.monotonic(), max_seconds)
    if isinstance(adapter, BudgetedExecutionAdapter):
        return adapter.execute_within_budget(sql, max_seconds, cancel_grace_seconds=cancel_grace_seconds)
    grace = 1.0 if cancel_grace_seconds is None else cancel_grace_seconds
    return _execute_with_watchdog(adapter, sql, max_seconds, cancel_grace_seconds=grace)


def _execute_with_watchdog(
    adapter: PlatformAdapter, sql: str, max_seconds: float, *, cancel_grace_seconds: float
) -> ExecutionResult:
    """Run a caller-owned adapter on a daemon worker without waiting for cancellation forever.

    The caller owns retirement of an adapter that outlives this function. A direct adapter has
    no pool lease to quarantine, so it must not be reused until its worker and cancellation have
    both finished.

    Returns:
        The adapter result when it completes before the deadline, else a budget error.
    """
    start = time.monotonic()
    deadline = start + max_seconds
    done = threading.Event()
    outcome: list[ExecutionResult] = []
    completed_at: list[float] = []

    def run() -> None:
        try:
            if isinstance(adapter, NativeTimeoutAdapter):
                outcome.append(adapter.execute_with_timeout(sql, max_seconds))
            else:
                outcome.append(adapter.execute(sql))
        except Exception as e:  # noqa: BLE001 - adapters promise errors-as-values, but watchdogs must finish safely
            outcome.append(
                ExecutionResult(
                    rows=[], schema=None, latency_seconds=time.monotonic() - start, error=execution_error(e)
                )
            )
        finally:
            completed_at.append(time.monotonic())
            done.set()

    threading.Thread(target=run, daemon=True).start()
    if done.wait(max(deadline - time.monotonic(), 0.0)) and completed_at[0] < deadline:
        return outcome[0]
    threading.Thread(target=_cancel_safely, args=(adapter,), daemon=True).start()
    grace_deadline = deadline + max(cancel_grace_seconds, 0.0)
    done.wait(max(grace_deadline - time.monotonic(), 0.0))
    return _budget_exceeded(start, max_seconds)


def _budget_exceeded(start: float, max_seconds: float) -> ExecutionResult:
    """Build the timeout result after the original deadline has elapsed.

    Returns:
        A result containing a `budget_exceeded` error.
    """
    return ExecutionResult(
        rows=[],
        schema=None,
        latency_seconds=time.monotonic() - start,
        error=ExecutionError(
            kind="budget_exceeded", message=f"exceeded cost budget: query did not complete within {max_seconds}s"
        ),
    )


def _cancel_safely(adapter: PlatformAdapter) -> None:
    """Request adapter cancellation without letting a broken driver escape its daemon thread."""
    try:
        adapter.cancel()
    except Exception:  # noqa: BLE001 - cancellation is best effort
        return


def _driver_call(exc: Exception, name: str) -> Any:
    """Return `exc.<name>()` when it is a no-arg method that doesn't raise, else `None`.

    Args:
        exc: The exception to probe.
        name: The accessor method to call (e.g. `"getSqlState"`).

    Returns:
        The method's return value, or `None` when the attribute is absent, not callable, or
        raises.
    """
    method = getattr(exc, name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:  # noqa: BLE001 - probing an optional driver accessor; any failure means "unavailable"
        return None


def execution_error(exc: Exception, kind: ExecutionErrorKind = "query_failed") -> ExecutionError:
    """Translate a driver exception into a typed `ExecutionError`, preserving structured detail.

    Reads whatever structured attributes the exception happens to expose — SQLSTATE, an error
    condition/class, message parameters — by duck-typing, so no driver package is imported or
    enumerated and no driver message is parsed. The live exception is kept as `cause` for
    debugging; `message` falls back to the exception's class name when its string form is empty.

    Args:
        exc: The driver exception to translate.
        kind: The `ExecutionError` classifier to assign.

    Returns:
        An `ExecutionError` carrying `kind`, a non-empty `message`, any recovered structured
        fields, and `exc` as `cause`.
    """
    sqlstate = getattr(exc, "sqlstate", None) or getattr(exc, "pgcode", None) or _driver_call(exc, "getSqlState")
    condition = (
        _driver_call(exc, "getCondition") or _driver_call(exc, "getErrorClass") or getattr(exc, "error_code", None)
    )
    raw_params = _driver_call(exc, "getMessageParameters")
    params = {str(k): str(v) for k, v in raw_params.items()} if isinstance(raw_params, dict) and raw_params else None
    return ExecutionError(
        kind=kind,
        message=str(exc) or type(exc).__name__,
        sqlstate=str(sqlstate) if sqlstate else None,
        condition=str(condition) if condition else None,
        params=params,
        cause=exc,
    )


def rows_or_error(columns: list[Column], rows_raw: list[tuple[Any, ...]], latency_seconds: float) -> ExecutionResult:
    """Build a successful typed `ExecutionResult`, or an error one if column names collide.

    Args:
        columns: The result-set columns, in order, as reported by the driver.
        rows_raw: The positional row tuples, aligned with `columns`.
        latency_seconds: Wall-clock time spent executing the query.

    Returns:
        An `ExecutionResult` with name-keyed rows and a `TypedSchema`, or `error` set (and no
        rows or schema) when one or more column names are duplicated.
    """
    return _result_or_error(lambda: TypedSchema(root=columns), [c.name for c in columns], rows_raw, latency_seconds)


def untyped_rows_or_error(names: list[str], rows_raw: list[tuple[Any, ...]], latency_seconds: float) -> ExecutionResult:
    """Build a successful untyped `ExecutionResult`, or an error one if column names collide.

    For engines whose driver reports no result-column types (e.g. SQLite): the schema carries
    column names only.

    Args:
        names: The result-column names, in order, as reported by the driver.
        rows_raw: The positional row tuples, aligned with `names`.
        latency_seconds: Wall-clock time spent executing the query.

    Returns:
        An `ExecutionResult` with name-keyed rows and an `UntypedSchema`, or `error` set (and
        no rows or schema) when one or more column names are duplicated.
    """
    return _result_or_error(lambda: UntypedSchema(root=names), names, rows_raw, latency_seconds)


def _result_or_error(
    build_schema: Callable[[], Schema], names: list[str], rows_raw: list[tuple[Any, ...]], latency_seconds: float
) -> ExecutionResult:
    """Key `rows_raw` by `names`, or return an error if names collide.

    Rows are keyed by name (`dict(zip(...))`), which cannot represent two columns sharing a
    name — the later value would silently overwrite the earlier. Rather than lose data,
    duplicate output column names are surfaced as `ExecutionResult.error`. The check runs
    before `build_schema`, which itself rejects duplicate names.

    Args:
        build_schema: Builds the schema to attach (typed or untyped); called only once the
            names are known unique.
        names: The column names, in order, aligned with `rows_raw`.
        rows_raw: The positional row tuples.
        latency_seconds: Wall-clock time spent executing the query.

    Returns:
        An `ExecutionResult` with name-keyed rows and the built schema, or `error` set (and no
        rows or schema) when one or more column names are duplicated.
    """
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        listed = ", ".join(repr(name) for name in duplicates)
        return ExecutionResult(
            rows=[],
            schema=None,
            latency_seconds=latency_seconds,
            error=ExecutionError(kind="duplicate_columns", message=f"duplicate output column name(s): {listed}"),
        )
    rows = [dict(zip(names, row, strict=True)) for row in rows_raw]
    return ExecutionResult(rows=rows, schema=build_schema(), latency_seconds=latency_seconds)
