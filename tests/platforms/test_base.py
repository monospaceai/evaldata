"""Unit tests for the platform-layer error translator."""

import threading

import pytest

import evaldata.platforms.base as platform_base
from evaldata.platforms.base import execute_within_budget, execution_error
from evaldata.types import ExecutionResult


@pytest.mark.unit
class TestExecutionError:
    def test_plain_exception_yields_query_failed_with_message_and_cause(self) -> None:
        exc = RuntimeError("boom")
        error = execution_error(exc)
        assert error.kind == "query_failed"
        assert error.message == "boom"
        assert error.cause is exc
        assert error.sqlstate is None
        assert error.condition is None
        assert error.params is None

    def test_kind_override(self) -> None:
        assert execution_error(ValueError("x"), kind="type_probe_failed").kind == "type_probe_failed"

    def test_empty_message_falls_back_to_class_name(self) -> None:
        assert execution_error(RuntimeError()).message == "RuntimeError"

    def test_sqlstate_from_attribute(self) -> None:
        exc = RuntimeError("nope")
        exc.sqlstate = "42P01"  # type: ignore[attr-defined]
        assert execution_error(exc).sqlstate == "42P01"

    def test_sqlstate_from_pgcode_attribute(self) -> None:
        exc = RuntimeError("nope")
        exc.pgcode = "42P01"  # type: ignore[attr-defined]
        assert execution_error(exc).sqlstate == "42P01"

    def test_structured_detail_from_spark_style_accessors(self) -> None:
        class SparkLikeError(Exception):
            def getSqlState(self) -> str:  # noqa: N802
                return "42P01"

            def getCondition(self) -> str:  # noqa: N802
                return "TABLE_OR_VIEW_NOT_FOUND"

            def getMessageParameters(self) -> dict[str, str]:  # noqa: N802
                return {"relationName": "`x`"}

        error = execution_error(SparkLikeError("table x not found"))
        assert error.sqlstate == "42P01"
        assert error.condition == "TABLE_OR_VIEW_NOT_FOUND"
        assert error.params == {"relationName": "`x`"}

    def test_condition_from_error_code_attribute(self) -> None:
        exc = RuntimeError("nope")
        exc.error_code = "RESOURCE_DOES_NOT_EXIST"  # type: ignore[attr-defined]
        assert execution_error(exc).condition == "RESOURCE_DOES_NOT_EXIST"

    def test_raising_accessor_is_ignored(self) -> None:
        class BadAccessor(Exception):
            def getSqlState(self) -> str:  # noqa: N802
                msg = "accessor blew up"
                raise ValueError(msg)

        assert execution_error(BadAccessor("x")).sqlstate is None

    def test_cause_excluded_from_serialization(self) -> None:
        dumped = execution_error(RuntimeError("boom")).model_dump()
        assert "cause" not in dumped


@pytest.mark.unit
class TestWatchdog:
    def test_unbounded_execution_calls_ordinary_execute(self) -> None:
        class Adapter:
            def execute(self, sql: str) -> ExecutionResult:
                return ExecutionResult(rows=[], schema=None, latency_seconds=0.0)

            def cancel(self) -> None:
                return

            def close(self) -> None:
                return

        assert execute_within_budget(Adapter(), "SELECT 1", None).error is None

    def test_native_timeout_adapter_executes_once_with_exact_watchdog_deadline(self) -> None:
        class Adapter:
            def __init__(self) -> None:
                self.calls: list[tuple[str, float]] = []

            def execute(self, sql: str) -> ExecutionResult:
                msg = "watchdog must use the native deadline capability"
                raise AssertionError(msg)

            def execute_with_timeout(self, sql: str, timeout_seconds: float) -> ExecutionResult:
                self.calls.append((sql, timeout_seconds))
                return ExecutionResult(rows=[], schema=None, latency_seconds=0.0)

            def cancel(self) -> None:
                return

            def close(self) -> None:
                return

        adapter = Adapter()
        result = execute_within_budget(adapter, "SELECT 1", 0.125)
        assert result.error is None
        assert adapter.calls == [("SELECT 1", 0.125)]

    def test_late_direct_completion_returns_a_budget_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Adapter:
            def __init__(self) -> None:
                self.started = threading.Event()
                self.finish = threading.Event()

            def execute(self, sql: str) -> ExecutionResult:
                self.started.set()
                self.finish.wait()
                now["value"] = 2.0
                return ExecutionResult(rows=[], schema=None, latency_seconds=0.0)

            def cancel(self) -> None:
                return

            def close(self) -> None:
                return

        now = {"value": 0.0}
        monkeypatch.setattr(platform_base.time, "monotonic", lambda: now["value"])
        adapter = Adapter()
        results: list[ExecutionResult] = []
        returned = threading.Event()

        def run() -> None:
            results.append(execute_within_budget(adapter, "SELECT 1", 1.0, cancel_grace_seconds=0.0))
            returned.set()

        thread = threading.Thread(target=run)
        thread.start()
        assert adapter.started.wait(1)
        adapter.finish.set()
        assert returned.wait(1)
        thread.join(1)
        assert results[0].error is not None
        assert results[0].error.kind == "budget_exceeded"

    def test_non_positive_budget_does_not_start_execution(self) -> None:
        class Adapter:
            def __init__(self) -> None:
                self.executed = False

            def execute(self, sql: str) -> ExecutionResult:
                self.executed = True
                return ExecutionResult(rows=[], schema=None, latency_seconds=0.0)

            def cancel(self) -> None:
                return

            def close(self) -> None:
                return

        adapter = Adapter()
        result = execute_within_budget(adapter, "SELECT 1", 0)
        assert result.error is not None
        assert result.error.kind == "budget_exceeded"
        assert adapter.executed is False

    def test_direct_timeout_tolerates_a_raising_cancel(self) -> None:
        import threading

        class Adapter:
            def __init__(self) -> None:
                self.release = threading.Event()
                self.cancel_attempted = threading.Event()

            def execute(self, sql: str) -> ExecutionResult:
                self.release.wait()
                return ExecutionResult(rows=[], schema=None, latency_seconds=0.0)

            def cancel(self) -> None:
                self.cancel_attempted.set()
                message = "cancel failed"
                raise RuntimeError(message)

            def close(self) -> None:
                return

        adapter = Adapter()
        result = execute_within_budget(adapter, "SELECT 1", 0.01, cancel_grace_seconds=0.0)
        assert adapter.cancel_attempted.wait(1)
        adapter.release.set()
        assert result.error is not None
