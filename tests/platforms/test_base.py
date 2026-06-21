"""Unit tests for the platform-layer error translator."""

import pytest

from evaldata.platforms.base import execution_error


@pytest.mark.unit
class TestExecutionError:
    """`execution_error` duck-types structured detail off a driver exception, losing nothing."""

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
            def getSqlState(self) -> str:  # noqa: N802 - mirrors PySpark's accessor name
                return "42P01"

            def getCondition(self) -> str:  # noqa: N802 - mirrors PySpark's accessor name
                return "TABLE_OR_VIEW_NOT_FOUND"

            def getMessageParameters(self) -> dict[str, str]:  # noqa: N802 - mirrors PySpark's accessor name
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
            def getSqlState(self) -> str:  # noqa: N802 - mirrors PySpark's accessor name
                msg = "accessor blew up"
                raise ValueError(msg)

        assert execution_error(BadAccessor("x")).sqlstate is None

    def test_cause_excluded_from_serialization(self) -> None:
        dumped = execution_error(RuntimeError("boom")).model_dump()
        assert "cause" not in dumped
