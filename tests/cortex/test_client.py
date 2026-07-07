"""Unit tests for `CortexAnalystClient` — request building, status mapping, and token reuse.

The client's HTTP call is exercised offline by injecting a fake `requests.Session`; the live
Cortex Analyst endpoint is covered by the vcr replay test and the `cortex`-marked e2e.
"""

from dataclasses import dataclass
from typing import Any

import pytest
import requests

from evaldata.cortex.client import CortexAnalystClient, _session_token
from evaldata.types import SolverError


class _FakeResponse:
    """A minimal stand-in for `requests.Response`."""

    def __init__(self, status_code: int, json_data: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self) -> Any:
        if self._json is None:
            msg = "no json body"
            raise ValueError(msg)
        return self._json


class _FakeSession:
    """Records the last `post` call and returns a fixed response or raises a fixed exception."""

    def __init__(self, response: _FakeResponse | None = None, exc: Exception | None = None) -> None:
        self._response = response
        self._exc = exc
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> _FakeResponse:
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return self._response


@dataclass
class _FakeRest:
    token: str | None


@dataclass
class _FakeConnection:
    host: str
    rest: _FakeRest | None


@pytest.mark.unit
class TestSend:
    def _client(self, session: _FakeSession) -> CortexAnalystClient:
        return CortexAnalystClient(host="acct.snowflakecomputing.com", token_provider=lambda: "tok", session=session)

    def test_builds_the_request(self) -> None:
        session = _FakeSession(_FakeResponse(200, {"message": {"content": []}}))
        self._client(session).send("how many customers?", {"semantic_view": "DB.S.V"})

        call = session.calls[0]
        assert call["url"] == "https://acct.snowflakecomputing.com/api/v2/cortex/analyst/message"
        assert call["headers"]["Authorization"] == 'Snowflake Token="tok"'
        assert call["json"]["stream"] is False
        assert call["json"]["semantic_view"] == "DB.S.V"
        assert call["json"]["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "how many customers?"}]}
        ]

    def test_returns_json_on_200(self) -> None:
        body = {"message": {"content": [{"type": "sql", "statement": "SELECT 1"}]}}
        result = self._client(_FakeSession(_FakeResponse(200, body))).send("q", {"semantic_view": "V"})
        assert result == body

    @pytest.mark.parametrize(
        ("status", "kind"),
        [
            (400, "bad_request"),
            (401, "auth"),
            (403, "auth"),
            (404, "bad_request"),
            (429, "rate_limit"),
            (500, "api_error"),
        ],
    )
    def test_maps_http_status_to_error_kind(self, status: int, kind: str) -> None:
        session = _FakeSession(_FakeResponse(status, {"message": "boom"}))
        result = self._client(session).send("q", {"semantic_view": "V"})
        assert isinstance(result, SolverError)
        assert result.kind == kind
        assert result.message == "boom"
        assert result.provider == "cortex_analyst"

    def test_error_falls_back_to_body_text_when_no_json(self) -> None:
        session = _FakeSession(_FakeResponse(500, json_data=None, text="internal error"))
        result = self._client(session).send("q", {"semantic_view": "V"})
        assert isinstance(result, SolverError)
        assert result.message == "internal error"

    def test_error_falls_back_to_body_text_when_message_missing(self) -> None:
        session = _FakeSession(_FakeResponse(400, json_data={"code": "x"}, text="fallback text"))
        result = self._client(session).send("q", {"semantic_view": "V"})
        assert isinstance(result, SolverError)
        assert result.message == "fallback text"

    @pytest.mark.parametrize(
        ("exc", "kind"),
        [
            (requests.Timeout("slow"), "timeout"),
            (requests.ConnectionError("down"), "api_connection"),
            (requests.TooManyRedirects("loop"), "api_error"),
        ],
    )
    def test_maps_transport_exceptions(self, exc: Exception, kind: str) -> None:
        result = self._client(_FakeSession(exc=exc)).send("q", {"semantic_view": "V"})
        assert isinstance(result, SolverError)
        assert result.kind == kind
        assert result.cause is exc

    def test_creates_a_session_when_none_given(self) -> None:
        client = CortexAnalystClient(host="h", token_provider=lambda: "t")
        assert isinstance(client._session, requests.Session)  # noqa: SLF001


@pytest.mark.unit
class TestFromConnection:
    def test_reuses_host_and_session_token(self) -> None:
        client = CortexAnalystClient.from_connection(
            _FakeConnection("acct.snowflakecomputing.com", _FakeRest("sess-tok"))
        )
        assert client._host == "acct.snowflakecomputing.com"  # noqa: SLF001
        assert client._token_provider() == "sess-tok"  # noqa: SLF001

    def test_session_token_raises_without_a_token(self) -> None:
        with pytest.raises(RuntimeError, match="no active session token"):
            _session_token(_FakeConnection("h", _FakeRest(None)))

    def test_session_token_raises_without_a_rest_channel(self) -> None:
        with pytest.raises(RuntimeError, match="no active session token"):
            _session_token(_FakeConnection("h", None))
