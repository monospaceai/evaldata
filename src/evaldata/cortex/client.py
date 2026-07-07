"""`CortexAnalystClient`: a thin HTTP client for the Snowflake Cortex Analyst REST endpoint."""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import requests
from snowflake.connector import SnowflakeConnection

from evaldata.types import SolverError, SolverErrorKind

_ENDPOINT = "/api/v2/cortex/analyst/message"


@runtime_checkable
class CortexTransport(Protocol):
    """Sends a question to Cortex Analyst and returns the raw JSON reply or a typed error."""

    def send(self, question: str, semantic_ref: dict[str, str]) -> "dict[str, Any] | SolverError":
        """Send `question` against `semantic_ref`, returning the JSON body or a `SolverError`."""
        ...


_STATUS_KINDS: dict[int, SolverErrorKind] = {
    400: "bad_request",
    401: "auth",
    403: "auth",
    404: "bad_request",
    429: "rate_limit",
}


def _session_token(connection: SnowflakeConnection) -> str:
    """Return `connection`'s live session token for the `Authorization` header.

    Args:
        connection: An open Snowflake connection.

    Returns:
        The connection's current session token.

    Raises:
        RuntimeError: If the connection has no active session token.
    """
    rest = connection.rest
    token = rest.token if rest is not None else None
    if token is None:
        msg = "the Snowflake connection has no active session token"
        raise RuntimeError(msg)
    return token


def _http_error(response: requests.Response) -> SolverError:
    """Translate a non-200 Cortex Analyst response into a `SolverError`.

    Args:
        response: The HTTP response whose status is not 200.

    Returns:
        A `SolverError` whose kind reflects the status code, carrying the response's error
        message when the body is JSON, else a truncated body.
    """
    kind: SolverErrorKind = _STATUS_KINDS.get(response.status_code, "api_error")
    try:
        message = response.json().get("message") or response.text[:500]
    except ValueError:
        message = response.text[:500] or f"HTTP {response.status_code}"
    return SolverError(kind=kind, message=message, provider="cortex_analyst")


class CortexAnalystClient:
    """Sends a question to the Cortex Analyst REST endpoint and returns the raw JSON reply."""

    def __init__(
        self,
        *,
        host: str,
        token_provider: Callable[[], str],
        session: requests.Session | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Configure the client.

        Args:
            host: The Snowflake account host, e.g. `"myacct.snowflakecomputing.com"`.
            token_provider: Returns the session token to place in the `Authorization` header,
                read fresh on each call.
            session: The `requests.Session` to send through, or `None` to create one.
            timeout: Per-request timeout in seconds.
        """
        self._host = host
        self._token_provider = token_provider
        self._session = session if session is not None else requests.Session()
        self._timeout = timeout

    @classmethod
    def from_connection(cls, connection: SnowflakeConnection, *, timeout: float = 60.0) -> "CortexAnalystClient":
        """Build a client that authenticates with `connection`'s live session token.

        Args:
            connection: An open Snowflake connection whose host and session token are reused.
            timeout: Per-request timeout in seconds.

        Returns:
            A `CortexAnalystClient` bound to `connection`'s host and token.
        """
        return cls(host=connection.host, token_provider=lambda: _session_token(connection), timeout=timeout)

    def send(self, question: str, semantic_ref: dict[str, str]) -> "dict[str, Any] | SolverError":
        """POST `question` against `semantic_ref`, returning the JSON body or a typed error.

        Args:
            question: The natural-language question to answer.
            semantic_ref: The semantic-model reference field, e.g. `{"semantic_view": "DB.SCH.SV"}`.

        Returns:
            The decoded JSON response body, or a `SolverError` on an expected transport or HTTP
            failure.
        """
        url = f"https://{self._host}{_ENDPOINT}"
        headers = {
            "Authorization": f'Snowflake Token="{self._token_provider()}"',
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": question}]}],
            "stream": False,
            **semantic_ref,
        }
        try:
            response = self._session.post(url, headers=headers, json=body, timeout=self._timeout)
        except requests.Timeout as e:
            return SolverError(
                kind="timeout", message=str(e) or "request timed out", provider="cortex_analyst", cause=e
            )
        except requests.ConnectionError as e:
            return SolverError(
                kind="api_connection", message=str(e) or "connection error", provider="cortex_analyst", cause=e
            )
        except requests.RequestException as e:
            return SolverError(kind="api_error", message=str(e) or type(e).__name__, provider="cortex_analyst", cause=e)
        if response.status_code != 200:
            return _http_error(response)
        return response.json()
