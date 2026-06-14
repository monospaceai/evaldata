"""Mocked model replies for the hosted-AI example, so it runs without a key or network."""

import json
from typing import Any

import litellm
import pytest

_SQL_BY_QUESTION = {
    "order_count": "SELECT count(*) AS order_count FROM orders",
    "customer_count": "SELECT count(DISTINCT customer_id) AS customer_count FROM orders",
}


def _mock_sql(messages: list[dict[str, Any]]) -> str:
    """Pick the correct SQL for a request by matching its question text.

    Args:
        messages: The chat messages of the request, whose prompt names the expected
            output column.

    Returns:
        The structured JSON reply for the matched question.

    Raises:
        AssertionError: When no known question is present in the messages.
    """
    prompt = " ".join(m.get("content", "") for m in messages)
    for marker, sql in _SQL_BY_QUESTION.items():
        if marker in prompt:
            return json.dumps({"sql": sql})
    msg = f"no mock SQL for prompt: {prompt!r}"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover


@pytest.fixture(autouse=True)
def _mock_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch model calls to return a deterministic reply per question, with no network."""
    real_completion = litellm.completion

    def fake(**kwargs: Any) -> Any:
        return real_completion(**kwargs, mock_response=_mock_sql(kwargs["messages"]))

    monkeypatch.setattr("litellm.completion", fake)
