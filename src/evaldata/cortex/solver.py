"""`CortexAnalystSolver`: a `Solver` that answers questions with Snowflake Cortex Analyst."""

from typing import Any

from evaldata.cortex.client import CortexTransport
from evaldata.types import EvalCase, SolverError, SolverOutput, Sql


def _metadata(response: dict[str, Any]) -> dict[str, Any]:
    """Collect the non-answer telemetry Cortex Analyst returns.

    Args:
        response: The decoded Cortex Analyst response body.

    Returns:
        A metadata mapping with `request_id` and `model_names` when present.
    """
    metadata: dict[str, Any] = {}
    if request_id := response.get("request_id"):
        metadata["request_id"] = request_id
    if model_names := response.get("response_metadata", {}).get("model_names"):
        metadata["model_names"] = model_names
    return metadata


def _to_output(response: dict[str, Any]) -> SolverOutput:
    """Turn a Cortex Analyst response body into a `SolverOutput`.

    Args:
        response: The decoded Cortex Analyst response body.

    Returns:
        A `SolverOutput` carrying the generated SQL, or a `SolverError` of kind `empty_response`
        when Cortex Analyst returned suggestions (or nothing) instead of SQL.
    """
    content = response.get("message", {}).get("content", [])
    metadata = _metadata(response)
    for item in content:
        if item.get("type") == "sql" and item.get("statement", "").strip():
            return SolverOutput(output=Sql(item["statement"]), metadata=metadata)
    suggestions = [s for item in content if item.get("type") == "suggestions" for s in item.get("suggestions", [])]
    message = (
        "Cortex Analyst returned suggestions instead of SQL: " + "; ".join(suggestions)
        if suggestions
        else "Cortex Analyst returned no SQL"
    )
    return SolverOutput(
        error=SolverError(kind="empty_response", message=message, provider="cortex_analyst"), metadata=metadata
    )


class CortexAnalystSolver:
    """A `Solver` that sends a question to Cortex Analyst and returns the SQL it generates."""

    def __init__(
        self,
        client: CortexTransport,
        *,
        semantic_view: str | None = None,
        semantic_model_file: str | None = None,
    ) -> None:
        """Configure the solver.

        Args:
            client: The Cortex Analyst client to send questions through.
            semantic_view: A fully qualified semantic view, e.g. `"DB.SCHEMA.VIEW"`.
            semantic_model_file: A stage path to a semantic-model YAML, e.g.
                `"@db.schema.stage/model.yaml"`.

        Raises:
            ValueError: If not exactly one of `semantic_view`/`semantic_model_file` is given.
        """
        refs = {"semantic_view": semantic_view, "semantic_model_file": semantic_model_file}
        chosen = {key: value for key, value in refs.items() if value is not None}
        if len(chosen) != 1:
            msg = "CortexAnalystSolver requires exactly one of 'semantic_view' or 'semantic_model_file'"
            raise ValueError(msg)
        self._client = client
        self._semantic_ref = chosen

    def solve(self, case: EvalCase) -> SolverOutput:
        """Produce SQL for `case` by asking Cortex Analyst its `input` question.

        Returns:
            A `SolverOutput` carrying the generated SQL and telemetry, or a typed `SolverError`
            on an expected transport, HTTP, or no-SQL failure.
        """
        response = self._client.send(case.input, self._semantic_ref)
        if isinstance(response, SolverError):
            return SolverOutput(error=response)
        return _to_output(response)
