"""The dataeval pytest plugin: the `case` fixture, run summary, and JSON artifact."""

from pathlib import Path
from typing import Any

import pytest

from dataeval.loaders.python import read_eval_case
from dataeval.platforms.registry import close_all
from dataeval.reporting.collector import CaseReport, extend, reports, run_report_json
from dataeval.reporting.terminal import render_summary
from dataeval.types import EvalCase

_JSON_OPTION = "--dataeval-json"
_WORKEROUTPUT_KEY = "dataeval_cases"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register `--dataeval-json=PATH` to write the structured results artifact."""
    group = parser.getgroup("dataeval")
    group.addoption(
        _JSON_OPTION,
        action="store",
        default=None,
        metavar="PATH",
        help="Write a structured dataeval results JSON artifact to PATH.",
    )


@pytest.fixture
def case(request: pytest.FixtureRequest) -> EvalCase:
    """Inject the `EvalCase` attached by `@eval_case` on the requesting test function.

    Args:
        request: The pytest fixture request, used to find the requesting test function.

    Returns:
        The `EvalCase` attached to the test by its `@eval_case(...)` decorator.

    Raises:
        UsageError: If the requesting test is not decorated with `@eval_case(...)`.
    """
    evalcase = read_eval_case(request.function)
    if evalcase is None:
        msg = (
            f"test {request.function.__name__!r} requests the 'case' fixture but is not decorated with @eval_case(...)"
        )
        raise pytest.UsageError(msg)
    return evalcase


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Print the dataeval run summary table (controller only; skipped when no case ran)."""
    if hasattr(config, "workerinput"):  # xdist worker — the controller reports
        return
    case_reports = reports()
    if not case_reports:
        return
    terminalreporter.write_sep("=", "dataeval summary")
    for line in render_summary(case_reports).splitlines():
        terminalreporter.write_line(line)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Close adapters; on a worker hand results to the controller, else write the JSON artifact."""
    close_all()
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None:  # xdist worker — ship results to the controller, which writes
        workeroutput[_WORKEROUTPUT_KEY] = [r.model_dump(mode="json") for r in reports()]
        return
    path = session.config.getoption(_JSON_OPTION)
    if path is not None:
        Path(path).write_text(run_report_json(reports()))


def pytest_testnodedown(node: Any, error: object) -> None:
    """Merge a finished xdist worker's serialized case outcomes into the controller's collector.

    Args:
        node: The xdist worker node that shut down; carries `workeroutput`.
        error: The shutdown error, if any (unused).
    """
    serialized = getattr(node, "workeroutput", {}).get(_WORKEROUTPUT_KEY)
    if serialized is not None:
        extend(CaseReport.model_validate(item) for item in serialized)
