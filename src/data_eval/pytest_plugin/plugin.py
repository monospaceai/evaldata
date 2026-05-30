"""The data-eval pytest plugin: the ``case`` fixture, run summary, and JSON artifact.

Loaded automatically via the ``pytest11`` entry point in ``pyproject.toml`` — so
``pytest tests/`` "just works" with zero conftest ceremony (design principle 4). The plugin
stays side-effect-free for projects that merely have data-eval installed: the ``case``
fixture is active only for tests that request it, the run summary prints only when at least
one case ran, and the JSON artifact is written only when ``--data-eval-json`` is passed.

Reporting reads the run accumulator in ``reporting.collector`` (populated by ``assert_eval``):
a Rich rollup table in ``pytest_terminal_summary`` and a structured JSON artifact in
``pytest_sessionfinish``. CI pass/fail comes from pytest's native ``--junitxml`` for free —
each failing ``assert_eval`` is an ordinary test failure, and its Rich diff lands in the
``<failure>`` body — so the plugin emits no JUnit XML of its own.

Under ``pytest-xdist`` the accumulator is process-local, so reporting runs only on the
controller (workers are skipped); aggregation across distributed workers is not yet wired.
"""

from pathlib import Path

import pytest

from data_eval.loaders.python import read_eval_case
from data_eval.platforms.registry import close_all
from data_eval.reporting.collector import reports, run_report_json
from data_eval.reporting.terminal import render_summary
from data_eval.types import EvalCase

_JSON_OPTION = "--data-eval-json"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--data-eval-json=PATH`` to write the structured results artifact."""
    group = parser.getgroup("data-eval")
    group.addoption(
        _JSON_OPTION,
        action="store",
        default=None,
        metavar="PATH",
        help="Write a structured data-eval results JSON artifact to PATH.",
    )


@pytest.fixture
def case(request: pytest.FixtureRequest) -> EvalCase:
    """Inject the ``EvalCase`` attached by ``@eval_case`` on the requesting test function."""
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
    """Print the data-eval run summary table (controller only; skipped when no case ran)."""
    if hasattr(config, "workerinput"):  # xdist worker — the controller reports
        return
    case_reports = reports()
    if not case_reports:
        return
    terminalreporter.write_sep("=", "data-eval summary")
    for line in render_summary(case_reports).splitlines():
        terminalreporter.write_line(line)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Close resolved adapters and, if requested, write the JSON results artifact."""
    close_all()
    if hasattr(session.config, "workerinput"):  # xdist worker — the controller writes
        return
    path = session.config.getoption(_JSON_OPTION)
    if path is not None:
        Path(path).write_text(run_report_json(reports()))
