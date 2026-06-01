"""Tests for the data-eval pytest plugin's `case` fixture, exercised via `pytester`."""

import json
import types
from pathlib import Path

import pytest

from data_eval.pytest_plugin import plugin

pytest_plugins = ["pytester"]


@pytest.mark.unit
def test_case_fixture_injects_the_decorated_case(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        from data_eval import eval_case
        from data_eval.platforms import duckdb_platform

        @eval_case(
            input="q",
            expected={"kind": "result_set", "rows": [{"n": 1}]},
            platform=duckdb_platform(name="p"),
        )
        def test_injected(case):
            assert case.id == "test_injected"
            assert case.input == "q"
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


@pytest.mark.unit
def test_case_fixture_without_decorator_errors_clearly(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        def test_missing_decorator(case):
            assert case is not None
        """
    )
    result = pytester.runpytest()
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*not*decorated with @eval_case*"])


_EVAL_TEST = """
    from pathlib import Path

    import duckdb

    from data_eval import CallableSolver, ResultSetEquivalence, assert_eval, eval_case
    from data_eval.platforms import duckdb_platform

    _DB = Path(__DIR__) / "t.duckdb"
    _con = duckdb.connect(str(_DB))
    _con.execute("CREATE TABLE t (genre VARCHAR)")
    _con.execute("INSERT INTO t VALUES ('Rock'), ('Rock')")
    _con.close()

    duck = duckdb_platform(name="p", path=str(_DB))

    @eval_case(input="q", expected={"kind": "result_set", "rows": [{"c": 2}]}, platform=duck)
    def test_eval(case):
        solver = CallableSolver(lambda c: "SELECT count(*) AS c FROM t WHERE genre = 'Rock'")
        assert_eval(case, solver, scorers=[ResultSetEquivalence()])
"""


def _make_eval_test(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(_EVAL_TEST.replace("__DIR__", repr(str(pytester.path))))


@pytest.mark.unit
def test_run_summary_printed_at_end_of_session(pytester: pytest.Pytester) -> None:
    _make_eval_test(pytester)
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*data-eval summary*", "*test_eval*PASS*", "*1 passed, 0 failed*"])


@pytest.mark.unit
def test_terminal_summary_skipped_on_xdist_worker() -> None:
    config = types.SimpleNamespace(workerinput={})
    calls: list[str] = []
    reporter = types.SimpleNamespace(
        write_sep=lambda *a, **k: calls.append("sep"),
        write_line=lambda *a, **k: calls.append("line"),
    )
    plugin.pytest_terminal_summary(reporter, 0, config)  # ty: ignore[invalid-argument-type]
    assert calls == []  # the controller prints the summary, never a worker


@pytest.mark.unit
def test_sessionfinish_on_xdist_worker_does_not_write_json(tmp_path: Path) -> None:
    artifact = tmp_path / "results.json"
    config = types.SimpleNamespace(workerinput={}, getoption=lambda name: str(artifact))
    session = types.SimpleNamespace(config=config)
    plugin.pytest_sessionfinish(session, 0)  # ty: ignore[invalid-argument-type]
    assert not artifact.exists()  # the controller writes the artifact, never a worker


@pytest.mark.unit
def test_json_artifact_written_when_flag_set(pytester: pytest.Pytester) -> None:
    _make_eval_test(pytester)
    artifact = pytester.path / "results.json"
    result = pytester.runpytest(f"--data-eval-json={artifact}")
    result.assert_outcomes(passed=1)
    assert artifact.exists()
    payload = json.loads(artifact.read_text())
    assert payload["passed"] == 1
    assert payload["failed"] == 0
    assert payload["cases"][0]["id"] == "test_eval"
