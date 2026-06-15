"""Tests for the dataeval pytest plugin's `case` fixture, exercised via `pytester`."""

import json
import types
from pathlib import Path

import pytest

from dataeval.pytest_plugin import plugin

pytest_plugins = ["pytester"]


@pytest.mark.unit
def test_case_fixture_injects_the_decorated_case(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        from dataeval import eval_case
        from dataeval.platforms import duckdb_platform

        @eval_case(
            input="q",
            expected={"rows": [{"n": 1}]},
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

    from dataeval import CallableSolver, ResultSetEquivalence, assert_eval, eval_case
    from dataeval.platforms import duckdb_platform

    _DB = Path(__DIR__) / "t.duckdb"
    _con = duckdb.connect(str(_DB))
    _con.execute("CREATE TABLE t (genre VARCHAR)")
    _con.execute("INSERT INTO t VALUES ('Rock'), ('Rock')")
    _con.close()

    duck = duckdb_platform(name="p", path=str(_DB))

    @eval_case(input="q", expected={"rows": [{"c": 2}]}, platform=duck)
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
    result.stdout.fnmatch_lines(["*dataeval summary*", "*test_eval*PASS*", "*1 passed, 0 failed*"])


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
def test_sessionfinish_on_xdist_worker_ships_results_not_json(tmp_path: Path) -> None:
    from dataeval.reporting.collector import CaseReport, clear, record

    clear()
    record(CaseReport(id="w1", input="q", passed=True))
    artifact = tmp_path / "results.json"
    workeroutput: dict[str, object] = {}
    config = types.SimpleNamespace(workerinput={}, workeroutput=workeroutput, getoption=lambda name: str(artifact))
    session = types.SimpleNamespace(config=config)
    plugin.pytest_sessionfinish(session, 0)  # ty: ignore[invalid-argument-type]
    clear()
    assert not artifact.exists()  # the controller writes the artifact, never a worker
    shipped = workeroutput[plugin._WORKEROUTPUT_KEY]
    assert [c["id"] for c in shipped] == ["w1"]  # ty: ignore[non-subscriptable]


@pytest.mark.unit
def test_testnodedown_merges_worker_cases_into_controller() -> None:
    from dataeval.reporting.collector import clear, reports

    clear()
    node = types.SimpleNamespace(workeroutput={plugin._WORKEROUTPUT_KEY: [{"id": "w1", "input": "q", "passed": True}]})
    plugin.pytest_testnodedown(node, None)
    assert [r.id for r in reports()] == ["w1"]
    clear()


@pytest.mark.unit
def test_testnodedown_without_worker_output_is_a_noop() -> None:
    from dataeval.reporting.collector import clear, reports

    clear()
    plugin.pytest_testnodedown(types.SimpleNamespace(), None)
    assert reports() == []


@pytest.mark.unit
def test_json_artifact_written_when_flag_set(pytester: pytest.Pytester) -> None:
    _make_eval_test(pytester)
    artifact = pytester.path / "results.json"
    result = pytester.runpytest(f"--dataeval-json={artifact}")
    result.assert_outcomes(passed=1)
    assert artifact.exists()
    payload = json.loads(artifact.read_text())
    assert payload["passed"] == 1
    assert payload["failed"] == 0
    assert payload["cases"][0]["id"] == "test_eval"


_XDIST_EVAL_TEST = """
    from pathlib import Path

    from dataeval import CallableSolver, ResultSetEquivalence, assert_eval, eval_case
    from dataeval.platforms import duckdb_platform

    duck = duckdb_platform(name="p", path=str(Path(__DIR__) / "t.duckdb"))

    @eval_case(input="q", expected={"rows": [{"c": 2}]}, platform=duck)
    def test_eval(case):
        solver = CallableSolver(lambda c: "SELECT count(*) AS c FROM t WHERE genre = 'Rock'")
        assert_eval(case, solver, scorers=[ResultSetEquivalence()])
"""


@pytest.mark.unit
def test_xdist_aggregates_worker_results_into_summary_and_json(pytester: pytest.Pytester) -> None:
    import duckdb

    db = pytester.path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE t (genre VARCHAR)")
    con.execute("INSERT INTO t VALUES ('Rock'), ('Rock')")
    con.close()
    pytester.makepyfile(_XDIST_EVAL_TEST.replace("__DIR__", repr(str(pytester.path))))

    artifact = pytester.path / "results.json"
    result = pytester.runpytest_subprocess("-n", "2", f"--dataeval-json={artifact}")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*dataeval summary*", "*test_eval*PASS*", "*1 passed, 0 failed*"])
    payload = json.loads(artifact.read_text())
    assert payload["passed"] == 1
    assert payload["failed"] == 0
    assert payload["cases"][0]["id"] == "test_eval"
