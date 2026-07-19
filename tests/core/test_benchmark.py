"""Tests for `run_benchmark` — the non-raising aggregate over a set of cases."""

import sqlite3
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from evaldata import CallableSolver, EvalCase, ExecutionAccuracy, run_benchmark
from evaldata.core.runner import CaseEvaluation, evaluate_case
from evaldata.platforms import duckdb_platform, sqlite_platform
from evaldata.platforms.registry import acquired, close_all, resolve
from evaldata.types import GoldQuery, PlatformRef, SolverError, SolverOutput, Sql


@pytest.fixture
def db(tmp_path: Path) -> Iterator[str]:
    path = tmp_path / "bench.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE items (id INTEGER)")
    con.executemany("INSERT INTO items VALUES (?)", [(1,), (2,), (3,)])
    con.commit()
    con.close()
    yield str(path)
    close_all()  # drop the per-name adapters this test resolved so names are free again


def _case(case_id: str, db: str) -> EvalCase:
    return EvalCase(
        id=case_id,
        input="q",
        expected=GoldQuery(sql="SELECT id FROM items"),
        platform=sqlite_platform(name=f"bench-{case_id}", path=db),
    )


@pytest.mark.unit
class TestRunBenchmark:
    def test_reports_accuracy(self, db: str) -> None:
        # The "good" case returns all rows; the "bad" case returns a subset and fails.
        solver = CallableSolver(
            lambda c: "SELECT id FROM items" if c.id == "good" else "SELECT id FROM items WHERE id < 2"
        )
        summary = run_benchmark([_case("good", db), _case("bad", db)], solver, scorers=[ExecutionAccuracy()])

        assert summary.total == 2
        assert summary.passed == 1
        assert summary.accuracy == 0.5
        assert [c.id for c in summary.cases] == ["good", "bad"]

    def test_limit_caps_cases_run(self, db: str) -> None:
        solver = CallableSolver(lambda c: "SELECT id FROM items")
        cases = [_case("a", db), _case("b", db), _case("c", db)]
        summary = run_benchmark(cases, solver, scorers=[ExecutionAccuracy()], limit=1)

        assert summary.total == 1
        assert summary.passed == 1

    def test_empty_cases_yield_zero_accuracy(self) -> None:
        solver = CallableSolver(lambda c: "SELECT 1")
        summary = run_benchmark([], solver, scorers=[ExecutionAccuracy()])

        assert summary.total == 0
        assert summary.passed == 0
        assert summary.accuracy == 0.0

    def test_concurrency_preserves_case_order(self, db: str) -> None:
        # Earlier cases sleep longest, so the solvers finish in reverse of submission order;
        # the reports must still come back in case order.
        def solve(case: EvalCase) -> str:
            time.sleep(0.02 * (3 - int(case.id)))
            return "SELECT id FROM items"

        cases = [_case(str(i), db) for i in range(3)]
        summary = run_benchmark(cases, CallableSolver(solve), scorers=[ExecutionAccuracy()], max_concurrency=4)

        assert [c.id for c in summary.cases] == ["0", "1", "2"]
        assert summary.passed == 3

    def test_concurrency_case_failure_does_not_kill_run(self, db: str) -> None:
        class _Solver:
            def solve(self, case: EvalCase) -> SolverOutput:
                if case.id == "b":
                    return SolverOutput(error=SolverError(kind="bad_request", message="boom"))
                return SolverOutput(output=Sql("SELECT id FROM items"))

        cases = [_case("a", db), _case("b", db), _case("c", db)]
        summary = run_benchmark(cases, _Solver(), scorers=[ExecutionAccuracy()], max_concurrency=3)

        assert summary.total == 3
        assert summary.passed == 2
        reports = {r.id: r for r in summary.cases}
        assert reports["b"].error is not None
        assert reports["a"].passed and reports["c"].passed


def _duck_case(platform: PlatformRef, grp: int, *, correct: bool) -> EvalCase:
    """A case whose gold counts rows in group `grp`; a `correct` solver matches it, else undercounts."""
    return EvalCase(
        id=f"grp-{grp}-{'ok' if correct else 'bad'}",
        input=str(grp),
        expected=GoldQuery(sql=f"SELECT count(*) AS c FROM items WHERE grp = {grp}"),
        platform=platform,
    )


def _duck_solver() -> CallableSolver:
    """Solve each case from its own gold, so a correct case matches and a `bad` case undercounts."""

    def solve(case: EvalCase) -> str:
        grp = int(case.input)
        if case.id.endswith("-ok"):
            return f"SELECT count(*) AS c FROM items WHERE grp = {grp}"
        return f"SELECT count(*) AS c FROM items WHERE grp = {grp} AND id < 0"

    return CallableSolver(solve)


@pytest.mark.unit
class TestDuckDBConcurrency:
    """Concurrency correctness over a shared-cursor DuckDB pool: results stay per-case.

    Cases run through concurrent `evaluate_case` calls (each acquiring its own pool session),
    which is the path that genuinely scores in parallel.
    """

    @staticmethod
    def _seed(platform: PlatformRef) -> None:
        resolve(platform).execute(
            "CREATE TABLE items (id INTEGER, grp INTEGER); "
            "INSERT INTO items SELECT i AS id, i % 4 AS grp FROM range(0, 200) t(i)"
        )

    @staticmethod
    def _run_concurrently(cases: list[EvalCase], workers: int) -> list[CaseEvaluation]:
        solver = _duck_solver()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # A fresh scorer per case keeps the test about connection concurrency, not scorers.
            return list(pool.map(lambda c: evaluate_case(c, solver, scorers=[ExecutionAccuracy()]), cases))

    def test_many_cases_score_independently_in_memory(self) -> None:
        platform = duckdb_platform(name="duck-conc-mem")
        try:
            self._seed(platform)
            # 40 cases (10 per group) over an 8-member pool: correct ones must pass, the
            # undercounting ones must fail, with no cross-talk between concurrent cursors.
            cases = [_duck_case(platform, g, correct=(i % 2 == 0)) for i in range(10) for g in range(4)]
            evaluations = self._run_concurrently(cases, workers=8)
            assert len(evaluations) == 40
            assert all(e.report.passed == e.report.id.endswith("-ok") for e in evaluations)
        finally:
            close_all()

    def test_many_cases_score_independently_file_backed(self, tmp_path: Path) -> None:
        platform = duckdb_platform(name="duck-conc-file", path=str(tmp_path / "conc.duckdb"))
        try:
            self._seed(platform)
            cases = [_duck_case(platform, g, correct=True) for _ in range(6) for g in range(4)]
            evaluations = self._run_concurrently(cases, workers=6)
            assert len(evaluations) == 24
            assert all(e.report.passed for e in evaluations)
        finally:
            close_all()

    def test_cases_queue_when_concurrency_exceeds_pool_size(self) -> None:
        # More concurrent workers (16) than the DuckDB pool's 8 members forces excess workers to
        # block on acquire; all must still complete correctly once members free up.
        platform = duckdb_platform(name="duck-conc-queue")
        try:
            self._seed(platform)
            cases = [_duck_case(platform, g, correct=True) for _ in range(4) for g in range(4)]
            evaluations = self._run_concurrently(cases, workers=16)
            assert len(evaluations) == 16
            assert all(e.report.passed for e in evaluations)
        finally:
            close_all()

    def test_acquire_blocks_until_release_under_a_bounded_pool(self) -> None:
        # Deterministically exercise the pool's blocking path through the registry: hold every
        # member via a barrier, then confirm a further acquire only proceeds after a release.
        from evaldata.platforms.registry import pool_for

        platform = duckdb_platform(name="duck-block")
        pool = pool_for(platform)
        held = [pool.acquire() for _ in range(8)]  # exhaust the default 8-member pool
        proceeded = threading.Event()

        def waiter() -> None:
            with acquired(platform):
                proceeded.set()

        t = threading.Thread(target=waiter)
        t.start()
        assert not proceeded.wait(timeout=0.1)  # blocked: no member free
        pool.release(held.pop())
        assert proceeded.wait(timeout=2)  # a release unblocks the waiter
        t.join(timeout=2)
        for member in held:
            pool.release(member)
