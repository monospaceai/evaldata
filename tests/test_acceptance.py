"""End-to-end test of the pytest-native surface against a real DuckDB file."""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from dataeval import CallableSolver, ResultSetEquivalence, assert_eval, eval_case
from dataeval.platforms import duckdb_platform
from dataeval.types import EvalCase

# Resolved at import (decoration) time, before fixtures run; the file is seeded by a
# fixture and opened lazily when the test executes.
_DB_PATH = Path(tempfile.mkdtemp(prefix="dataeval_accept_")) / "chinook.duckdb"
_ROCK_SQL = "SELECT count(*) AS count FROM tracks WHERE genre = 'Rock'"


@pytest.fixture(scope="module", autouse=True)
def _seed_db() -> Iterator[None]:
    con = duckdb.connect(str(_DB_PATH))
    con.execute("CREATE TABLE tracks (id INTEGER, genre VARCHAR)")
    con.execute("INSERT INTO tracks VALUES (1, 'Rock'), (2, 'Rock'), (3, 'Jazz')")
    con.close()
    yield


@pytest.mark.unit
@eval_case(
    input="How many tracks are in the 'Rock' genre?",
    expected={"rows": [{"count": 2}]},
    platform=duckdb_platform(name="acceptance-local", path=str(_DB_PATH)),
)
def test_rock_track_count(case: EvalCase) -> None:
    solver = CallableSolver(lambda c: _ROCK_SQL)
    assert_eval(case, solver, scorers=[ResultSetEquivalence()])
