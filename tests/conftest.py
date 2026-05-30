"""Shared pytest fixtures for data-eval's own test suite."""

from collections.abc import Iterator

import pytest

from data_eval.platforms.registry import close_all
from data_eval.reporting.collector import clear


@pytest.fixture(autouse=True)
def _reset_global_state() -> Iterator[None]:
    """Reset module-level session state after each test, isolating tests from each other.

    In production the plugin closes the adapter cache at session end and the run
    accumulator lives for the process; for our own suite we clear both per test so cached
    adapters (and ``PlatformRef.name`` bindings) and recorded case outcomes never leak
    across tests — and the data-eval run summary stays out of our own pytest output.
    """
    yield
    close_all()
    clear()
