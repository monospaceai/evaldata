"""Completeness guard: every scorer-conformance suite must exercise every registered adapter.

`test_conformance_equivalence.py` and `test_conformance_pushdown.py` each parametrise an
`engine` fixture over the platform adapters they run against. A hardcoded, suite-local adapter
list can silently omit an adapter (as happened for SQLite and Snowflake before both suites were
switched to share `conftest.engine_params`); this module fails loudly instead, with no live
connection needed, whenever a suite's coverage drifts from `PlatformKind`.
"""

from typing import Any, get_args

import pytest

from evaldata.types import PlatformKind

from . import test_conformance_equivalence, test_conformance_pushdown
from .conftest import ADAPTER_IDS


def _engine_fixture_ids(suite: Any) -> set[str]:
    """The `id`s of every `pytest.param` the suite's `engine` fixture is parametrised over."""
    marker = suite.engine._fixture_function_marker
    return {param.id for param in marker.params}


@pytest.mark.unit
def test_adapter_registry_covers_every_platform_kind() -> None:
    """`ADAPTER_SPECS` (shared by `under_test` and every `engine` fixture) lists every `PlatformKind`."""
    assert set(get_args(PlatformKind)) == ADAPTER_IDS


@pytest.mark.unit
@pytest.mark.parametrize(
    "suite", [test_conformance_equivalence, test_conformance_pushdown], ids=["equivalence", "pushdown"]
)
def test_scorer_conformance_suite_covers_every_platform_kind(suite: Any) -> None:
    """Each scorer-conformance suite's `engine` fixture must cover every `PlatformKind`, not a subset."""
    assert _engine_fixture_ids(suite) == set(get_args(PlatformKind))
