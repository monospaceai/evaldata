"""`@eval_case`: the Python authoring decorator for test cases."""

from collections.abc import Callable
from typing import Any, TypeVar
from weakref import WeakKeyDictionary

from pydantic import TypeAdapter

from dataeval.types import ComparisonConfig, CostBudget, EvalCase, Expected, PlatformRef

_TestFn = TypeVar("_TestFn", bound=Callable[..., Any])

# Built once: constructing a TypeAdapter compiles a core schema. Validates a dict into the
# discriminated `Expected` union (dispatch on "kind").
_EXPECTED_ADAPTER: TypeAdapter[Expected] = TypeAdapter(Expected)

# Function object -> its EvalCase. Weak keys so a collected test function that goes away
# takes its entry with it; identity lookup matches what pytest passes as `request.function`.
_CASES: WeakKeyDictionary[Callable[..., Any], EvalCase] = WeakKeyDictionary()


def eval_case(
    *,
    input: str,
    expected: dict[str, Any] | Expected,
    platform: PlatformRef,
    id: str | None = None,
    metadata: dict[str, Any] | None = None,
    comparison: ComparisonConfig | None = None,
    cost_budget: CostBudget | None = None,
) -> Callable[[_TestFn], _TestFn]:
    """Attach an `EvalCase` to a test function for the `case` fixture to inject.

    Args:
        input: The natural-language question / instruction under test.
        expected: The expected outcome — a typed `Expected` or a dict coerced to one.
        platform: A `PlatformRef` (build one with `duckdb_platform` / `postgres_platform`).
        id: Case identifier; defaults to the decorated function's name.
        metadata: Optional free-form tags/owner/source metadata.
        comparison: Optional result-set comparison rules; defaults to `ComparisonConfig()`.
        cost_budget: Optional ceiling on platform resource consumption for the case.

    Returns:
        A decorator that records the case and returns the function unchanged.
    """
    coerced: Expected = _EXPECTED_ADAPTER.validate_python(expected) if isinstance(expected, dict) else expected

    def decorator(func: _TestFn) -> _TestFn:
        extra: dict[str, Any] = {}
        if metadata is not None:
            extra["metadata"] = metadata
        if comparison is not None:
            extra["comparison"] = comparison
        if cost_budget is not None:
            extra["cost_budget"] = cost_budget
        _CASES[func] = EvalCase(
            id=id or getattr(func, "__name__", ""),
            input=input,
            expected=coerced,
            platform=platform,
            **extra,
        )
        return func

    return decorator


def read_eval_case(func: Callable[..., Any]) -> EvalCase | None:
    """Return the `EvalCase` attached to `func` by `@eval_case`, or `None`."""
    return _CASES.get(func)
