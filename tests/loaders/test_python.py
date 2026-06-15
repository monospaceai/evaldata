"""Unit tests for the `@eval_case` decorator and its dict->Expected coercion."""

import pytest
from pydantic import ValidationError

from dataeval.loaders.python import eval_case, read_eval_case
from dataeval.platforms import duckdb_platform
from dataeval.types import ComparisonConfig, CostBudget, GoldQuery, TypedResultSet, UntypedResultSet

_PLATFORM = duckdb_platform(name="local")


@pytest.mark.unit
class TestEvalCaseDecorator:
    def test_id_defaults_to_function_name(self) -> None:
        @eval_case(input="q", expected={"rows": [{"n": 1}]}, platform=_PLATFORM)
        def test_thing(case: object) -> None: ...

        recorded = read_eval_case(test_thing)
        assert recorded is not None
        assert recorded.id == "test_thing"

    def test_explicit_id_overrides_function_name(self) -> None:
        @eval_case(input="q", expected={"rows": [{"n": 1}]}, platform=_PLATFORM, id="custom")
        def test_thing(case: object) -> None: ...

        recorded = read_eval_case(test_thing)
        assert recorded is not None
        assert recorded.id == "custom"

    def test_dict_expected_is_coerced_to_typed_model(self) -> None:
        @eval_case(input="q", expected={"rows": [{"n": 1}]}, platform=_PLATFORM)
        def test_thing(case: object) -> None: ...

        recorded = read_eval_case(test_thing)
        assert recorded is not None
        assert isinstance(recorded.expected, UntypedResultSet)
        assert recorded.expected.rows == [{"n": 1}]

    def test_dict_with_schema_is_coerced_to_typed_result_set(self) -> None:
        @eval_case(
            input="q",
            expected={"rows": [{"n": 1}], "schema": [{"name": "n", "type": "INTEGER"}]},
            platform=_PLATFORM,
        )
        def test_thing(case: object) -> None: ...

        recorded = read_eval_case(test_thing)
        assert recorded is not None
        assert isinstance(recorded.expected, TypedResultSet)
        assert recorded.expected.schema_.names == ["n"]

    def test_gold_query_dict_is_coerced(self) -> None:
        @eval_case(
            input="q",
            expected={"kind": "gold_query", "sql": "SELECT 1"},
            platform=_PLATFORM,
        )
        def test_thing(case: object) -> None: ...

        recorded = read_eval_case(test_thing)
        assert recorded is not None
        assert isinstance(recorded.expected, GoldQuery)
        assert recorded.expected.sql == "SELECT 1"

    def test_typed_expected_passes_through(self) -> None:
        expected = UntypedResultSet(rows=[{"n": 1}])

        @eval_case(input="q", expected=expected, platform=_PLATFORM)
        def test_thing(case: object) -> None: ...

        recorded = read_eval_case(test_thing)
        assert recorded is not None
        assert recorded.expected == expected

    def test_metadata_and_comparison_are_forwarded(self) -> None:
        comparison = ComparisonConfig(float_tolerance=0.5)

        @eval_case(
            input="q",
            expected={"rows": [{"n": 1}]},
            platform=_PLATFORM,
            metadata={"owner": "alice"},
            comparison=comparison,
        )
        def test_thing(case: object) -> None: ...

        recorded = read_eval_case(test_thing)
        assert recorded is not None
        assert recorded.metadata == {"owner": "alice"}
        assert recorded.comparison.float_tolerance == 0.5

    def test_cost_budget_is_forwarded(self) -> None:
        @eval_case(
            input="q",
            expected={"rows": [{"n": 1}]},
            platform=_PLATFORM,
            cost_budget=CostBudget(max_seconds=2.5),
        )
        def test_thing(case: object) -> None: ...

        recorded = read_eval_case(test_thing)
        assert recorded is not None
        assert recorded.cost_budget == CostBudget(max_seconds=2.5)

    def test_malformed_dict_raises_at_decoration_time(self) -> None:
        # An unknown discriminator value fails loudly when the module is imported/collected,
        # not lazily at test-run time.
        with pytest.raises(ValidationError):
            eval_case(input="q", expected={"kind": "not_a_real_kind"}, platform=_PLATFORM)

    def test_decorator_returns_the_function_unchanged(self) -> None:
        def original(case: object) -> None: ...

        decorated = eval_case(input="q", expected={"rows": [{"n": 1}]}, platform=_PLATFORM)(original)
        assert decorated is original

    def test_read_eval_case_returns_none_for_undecorated_function(self) -> None:
        def plain(case: object) -> None: ...

        assert read_eval_case(plain) is None
