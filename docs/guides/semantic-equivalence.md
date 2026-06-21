# Check semantic equivalence

A correct answer can be written many ways. `SELECT a, b` and `SELECT b, a` reordered, a
predicate flipped, `amount + 1` written as `1 + amount` — all return the same result, but a
text comparison rejects them. `SemanticEquivalence` accepts them: it asks whether the AI's
query is *semantically equivalent* to a gold query — the same result, however the SQL is
written — not whether the SQL strings match.

## How it decides

`SemanticEquivalence` runs an ordered list of checks and stops at the first one that reaches a
decision. Each check returns one of three verdicts: `equivalent`, `not_equivalent`, or
`unknown` (it could not decide). The default order is cheapest first:

1. **`AstEquivalence`** — parses both queries, normalizes their syntax trees conservatively
   (preserving meaning), then compares them. Matching trees mean the queries are equivalent,
   decided without touching the warehouse. A difference is inconclusive, not a refutation, so
   it returns `unknown` and the next check runs. Because the normalization is conservative,
   some true equivalences (for example commutative arithmetic over columns, `x + 1` versus
   `1 + x`) are not recognized here.

2. **`ExecutionEquivalence`** — runs both queries and diffs the result sets in the warehouse,
   honoring the case's `ComparisonConfig` (row order, NULL handling, float tolerance). Equal
   result sets pass; a difference fails and carries the diff.

When the trees match, no query runs. When they don't, execution decides. If no check can
decide, the result fails.

The expected outcome must be a [`GoldQuery`][evaldata.types.GoldQuery]: equivalence is a
query-against-query comparison, so there must be a reference query to compare against.

## Write the eval

```python
--8<-- "examples/01_deterministic/test_semantic_equivalence.py"
```

The first case is confirmed by `AstEquivalence` alone — the AI reorders the `AND` predicates
and changes casing, the normalized trees match, and no warehouse query runs. The second case
uses `1 + amount` where the gold uses `amount + 1`; the syntax check abstains and
`ExecutionEquivalence` confirms the result sets are identical.

## Run it

```bash
uv run pytest test_semantic_equivalence.py -q
```

!!! tip "Run it from a clone"
    This is the bundled `examples/01_deterministic/test_semantic_equivalence.py` example. If
    you've cloned the repo, run it directly with
    `uv run pytest examples/01_deterministic/test_semantic_equivalence.py`.

## Choose the checks

Pass your own checks to control the order and which ones run:

```python
from evaldata import SemanticEquivalence
from evaldata.scorers import AstEquivalence, ExecutionEquivalence

# Execution only — always run the queries, never trust the syntax check.
SemanticEquivalence([ExecutionEquivalence()])

# Syntax only — decide without touching the warehouse (fails when it cannot decide).
SemanticEquivalence([AstEquivalence()])
```

Omitting the argument uses the default `[AstEquivalence(), ExecutionEquivalence()]`.

## Next steps

- [Concepts](../concepts.md) — scorers, solvers, and expected-types in depth.
- [Scorers reference](../reference/scorers.md) — the scorer and check API.
