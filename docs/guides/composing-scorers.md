# Compose and tier scorers

A scorer turns an executed case into a pass/fail verdict. `FirstDecisive` runs scorers in order and
stops at the first verdict. Put lower-cost checks before model or warehouse calls.

## The scorer contract

Every scorer implements a single method:

```python
def score(self, case, output, result, *, context) -> ScoreResult: ...
```

It receives the case, the solver's output, the executed result, and a score context, and returns a
[`ScoreResult`](../reference/types.md). evaldata ships scorers for common jobs:
`ResultSetEquivalence`, `SemanticEquivalence`, `ExpectationSuiteScorer`, and `LlmJudge`. Pass a
list to `assert_eval`:

```python
assert_eval(case, solver, scorers=[observed_equivalence()])
```

`assert_eval` **ANDs** the list: the case passes only when every scorer in it passes. A cascade
stops at the first verdict, so it is a single composed scorer rather than several entries in the
list.

## Verdicts

A [`ScoreResult`](../reference/types.md) has a `verdict` that is one of three values:

- **`pass`**: the scorer decided the answer is correct.
- **`fail`**: the scorer decided the answer is wrong; the result may carry a `diff` or per-check
  `outcomes` explaining why.
- **`inconclusive`**: the scorer could not decide either way (a query that wouldn't parse, a grader
  call that errored). Inconclusive is not a fail; it means this scorer has nothing to say, so a
  cascade moves on to the next member.

Read `.verdict` for the three-way value, or `.passed` for the boolean.

A decided result also carries a `basis`: `proven` (by reasoning about the queries), `observed`
(seen to hold on the data that ran), or `judged` (a grader model's probabilistic call). An
inconclusive result carries no `score` and no `basis`.

## FirstDecisive: a tiered cascade

`FirstDecisive` takes an ordered list of scorers and runs them in order. It stops at the first
member that returns `pass` or `fail` and returns that result; a member that returns `inconclusive`
hands off to the next. If every member is inconclusive, the last member's result stands, so its
diagnostics (such as a row diff) still surface.

```python
from evaldata import FirstDecisive, ResultSetEquivalence, SemanticEquivalence

scorer = FirstDecisive([SemanticEquivalence(), ResultSetEquivalence()])
```

In this cascade, `SemanticEquivalence` checks SQL ASTs first. If it returns `inconclusive`,
`ResultSetEquivalence` executes both queries and compares their rows.

Each run records which members ran and how each one voted, so you can see which layer decided:

```python
result.metadata["first_decisive"]
# [{"scorer": "semantic_equivalence", "passed": False, "verdict": "inconclusive"},
#  {"scorer": "result_set_equivalence", "passed": True, "verdict": "pass"}]
```

## Ready-made equivalence cascades

Two presets wrap the common query-vs-query cascades. Each expects the case's `expected` to be a
[`GoldQuery`][evaldata.types.GoldQuery], since equivalence compares one query against a reference.

`observed_equivalence()` tries semantic equivalence first, else runs both queries and diffs the
results:

```python
from evaldata import observed_equivalence

scorer = observed_equivalence()
# FirstDecisive([SemanticEquivalence(), ResultSetEquivalence()])
```

`judged_equivalence(model)` tries semantic equivalence first, then asks an LLM judge whether the
two queries are equivalent when no warehouse is available:

```python
from evaldata import judged_equivalence

scorer = judged_equivalence("openai/gpt-4o-mini")
# FirstDecisive([SemanticEquivalence(), sql_equivalence_judge(model)])
```

`sql_equivalence_judge(model)` is the judge those cascades reach for: an `LlmJudge` pre-loaded with
SQL-equivalence criteria and few-shot examples. Use it on its own when you want the grader's
judgment directly:

```python
from evaldata import sql_equivalence_judge

judge = sql_equivalence_judge("openai/gpt-4o-mini")
```

## Compose your own cascade

The presets are `FirstDecisive` instances, so you can build your own. This cascade tries semantic
equivalence, execution, then the judge. A case with inconclusive execution can still receive a
judged verdict:

```python
from evaldata import FirstDecisive, ResultSetEquivalence, SemanticEquivalence, sql_equivalence_judge

scorer = FirstDecisive(
    [
        SemanticEquivalence(),                        # proven, runs no query
        ResultSetEquivalence(),                       # observed, runs both queries
        sql_equivalence_judge("openai/gpt-4o-mini"),  # judged, when execution was inconclusive
    ]
)
```

Any object with a `score` method is a valid member.

## Anchor a judge with a rubric

A cascade member can be a fully configured `LlmJudge`. `RubricBand` defines score ranges and their
descriptions:

```python
from evaldata import LlmJudge, RubricBand

judge = LlmJudge(
    model="openai/gpt-4o-mini",
    criteria="The SQL answers the question and is idiomatic for an analytics team.",
    rubric=[
        RubricBand(min_score=0.0, max_score=0.4, description="Wrong result or unreadable."),
        RubricBand(min_score=0.4, max_score=0.8, description="Correct result, rough style."),
        RubricBand(min_score=0.8, max_score=1.0, description="Correct and idiomatic."),
    ],
    threshold=0.8,
)
```

Add `judge` to a `FirstDecisive` list wherever a judged verdict belongs in the cascade.

## Next steps

- [Concepts](../concepts.md): cases, solvers, scorers, and platforms in depth.
- [Check semantic equivalence](semantic-equivalence.md): how AST normalization confirms or
  returns `unknown`, and the comparison options for the execution member.
- [Score with an LLM judge](llm-judge.md): writing criteria, steps, examples, and rubrics for
  `LlmJudge`.
- [Scorers reference](../reference/scorers.md): the scorer and combinator API.
</content>
</invoke>
