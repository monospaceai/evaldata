# Score with an LLM judge

Not every answer can be checked by comparing rows or normalizing SQL ASTs. Does the SQL read
clearly? Does it follow your team's style? Do two queries return the same thing when you have no
warehouse to run them on? `LlmJudge` answers questions like these: it asks a grader model to
score the case against criteria you write, and turns that score into a pass/fail verdict.

## What the judge does

You give the judge a standard to grade against, written as plain text in `criteria`. It builds a
prompt from that standard and the case, asks the grader model to reason about how well the case
meets it and give a score from `0.0` to `1.0`, then compares that score to a `threshold` to
decide the verdict:

```python
from evaldata import LlmJudge

judge = LlmJudge(
    model="openai/gpt-4o-mini",
    criteria="The SQL answers the question, reads clearly, and uses no SELECT *.",
    threshold=0.7,
)
```

The grader's score is in `result.score` and its rationale is in `result.explanation`. The result
has `basis="judged"`, which marks it as a probabilistic judgment rather than a proof.

Pass it to `assert_eval` like any other scorer:

```python
assert_eval(case, solver, scorers=[judge])
```

## Inconclusive, not a guess

A judge can fail to reach a verdict if the provider call errors or the response doesn't parse. The
result is then **inconclusive** (`verdict="inconclusive"`) and carries the failure in its
explanation. Inconclusive means the judge couldn't decide; it doesn't mean the answer is wrong.

## Configure the judge

`criteria` alone is often enough. Four optional arguments control the grading prompt and pass
threshold:

- **`steps`**: an ordered checklist the grader works through before scoring, rendered as a
  numbered block.
- **`rubric`**: score bands that describe each range.
- **`examples`**: few-shot anchors, each a graded output with its score and reason.
- **`threshold`**: the minimum score (inclusive) for a pass. Defaults to `0.5`.

```python
from evaldata import JudgeExample, LlmJudge, RubricBand

judge = LlmJudge(
    model="openai/gpt-4o-mini",
    criteria="The SQL answers the question and is idiomatic for an analytics team.",
    steps=[
        "Check the query returns the columns the question asks for.",
        "Check it filters and aggregates correctly.",
        "Judge readability: aliases, no SELECT *, sensible formatting.",
    ],
    rubric=[
        RubricBand(min_score=0.0, max_score=0.4, description="Wrong result or unreadable."),
        RubricBand(min_score=0.4, max_score=0.8, description="Correct result, rough style."),
        RubricBand(min_score=0.8, max_score=1.0, description="Correct and idiomatic."),
    ],
    examples=[
        JudgeExample(
            actual_output="SELECT name FROM customers WHERE country = 'US'",
            score=1.0,
            reason="Correct filter, named column, clear.",
        ),
    ],
    threshold=0.8,
)
```

Unset `rubric` and `examples` values are not included in the prompt.

## Choose what the grader sees

By default, the judge shows the grader these case fields when they are available:

- **`input`**: the case's question.
- **`actual_output`**: the SQL the solver produced.
- **`expected_output`**: the gold query's SQL, when the case has a `GoldQuery` expected.

Use `show` to hide fields that would bias or distract the grader. For example, grade the output
without revealing the reference answer:

```python
judge = LlmJudge(
    model="openai/gpt-4o-mini",
    criteria="The SQL answers the question.",
    show=["input", "actual_output"],   # withhold the expected query
)
```

## Choosing a grader model

The grader is separate from the solver. `model` takes any litellm identifier (or an `Llm` to use
directly), so the grader can run on a different provider from the model being evaluated.

The grader must support structured output: the judge requests a JSON `{score, reason}` response
and treats a malformed one as inconclusive. Grading defaults to `temperature=0.0` for repeatable
scores; pass `temperature=None` to leave the provider default, or a `timeout` to bound the call.

## Use fixed judge responses in CI

A judged eval calls a model, which costs money and can vary between runs. Add a `conftest.py` next
to your test that returns the structured `{score, reason}` response the judge expects:

```python
--8<-- "examples/05_llm_judge/conftest.py"
```

`conftest.py` makes the eval use fixed judge responses:

```python
--8<-- "examples/05_llm_judge/test_judged_equivalence.py"
```

```bash
uv run pytest test_judged_equivalence.py -q
```

Remove `conftest.py` to grade with the model.

!!! tip "Run it from a clone"
    This is the bundled `examples/05_llm_judge/` example. If you've cloned the repo, run it
    with `uv run pytest examples/05_llm_judge`. It includes fixed judge responses.

## A ready-made judge: SQL equivalence

`sql_equivalence_judge(model)` is an `LlmJudge` pre-loaded with criteria and examples for one
common question: do two SQL queries return the same rows? It grades `actual_output` against
`expected_output` (the gold query), forgiving differences that never change the result and
penalising those that do.

```python
from evaldata import sql_equivalence_judge

judge = sql_equivalence_judge("openai/gpt-4o-mini")
```

Most often you don't reach for it directly. `judged_equivalence(model)` first tries to confirm
equivalence by normalizing and comparing SQL ASTs, and only falls back to this judge when that
check is inconclusive. Use it when you have no warehouse to run the queries against. See
[Check semantic equivalence](semantic-equivalence.md) for that cascade.

## Next steps

- [Check semantic equivalence](semantic-equivalence.md): confirm two queries match by AST
  normalization, with the judge as the fallback.
- [Scorers reference](../reference/scorers.md): `LlmJudge`, `JudgeExample`, `RubricBand`.
- [Concepts](../concepts.md): scorers, solvers, and expected types in depth.
