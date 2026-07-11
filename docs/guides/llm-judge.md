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

The grader's score lands in `result.score` and its rationale in `result.explanation`; the result
is stamped `basis="judged"` to mark it a probabilistic judgment, not a proof.

Pass it to `assert_eval` like any other scorer:

```python
assert_eval(case, solver, scorers=[judge])
```

## Inconclusive, not a guess

A judge can fail to reach a verdict — the provider call errors, or the reply doesn't parse. When
that happens the result is **inconclusive** (`verdict="inconclusive"`), carrying the failure in
its explanation, rather than a fail. Inconclusive means the judge couldn't decide; it doesn't mean
the answer is wrong.

## Shape the grader's judgment

`criteria` alone is often enough. Four optional arguments make the grade more consistent when it
isn't:

- **`steps`** — an ordered checklist the grader works through before scoring, rendered as a
  numbered block.
- **`rubric`** — score bands that say what each range means, so the number is anchored.
- **`examples`** — few-shot anchors, each a graded output with its score and the reason for it.
- **`threshold`** — the minimum score (inclusive) for a pass. Defaults to `0.5`.

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

Only the arguments you set appear in the prompt — an unset `rubric` or `examples` is simply
omitted.

## Choose what the grader sees

By default the judge shows the grader three case fields, each only when it is available:

- **`input`** — the case's question.
- **`actual_output`** — the SQL the solver produced.
- **`expected_output`** — the gold query's SQL, when the case has a `GoldQuery` expected.

Restrict this with `show` when a field would bias or distract the grade — for example, grade the
output on its own merits without revealing the reference answer:

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

The grader must support structured output: the judge requests a JSON `{score, reason}` reply and
treats a malformed one as inconclusive. Grading defaults to `temperature=0.0` for repeatable
scores; pass `temperature=None` to leave the provider default, or a `timeout` to bound the call.

## Run it deterministically in CI

A judged eval calls a live model, which costs money and varies run to run. To exercise the eval
path in CI without a live model call, mock the grader reply. Add a `conftest.py` next to your
test that returns the structured `{score, reason}` the judge
expects:

```python
--8<-- "examples/05_llm_judge/conftest.py"
```

With the mock in place the eval runs offline, with no key:

```python
--8<-- "examples/05_llm_judge/test_judged_equivalence.py"
```

```bash
uv run pytest test_judged_equivalence.py -q
```

Remove the `conftest.py` (and set the provider key, e.g. `OPENAI_API_KEY`) to grade against the
live model instead.

!!! tip "Run it from a clone"
    This is the bundled `examples/05_llm_judge/` example. If you've cloned the repo, run it
    directly with `uv run pytest examples/05_llm_judge` — it runs mocked, with no key needed.

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

- [Check semantic equivalence](semantic-equivalence.md) — confirm two queries match by AST
  normalization, with the judge as the fallback.
- [Scorers reference](../reference/scorers.md) — `LlmJudge`, `JudgeExample`, `RubricBand`.
- [Concepts](../concepts.md) — scorers, solvers, and expected-types in depth.
