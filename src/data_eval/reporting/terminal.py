"""Rich-backed terminal rendering of eval failures, for embedding in the ``AssertionError``.

``assert_eval`` raises an ``AssertionError`` whose message is what pytest prints on failure.
The structured diagnostic — the ``ResultSetDiff`` — is rendered here as aligned Rich tables;
the surrounding scaffolding (input, SQL, error text) stays verbatim plain text, because
routing SQL through Rich would soft-wrap and mangle it.

Tables are rendered to a string via ``Console(file=StringIO(), no_color=True)``: Rich emits
no ANSI for non-terminal output, so the result is plain Unicode tables that read correctly
under ``-q``/``-s``, pytest-xdist, and non-TTY CI logs alike — no escape codes leak into CI.
Colored, run-level summaries via ``pytest_terminal_summary`` and a machine-readable JSON
artifact are a later increment; this module owns the per-failure human rendering.
"""

import io
import textwrap
from collections.abc import Sequence

from rich import box
from rich.console import Console, RenderableType
from rich.table import Table

from data_eval.reporting.collector import CaseReport
from data_eval.types import EvalCase, ExecutionResult, ResultSetDiff, ScoreResult, SolverError, SolverOutput


def render_failure(
    case: EvalCase,
    output: SolverOutput,
    result: ExecutionResult,
    failures: Sequence[ScoreResult],
) -> str:
    """Render a scorer failure: the case, the generated SQL, and each failing score's diff."""
    lines = [
        f"data-eval case {case.id!r} failed",
        f"  input: {case.input}",
        f"  sql:   {output.output}",
    ]
    if result.error is not None:
        lines.append(f"  execution error: {result.error}")
    for score in failures:
        lines.append(f"  scorer {score.scorer!r}: FAIL")
        if score.explanation:
            lines.append(f"    {score.explanation}")
        if score.diff is not None:
            lines.append(_render_diff(score.diff))
    return "\n".join(lines)


def render_solver_error(case: EvalCase, error: SolverError) -> str:
    """Render a solver failure (no SQL was executed)."""
    return "\n".join(
        [
            f"data-eval case {case.id!r} failed: solver error",
            f"  input: {case.input}",
            f"  solver error [{error.kind}]: {error.message}",
        ]
    )


def render_summary(case_reports: Sequence[CaseReport]) -> str:
    """Render a run-level rollup table (one row per case) plus a pass/fail tally."""
    table = Table(box=box.SIMPLE, pad_edge=False)
    table.add_column("case")
    table.add_column("result")
    table.add_column("detail")
    for report in case_reports:
        table.add_row(report.id, "PASS" if report.passed else "FAIL", _summary_detail(report))

    buffer = io.StringIO()
    console = Console(file=buffer, no_color=True, highlight=False, markup=False, width=100)
    console.print(table)
    passed = sum(1 for r in case_reports if r.passed)
    console.print(f"{passed} passed, {len(case_reports) - passed} failed")
    return "\n".join(line.rstrip() for line in buffer.getvalue().splitlines())


def _summary_detail(report: CaseReport) -> str:
    """The detail cell for a case: its solver error, or the names of any failed scorers."""
    if report.error is not None:
        return report.error
    return ", ".join(score.scorer for score in report.scores if not score.passed)


def _render_diff(diff: ResultSetDiff) -> str:
    """Render a ``ResultSetDiff`` as Rich tables, returned as an indented plain-text block.

    Section labels are emitted as preceding plain lines rather than table titles: a table
    sizes to its content, which would otherwise wrap a longer title into that narrow width.
    """
    renderables: list[RenderableType] = ["result-set diff", _summary_table(diff)]

    if diff.missing_columns:
        renderables.append(f"missing columns: {diff.missing_columns}")
    if diff.extra_columns:
        renderables.append(f"extra columns: {diff.extra_columns}")
    if diff.column_order_mismatch:
        renderables.append("column order differs")
    if diff.type_mismatches:
        renderables += ["type mismatches", _type_mismatch_table(diff)]
    if diff.missing_row_count:
        renderables += [f"missing rows ({diff.missing_row_count}); sample:", _rows_table(diff.sample_missing_rows)]
    if diff.extra_row_count:
        renderables += [f"extra rows ({diff.extra_row_count}); sample:", _rows_table(diff.sample_extra_rows)]

    buffer = io.StringIO()
    console = Console(file=buffer, no_color=True, highlight=False, markup=False, width=100)
    for renderable in renderables:
        console.print(renderable)
    block = "\n".join(line.rstrip() for line in buffer.getvalue().splitlines())
    return textwrap.indent(block, "    ")


def _summary_table(diff: ResultSetDiff) -> Table:
    table = Table(box=box.SIMPLE, pad_edge=False)
    table.add_column("")
    table.add_column("expected", justify="right")
    table.add_column("actual", justify="right")
    table.add_row("rows", str(diff.expected_row_count), str(diff.actual_row_count))
    return table


def _type_mismatch_table(diff: ResultSetDiff) -> Table:
    table = Table(box=box.SIMPLE, pad_edge=False)
    table.add_column("column")
    table.add_column("expected")
    table.add_column("actual")
    for tm in diff.type_mismatches:
        table.add_row(tm.column, tm.expected, tm.actual)
    return table


def _rows_table(rows: Sequence[dict[str, object]]) -> Table:
    table = Table(box=box.SIMPLE, pad_edge=False)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    for key in keys:
        table.add_column(key)
    for row in rows:
        table.add_row(*(repr(row.get(key)) for key in keys))
    return table
