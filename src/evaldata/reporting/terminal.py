"""Rich-backed terminal rendering of eval failures, for embedding in the `AssertionError`."""

import io
import textwrap
from collections.abc import Sequence

from rich import box
from rich.console import Console, RenderableType
from rich.table import Table

from evaldata.reporting.collector import CaseReport
from evaldata.types import EvalCase, ExecutionResult, ResultSetDiff, ScoreResult, SolverError, SolverOutput


def render_failure(
    case: EvalCase,
    output: SolverOutput,
    result: ExecutionResult,
    failures: Sequence[ScoreResult],
) -> str:
    """Render a scorer failure: the case, the generated SQL, and each failing score's diff.

    Args:
        case: The eval case that failed.
        output: The solver output (carries the generated SQL).
        result: The execution result for the generated SQL.
        failures: The scorer results that failed.

    Returns:
        A plain-text failure message for embedding in the `AssertionError`.
    """
    lines = [
        f"evaldata case {case.id!r} failed",
        f"  input: {case.input}",
        f"  sql:   {output.output}",
    ]
    if result.error is not None:
        lines.append(f"  execution error: {result.error.message}")
    for score in failures:
        lines.append(f"  scorer {score.scorer!r}: FAIL")
        if score.explanation:
            lines.append(f"    {score.explanation}")
        if score.diff is not None:
            lines.append(_render_diff(score.diff))
    return "\n".join(lines)


def render_solver_error(case: EvalCase, error: SolverError) -> str:
    """Render a solver failure (no SQL was executed).

    Args:
        case: The eval case that failed.
        error: The typed solver error to render.

    Returns:
        A plain-text failure message for embedding in the `AssertionError`.
    """
    return "\n".join(
        [
            f"evaldata case {case.id!r} failed: solver error",
            f"  input: {case.input}",
            f"  solver error [{error.kind}]: {error.message}",
        ]
    )


def render_summary(case_reports: Sequence[CaseReport]) -> str:
    """Render a run-level rollup table (one row per case) plus a pass/fail tally.

    Args:
        case_reports: The accumulated case outcomes.

    Returns:
        A plain-text (no-ANSI) table plus a `N passed, M failed` line.
    """
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
    """Build the detail cell for a case: its solver error, or any failed scorer names.

    Args:
        report: The case outcome to summarize.

    Returns:
        The solver error if present, else a comma-separated list of failed scorer names.
    """
    if report.error is not None:
        return f"solver error [{report.error.kind}]: {report.error.message}"
    return ", ".join(score.scorer for score in report.scores if not score.passed)


def _render_diff(diff: ResultSetDiff) -> str:
    """Render a `ResultSetDiff` as Rich tables, returned as an indented plain-text block.

    Section labels are emitted as preceding plain lines rather than table titles: a table
    sizes to its content, which would otherwise wrap a longer title into that narrow width.

    Args:
        diff: The structured result-set difference to render.

    Returns:
        An indented, plain-text block of Rich tables describing the differences.
    """
    renderables: list[RenderableType] = ["result-set diff", _summary_table(diff)]

    if diff.missing_columns:
        renderables.append(f"missing columns: {diff.missing_columns}")
    if diff.unexpected_columns:
        renderables.append(f"unexpected columns: {diff.unexpected_columns}")
    if diff.column_order_mismatch:
        renderables.append("column order differs")
    if diff.type_mismatches:
        renderables += ["type mismatches", _type_mismatch_table(diff)]
    if diff.column_mismatches:
        renderables += ["column mismatches", _column_mismatch_table(diff)]
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


def _column_mismatch_table(diff: ResultSetDiff) -> Table:
    table = Table(box=box.SIMPLE, pad_edge=False)
    table.add_column("column")
    table.add_column("unexpected")
    for cm in diff.column_mismatches:
        table.add_row(cm.column, str(cm.unexpected_count))
    return table


def _rows_table(rows: Sequence[dict[str, object]]) -> Table:
    table = Table(box=box.SIMPLE, pad_edge=False)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    for key in keys:
        table.add_column(key)
    for row in rows:
        table.add_row(*(repr(row.get(key)) for key in keys))
    return table
