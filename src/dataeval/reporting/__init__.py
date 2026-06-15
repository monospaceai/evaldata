"""Reporting: render eval outcomes for humans (terminal) and machines (JUnit/JSON, later)."""

from dataeval.reporting.terminal import render_failure, render_solver_error, render_summary

__all__ = ["render_failure", "render_solver_error", "render_summary"]
