"""Thin notebook helpers; all business logic remains in chunkbench."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root for display and diagnostics."""
    return Path.cwd()


def show_summary(result: dict) -> None:
    """Display a compact pipeline summary."""
    print(f"Run directory: {result.get('run_dir')}")
    print(f"Completed: {result.get('completed_methods', [])}")
    print(f"Failed: {result.get('failed_methods', [])}")

