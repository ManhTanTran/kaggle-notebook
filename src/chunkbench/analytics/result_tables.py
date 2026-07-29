"""Helpers for loading artifact tables."""

from pathlib import Path

import pandas as pd


def load_result_tables(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load run-level metrics and chunk statistics."""
    root = Path(run_dir)
    return (
        pd.read_csv(root / "benchmark_metrics.csv"),
        pd.read_csv(root / "chunk_statistics.csv"),
    )
