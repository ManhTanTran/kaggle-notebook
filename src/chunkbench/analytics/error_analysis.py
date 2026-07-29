"""Small, artifact-oriented error analysis queries."""

from pathlib import Path

import pandas as pd


def find_error_cases(run_dir: str | Path, method: str) -> pd.DataFrame:
    """Return retrieval rows for queries with no top-10 evidence hit."""
    root = Path(run_dir) / method
    metrics = pd.read_json(root / "metrics.json", typ="series")
    retrieval = pd.read_csv(root / "retrieval.csv")
    if float(metrics.get("Hit@10", 1.0)) > 0:
        return retrieval.iloc[0:0]
    return retrieval
