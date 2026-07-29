"""Method comparison helpers."""

import pandas as pd


def rank_methods(
    metrics: pd.DataFrame, metric: str = "EvidenceCoverage@10"
) -> pd.DataFrame:
    """Return methods sorted descending by a selected metric."""
    if metric not in metrics:
        raise KeyError(f"Metric not found: {metric}")
    return metrics.sort_values(metric, ascending=False).reset_index(drop=True)
