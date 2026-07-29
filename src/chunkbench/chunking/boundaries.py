"""Boundary scoring and deterministic threshold policies."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BoundaryScore:
    """Score assigned between two consecutive source segments."""

    left_segment_id: str
    right_segment_id: str
    position: int
    score: float
    is_boundary: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def cosine_distance(left: NDArray[np.float32], right: NDArray[np.float32]) -> float:
    """Return one minus cosine similarity; zero vectors are maximally distant."""
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0
    similarity = float(np.dot(left, right) / denominator)
    return 1.0 - similarity


def threshold_value(scores: list[float], policy: dict[str, Any] | None) -> float:
    """Resolve an explicit threshold policy without hidden defaults."""
    if not scores:
        return float("inf")
    policy = policy or {"type": "percentile", "value": 90.0}
    kind = str(policy.get("type", "percentile"))
    value = float(policy.get("value", 90.0))
    values = np.asarray(scores, dtype=np.float32)
    if kind == "percentile":
        return float(np.percentile(values, value))
    if kind == "absolute":
        return value
    if kind == "standard_deviation":
        return float(values.mean() + value * values.std())
    if kind == "interquartile":
        return float(
            values.mean()
            + value * (np.percentile(values, 75) - np.percentile(values, 25))
        )
    raise ValueError(f"Unknown threshold policy: {kind}")


def select_boundaries(
    scores: list[float],
    policy: dict[str, Any] | Callable[[list[float]], float] | None,
) -> tuple[float, set[int]]:
    """Select positions whose score strictly exceeds a resolved threshold."""
    if callable(policy):
        threshold = float(policy(scores))
    else:
        threshold = threshold_value(scores, policy)
    return threshold, {index for index, score in enumerate(scores) if score > threshold}
