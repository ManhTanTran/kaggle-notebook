from __future__ import annotations

import math

import pytest

from hhr_dapr.metrics import ndcg_at_k, recall_at_k


def test_ndcg_at_10_uses_graded_labels():
    relevance = {"high": 2, "medium": 1, "irrelevant": 0}
    actual = ndcg_at_k(["medium", "high", "irrelevant"], relevance, 10)
    actual_dcg = 1.0 + 3.0 / math.log2(3)
    ideal_dcg = 3.0 + 1.0 / math.log2(3)
    assert actual == pytest.approx(actual_dcg / ideal_dcg)


def test_recall_at_k_counts_binary_relevance_set():
    relevance = {"a": 2, "b": 1, "c": 0}
    assert recall_at_k(["c", "a"], relevance, 2) == pytest.approx(0.5)
    assert recall_at_k(["a", "b"], relevance, 2) == pytest.approx(1.0)
