from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from hhr_dapr.config import RECOMMENDED_METHODS, RunConfig


@dataclass(frozen=True)
class TestDataset:
    name: str
    documents: pd.DataFrame
    passages: pd.DataFrame
    queries: pd.DataFrame
    qrels: pd.DataFrame
    query_metadata: pd.DataFrame | None = None


def make_test_dataset(name: str = "synthetic") -> TestDataset:
    documents = pd.DataFrame(
        [
            {"document_id": "d1", "title": "Paris", "text": "Paris France Seine"},
            {
                "document_id": "d2",
                "title": "Marie Curie",
                "text": "Marie Curie Warsaw radioactivity",
            },
            {"document_id": "d3", "title": "Mars", "text": "Mars Phobos Deimos"},
            {
                "document_id": "d4",
                "title": None,
                "text": "Pacific Ocean largest ocean",
            },
        ]
    )
    passage_values = [
        ("p1", "d1", "Paris is the capital of France.", 0),
        ("p2", "d1", "The Seine river crosses Paris.", 1),
        ("p3", "d2", "Marie Curie was born in Warsaw.", 0),
        ("p4", "d2", "She researched radioactivity.", 1),
        ("p5", "d3", "Mars is the fourth planet from the Sun.", 0),
        ("p6", "d3", "Phobos and Deimos are moons of Mars.", 1),
        ("p7", "d4", "The Pacific Ocean is Earth's largest ocean.", 0),
    ]
    passages = pd.DataFrame(
        passage_values,
        columns=["passage_id", "document_id", "passage_text", "passage_position"],
    )
    query_values = [
        ("q1", "capital of France", name, "test"),
        ("q2", "where was Marie Curie born", name, "test"),
        ("q3", "moons of Mars", name, "test"),
        ("q4", "largest ocean", name, "test"),
    ]
    queries = pd.DataFrame(
        query_values, columns=["query_id", "query_text", "dataset", "split"]
    )
    qrels = pd.DataFrame(
        [("q1", "p1", 2), ("q2", "p3", 1), ("q3", "p6", 1), ("q4", "p7", 1)],
        columns=["query_id", "passage_id", "relevance"],
    )
    return TestDataset(name, documents, passages, queries, qrels)


@pytest.fixture
def synthetic_dataset():
    return make_test_dataset()


@pytest.fixture
def smoke_config(tmp_path: Path):
    return RunConfig(
        run_mode="smoke",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "outputs",
        methods=RECOMMENDED_METHODS,
    )
