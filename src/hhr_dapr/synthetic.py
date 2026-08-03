"""Small deterministic corpus for structural and end-to-end smoke tests only."""

from __future__ import annotations

import pandas as pd

from .schema import NormalizedDAPRDataset, validate_dataset


def make_synthetic_dataset(name: str = "synthetic") -> NormalizedDAPRDataset:
    documents = pd.DataFrame(
        [
            {
                "document_id": "d1",
                "title": "Paris",
                "text": "Paris is the capital of France. The Seine crosses Paris.",
            },
            {
                "document_id": "d2",
                "title": "Marie Curie",
                "text": "Marie Curie was born in Warsaw and studied radioactivity.",
            },
            {
                "document_id": "d3",
                "title": "Mars",
                "text": (
                    "Mars is the fourth planet. It has two moons, Phobos and Deimos."
                ),
            },
            {
                "document_id": "d4",
                "title": None,
                "text": "The Pacific Ocean is the largest ocean on Earth.",
            },
        ]
    )
    passages = pd.DataFrame(
        [
            {
                "passage_id": "p1",
                "document_id": "d1",
                "passage_text": "Paris is the capital of France.",
                "passage_position": 0,
            },
            {
                "passage_id": "p2",
                "document_id": "d1",
                "passage_text": "The Seine river crosses Paris.",
                "passage_position": 1,
            },
            {
                "passage_id": "p3",
                "document_id": "d2",
                "passage_text": "Marie Curie was born in Warsaw.",
                "passage_position": 0,
            },
            {
                "passage_id": "p4",
                "document_id": "d2",
                "passage_text": "She researched radioactivity.",
                "passage_position": 1,
            },
            {
                "passage_id": "p5",
                "document_id": "d3",
                "passage_text": "Mars is the fourth planet from the Sun.",
                "passage_position": 0,
            },
            {
                "passage_id": "p6",
                "document_id": "d3",
                "passage_text": "Phobos and Deimos are moons of Mars.",
                "passage_position": 1,
            },
            {
                "passage_id": "p7",
                "document_id": "d4",
                "passage_text": "The Pacific Ocean is Earth's largest ocean.",
                "passage_position": 0,
            },
        ]
    )
    queries = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query_text": "capital of France",
                "dataset": name,
                "split": "test",
            },
            {
                "query_id": "q2",
                "query_text": "where was Marie Curie born",
                "dataset": name,
                "split": "test",
            },
            {
                "query_id": "q3",
                "query_text": "moons of Mars",
                "dataset": name,
                "split": "test",
            },
            {
                "query_id": "q4",
                "query_text": "largest ocean",
                "dataset": name,
                "split": "test",
            },
        ]
    )
    qrels = pd.DataFrame(
        [
            {"query_id": "q1", "passage_id": "p1", "relevance": 2},
            {"query_id": "q2", "passage_id": "p3", "relevance": 1},
            {"query_id": "q3", "passage_id": "p6", "relevance": 1},
            {"query_id": "q4", "passage_id": "p7", "relevance": 1},
        ]
    )
    result = NormalizedDAPRDataset(name, documents, passages, queries, qrels)
    validate_dataset(result)
    return result
