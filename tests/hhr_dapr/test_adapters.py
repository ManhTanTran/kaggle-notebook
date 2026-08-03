from __future__ import annotations

import pytest

from hhr_dapr.adapters import ADAPTER_REGISTRY
from hhr_dapr.schema import NormalizedDAPRDataset, validate_dataset


@pytest.mark.parametrize("dataset_name", sorted(ADAPTER_REGISTRY))
def test_each_adapter_emits_normalized_schema(dataset_name, dapr_root):
    dataset = ADAPTER_REGISTRY[dataset_name](dapr_root).load()
    validate_dataset(dataset)
    assert dataset.name == dataset_name
    assert list(dataset.documents.columns) == ["document_id", "title", "text"]
    assert list(dataset.passages.columns) == [
        "passage_id",
        "document_id",
        "passage_text",
        "passage_position",
    ]
    assert list(dataset.queries.columns) == [
        "query_id",
        "query_text",
        "dataset",
        "split",
    ]
    assert list(dataset.qrels.columns) == ["query_id", "passage_id", "relevance"]


def test_qrel_integrity_rejects_unknown_passage(synthetic_dataset):
    broken_qrels = synthetic_dataset.qrels.copy()
    broken_qrels.loc[0, "passage_id"] = "does-not-exist"
    broken = NormalizedDAPRDataset(
        synthetic_dataset.name,
        synthetic_dataset.documents,
        synthetic_dataset.passages,
        synthetic_dataset.queries,
        broken_qrels,
    )
    with pytest.raises(ValueError, match="unknown passages"):
        validate_dataset(broken)


def test_passage_position_must_be_non_negative(synthetic_dataset):
    passages = synthetic_dataset.passages.copy()
    passages.loc[0, "passage_position"] = -1
    broken = NormalizedDAPRDataset(
        synthetic_dataset.name,
        synthetic_dataset.documents,
        passages,
        synthetic_dataset.queries,
        synthetic_dataset.qrels,
    )
    with pytest.raises(ValueError, match="passage_position"):
        validate_dataset(broken)
