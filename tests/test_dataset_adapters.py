from pathlib import Path

import pytest

from chunkbench.common.exceptions import ContractError
from chunkbench.common.types import DatasetBundle, Document, Evidence, Query
from chunkbench.config.loader import load_yaml
from chunkbench.data.validation import validate_dataset
from chunkbench.registry.datasets import (
    CORE_DATASETS,
    DATASET_REGISTRY,
    build_dataset_adapter,
    list_registered_datasets,
    register_dataset,
)


def _load(name: str):
    config = load_yaml(Path("configs/datasets/fixtures") / f"{name}.yaml")
    bundle = build_dataset_adapter(name, config).load()
    report = validate_dataset(bundle, config["validation"])
    return bundle, report


@pytest.mark.parametrize("name", CORE_DATASETS)
def test_core_adapter_contracts_and_mapping(name: str):
    bundle, report = _load(name)
    assert len(bundle.documents) >= 2
    assert len(bundle.queries) >= 2
    assert any(
        sum(item.query_id == query.query_id for item in bundle.evidence) > 1
        for query in bundle.queries
    )
    assert report["evidence_mapping_rate"] == 1.0
    document_ids = {document.document_id for document in bundle.documents}
    assert all(
        set(query.relevant_document_ids) <= document_ids for query in bundle.queries
    )
    assert all(item.document_id in document_ids for item in bundle.evidence)


def test_registry_contains_four_core_datasets_and_supports_extension():
    assert CORE_DATASETS == [
        "qasper",
        "hotpotqa_fullwiki",
        "uit_viquad",
        "vimqa",
    ]
    assert set(CORE_DATASETS) <= set(list_registered_datasets())

    class CustomAdapter:
        def __init__(self, **config):
            self.config = config

        def load(self):
            return DatasetBundle([], [], [], {})

    register_dataset("custom_dataset", lambda config: CustomAdapter(**config))
    try:
        assert isinstance(build_dataset_adapter("custom_dataset"), CustomAdapter)
    finally:
        DATASET_REGISTRY.pop("custom_dataset")


def test_qasper_uses_annotated_evidence_and_paragraph_locators():
    bundle, _ = _load("qasper")
    assert len(bundle.evidence) == 3
    assert all(item.metadata["granularity"] == "paragraph" for item in bundle.evidence)
    assert all(item.metadata["section_name"] for item in bundle.evidence)


def test_hotpot_and_vimqa_preserve_multi_hop_supporting_facts():
    for name in ("hotpotqa_fullwiki", "vimqa"):
        bundle, _ = _load(name)
        first = bundle.queries[0]
        assert len(first.relevant_document_ids) == 2
        first_evidence = [
            item for item in bundle.evidence if item.query_id == first.query_id
        ]
        assert len(first_evidence) == 2
        assert all(
            item.metadata["granularity"] == "sentence" for item in first_evidence
        )


def test_uit_viquad_sentence_containing_answer_policy():
    bundle, _ = _load("uit_viquad")
    by_query = {
        query.query_id: [
            item for item in bundle.evidence if item.query_id == query.query_id
        ]
        for query in bundle.queries
    }
    assert len(by_query["viquad-q1"]) == 1
    assert len(by_query["viquad-q1"][0].metadata["answer_spans"]) == 2
    assert len(by_query["viquad-q2"]) == 2
    assert by_query["viquad-q2"][0].metadata["answer_start"] == 0
    assert by_query["viquad-q2"][1].metadata["answer_spans"][-1]["answer_start"] == 52


def test_validation_rejects_out_of_bounds_evidence_span():
    bundle = DatasetBundle(
        [Document("d", "valid text")],
        [Query("q", "question", ("d",))],
        [
            Evidence(
                "q",
                "e",
                "d",
                "valid",
                1,
                {"char_spans": [[0, 100]]},
            )
        ],
    )
    with pytest.raises(ContractError, match="mapping validation failed"):
        validate_dataset(bundle)
