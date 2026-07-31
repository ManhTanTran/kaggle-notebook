import json
from pathlib import Path

import pytest

from chunkbench.common.exceptions import ContractError
from chunkbench.common.types import DatasetBundle, Document, Evidence, Query
from chunkbench.config.loader import load_yaml
from chunkbench.data.uit_viquad_evidence import answer_sentence_groups
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


def test_hotpot_preserves_case_sensitive_wikipedia_title_identity(tmp_path: Path):
    questions_path = tmp_path / "validation.json"
    corpus_path = tmp_path / "articles.jsonl"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "_id": "case-sensitive",
                    "question": "Which page is the magazine?",
                    "answer": "Popular Science",
                    "supporting_facts": [["Popular Science", 0]],
                }
            ]
        ),
        encoding="utf-8",
    )
    articles = [
        {
            "title": "Popular Science",
            "text": [["Popular Science is an American magazine."]],
        },
        {
            "title": "Popular science",
            "text": [["Popular science explains science to a general audience."]],
        },
    ]
    corpus_path.write_text(
        "\n".join(json.dumps(article) for article in articles) + "\n",
        encoding="utf-8",
    )

    bundle = build_dataset_adapter(
        "hotpotqa_fullwiki",
        {
            "questions_path": str(questions_path),
            "corpus_path": str(corpus_path),
            "split": "validation",
        },
    ).load()

    assert bundle.evidence[0].text == "Popular Science is an American magazine."
    assert bundle.evidence[0].metadata["article_title"] == "Popular Science"


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


def test_uit_viquad_uses_minimal_sentence_cover_for_ellipsis_span():
    context = (
        "Mở đầu. Là nơi đặt trụ sở OECD, UNESCO... cộng với hoạt động tài chính "
        "và du lịch đã khiến Paris trở nên nổi tiếng. Kết thúc."
    )
    answer_text = (
        "Là nơi đặt trụ sở OECD, UNESCO... cộng với hoạt động tài chính và du lịch"
    )
    answer_start = context.index(answer_text)

    groups = answer_sentence_groups(
        context, [{"text": answer_text, "answer_start": answer_start}]
    )

    assert len(groups) == 1
    assert context[groups[0]["sentence_start"] : groups[0]["sentence_end"]] == (
        "Là nơi đặt trụ sở OECD, UNESCO... cộng với hoạt động tài chính "
        "và du lịch đã khiến Paris trở nên nổi tiếng."
    )


def test_uit_viquad_repairs_one_indexed_answer_start_with_provenance():
    context = "Mở đầu. Quận 12 ở phía Đông."
    answer_text = "Quận 12 ở phía Đông"
    one_indexed_start = context.index(answer_text) - 1

    groups = answer_sentence_groups(
        context, [{"text": answer_text, "answer_start": one_indexed_start}]
    )

    span = groups[0]["answer_spans"][0]
    assert span["answer_start"] == context.index(answer_text)
    assert span["original_answer_start"] == one_indexed_start
    assert span["answer_start_repaired"] is True


def test_uit_viquad_does_not_split_decimal_punctuation_inside_sentence():
    context = "Cách đây ít nhất 40.000 năm đã có sự hiện diện của con người."
    answer_text = "40.000 năm"
    answer_start = context.index(answer_text)

    groups = answer_sentence_groups(
        context, [{"text": answer_text, "answer_start": answer_start}]
    )

    assert context[groups[0]["sentence_start"] : groups[0]["sentence_end"]] == context


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
