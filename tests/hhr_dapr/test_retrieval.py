from __future__ import annotations

from hhr_dapr.retrieval import (
    SparseDocumentRetriever,
    SparsePassageRetriever,
    hhr_interleave,
    reciprocal_rank_fusion,
)


def _hit(item_id, score, sparse_rank=None, dense_rank=None):
    return {
        "item_id": item_id,
        "score": score,
        "sparse_rank": sparse_rank,
        "dense_rank": dense_rank,
        "source_method": "sparse" if sparse_rank else "dense",
    }


def test_hhr_interleave_halves_and_deduplicates():
    sparse = [_hit("a", 4, sparse_rank=1), _hit("b", 3, sparse_rank=2)]
    dense = [_hit("a", 2, dense_rank=1), _hit("c", 1, dense_rank=2)]
    result, metadata = hhr_interleave(sparse, dense, k=4)
    assert [hit["item_id"] for hit in result] == ["a", "b", "c"]
    assert result[0]["sparse_rank"] == 1
    assert result[0]["dense_rank"] == 1
    assert result[0]["source_method"] == "sparse+dense"
    assert metadata["overlap_shortfall"] is True
    assert metadata["unique_results"] == 3


def test_rrf_scores_union_and_orders_by_rank_evidence():
    sparse = [_hit("a", 3, sparse_rank=1), _hit("b", 2, sparse_rank=2)]
    dense = [_hit("b", 3, dense_rank=1), _hit("c", 2, dense_rank=2)]
    result, metadata = reciprocal_rank_fusion(sparse, dense, k=3, rrf_k=10)
    assert [hit["item_id"] for hit in result] == ["b", "a", "c"]
    assert result[0]["source_method"] == "sparse+dense"
    assert metadata["strategy"] == "rrf"


def test_passage_retrieval_filters_to_selected_documents(
    synthetic_dataset, smoke_config
):
    retriever = SparsePassageRetriever(synthetic_dataset.passages, smoke_config)
    results = retriever.retrieve("capital France Mars", k=10, document_ids={"d1"})
    returned = {hit["item_id"] for hit in results}
    assert returned == {"p1", "p2"}


def test_sparse_indices_resume_from_cache(synthetic_dataset, smoke_config):
    first = SparseDocumentRetriever(synthetic_dataset.documents, smoke_config)
    second = SparseDocumentRetriever(synthetic_dataset.documents, smoke_config)
    assert first.cache_path.is_file()
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert [hit["item_id"] for hit in first.retrieve("Mars", 2)] == [
        hit["item_id"] for hit in second.retrieve("Mars", 2)
    ]
