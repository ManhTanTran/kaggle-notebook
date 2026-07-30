import numpy as np

from chunkbench.chunking.semantic_single_linkage import SemanticSingleLinkageChunker
from chunkbench.common.types import Document
from chunkbench.embedding.base import BaseEmbedder


class ClusterEmbedder(BaseEmbedder):
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1, 0], [0, 1], [1, 0]][: len(texts)], dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_documents(texts)


def test_single_linkage_merges_expected_non_contiguous_pair():
    chunks = SemanticSingleLinkageChunker(
        embedder=ClusterEmbedder(),
        lambda_weight=0,
        target_clusters=2,
        distance_threshold=0.5,
    ).chunk(Document("doc", "A. B. C."))
    assert chunks[0].source_segment_ids == ("doc:sentence:0", "doc:sentence:2")
    assert not chunks[0].is_contiguous
    assert chunks[1].source_segment_ids == ("doc:sentence:1",)


class FiveSentenceEmbedder(BaseEmbedder):
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.9999, 0.01],
            ],
            dtype=np.float32,
        )
        return vectors[: len(texts)]

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_documents(texts)


def test_single_linkage_uses_cluster_positions_after_a_noncontiguous_merge():
    chunks = SemanticSingleLinkageChunker(
        embedder=FiveSentenceEmbedder(),
        lambda_weight=0,
        target_clusters=2,
        distance_threshold=0.1,
    ).chunk(Document("doc", "A. B. C. D. E."))

    assert len(chunks) == 2
    assert any(
        chunk.source_segment_ids
        == ("doc:sentence:0", "doc:sentence:2", "doc:sentence:4")
        for chunk in chunks
    )
