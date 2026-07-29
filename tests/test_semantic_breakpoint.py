import numpy as np

from chunkbench.chunking.semantic_breakpoint import SemanticBreakpointChunker
from chunkbench.common.types import Document
from chunkbench.embedding.base import BaseEmbedder


class OrderedEmbedder(BaseEmbedder):
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1, 0], [1, 0], [0, 1]][: len(texts)], dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_documents(texts)


def test_breakpoint_has_expected_boundary_and_preserves_segments():
    chunks = SemanticBreakpointChunker(
        embedder=OrderedEmbedder(),
        threshold={"type": "absolute", "value": 0.5},
        min_chunk_tokens=0,
    ).chunk(Document("doc", "A. B. C."))
    assert [chunk.text for chunk in chunks] == ["A. B.", "C."]
    assert chunks[0].metadata["selected_boundary_positions"] == [1]
    assert [item for chunk in chunks for item in chunk.source_segment_ids] == [
        "doc:sentence:0",
        "doc:sentence:1",
        "doc:sentence:2",
    ]
