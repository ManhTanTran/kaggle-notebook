import numpy as np

from chunkbench.chunking.pic import BaseSummarizer, PICChunker
from chunkbench.common.types import Document
from chunkbench.embedding.base import BaseEmbedder


class FakeSummary(BaseSummarizer):
    def summarize(self, document: Document) -> str:
        return "theme"


class PICEmbedder(BaseEmbedder):
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        values = {"theme": [1, 0], "A.": [1, 0], "B.": [1, 0], "C.": [0, 1]}
        return np.asarray([values[text] for text in texts], dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_documents(texts)


def test_pic_groups_runs_above_and_below_mean_similarity():
    chunks = PICChunker(
        "pic_paper_reimplementation", embedder=PICEmbedder(), summarizer=FakeSummary()
    ).chunk(Document("doc", "A. B. C."))
    assert [chunk.text for chunk in chunks] == ["A. B.", "C."]
    assert chunks[0].metadata["threshold"] == 2 / 3
