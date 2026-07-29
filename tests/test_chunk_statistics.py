from chunkbench.chunking.fixed import FixedTokenChunker
from chunkbench.common.types import Document
from chunkbench.eval.chunk_statistics import chunk_statistics


def test_chunk_statistics():
    chunks = FixedTokenChunker(2).chunk(Document("d", "a b c"))
    stats = chunk_statistics(chunks)
    assert stats["Total Chunks"] == 2
    assert "Percentage of Chunks > 256" in stats
