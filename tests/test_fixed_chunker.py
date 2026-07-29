from chunkbench.chunking.fixed import FixedTokenChunker
from chunkbench.common.types import Document


def test_fixed_chunker():
    chunks = FixedTokenChunker(3).chunk(Document("d", "a b c d e"))
    assert [chunk.text for chunk in chunks] == ["a b c", "d e"]
    assert all(chunk.is_contiguous for chunk in chunks)
