from chunkbench.chunking.sentence_fixed import SentenceFixedChunker
from chunkbench.common.types import Document


def test_sentence_fixed_chunker_respects_sentences():
    chunks = SentenceFixedChunker(4).chunk(Document("d", "one two. three four. five."))
    assert chunks
    assert all(chunk.token_count <= 4 for chunk in chunks)
