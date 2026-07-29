from chunkbench.common.types import Chunk, RetrievalHit
from chunkbench.eval.token_budget import select_within_token_budget


def test_strict_budget():
    chunks = [
        Chunk("a", "d", "a a", 2, 0, True, ("s",), ((0, 3),)),
        Chunk("b", "d", "b b", 2, 1, True, ("s",), ((4, 7),)),
        Chunk("c", "d", "c", 1, 2, True, ("s",), ((8, 9),)),
    ]
    hits = [
        RetrievalHit("q", "a", "d", 1, 1.0),
        RetrievalHit("q", "b", "d", 2, 0.9),
        RetrievalHit("q", "c", "d", 3, 0.8),
    ]
    selected = select_within_token_budget(hits, chunks, 3, strict=True)
    assert [hit.chunk_id for hit in selected] == ["a"]
