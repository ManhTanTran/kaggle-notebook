from chunkbench.chunking.meta_ppl import MetaPPLChunker
from chunkbench.common.types import Document
from chunkbench.scoring.perplexity import BasePerplexityScorer


class FakePerplexityScorer(BasePerplexityScorer):
    def score_transition(self, left_context: str, candidate_segment: str) -> float:
        return {"A.": 1.0, "B.": 0.1, "C.": 1.0}.get(candidate_segment, 1.0)


class RecordingPerplexityScorer(BasePerplexityScorer):
    def __init__(self) -> None:
        self.contexts: list[str] = []

    def score_transition(self, left_context: str, candidate_segment: str) -> float:
        self.contexts.append(left_context)
        return 1.0


def test_meta_ppl_uses_expected_local_minimum_boundary():
    chunks = MetaPPLChunker(
        "meta_ppl_raw", scorer=FakePerplexityScorer(), prominence=0.5
    ).chunk(Document("doc", "A. B. C."))
    assert [chunk.text for chunk in chunks] == ["A. B.", "C."]
    assert chunks[0].metadata["boundary_after"] == [1]


def test_dynamic_meta_ppl_enforces_declared_cap():
    chunks = MetaPPLChunker(
        "meta_ppl_dynamic_512", scorer=FakePerplexityScorer(), max_chunk_tokens=2
    ).chunk(Document("doc", "one two. three four. five six."))
    assert [chunk.token_count for chunk in chunks] == [2, 2, 2]


def test_meta_ppl_supports_bounded_previous_segment_context_policy():
    scorer = RecordingPerplexityScorer()
    MetaPPLChunker(
        "meta_ppl_raw", scorer=scorer, context_policy="previous_segment"
    ).chunk(Document("doc", "A. B. C."))
    assert scorer.contexts == ["", "A.", "B."]
