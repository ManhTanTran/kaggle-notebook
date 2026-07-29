"""ViMQA supporting-fact evidence construction."""

from chunkbench.common.types import Evidence


def make_evidence(
    query_id: str,
    evidence_index: int,
    document_id: str,
    title: str,
    sentence_index: int,
    sentences: list[str],
    spans: list[tuple[int, int]],
) -> Evidence:
    """Preserve each official sentence-level supporting fact separately."""
    if sentence_index < 0 or sentence_index >= len(sentences):
        raise ValueError(
            f"ViMQA supporting fact out of range: {title!r}, {sentence_index}"
        )
    text = sentences[sentence_index]
    start, end = spans[sentence_index]
    return Evidence(
        query_id,
        f"{query_id}:e{evidence_index}",
        document_id,
        text,
        len(text.split()),
        {
            "granularity": "sentence",
            "char_spans": [[start, end]],
            "sentence_ids": [sentence_index],
            "article_title": title,
            "raw_locator": [title, sentence_index],
        },
    )
