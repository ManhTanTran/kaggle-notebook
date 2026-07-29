"""HotpotQA supporting-fact normalization."""

from chunkbench.common.types import Evidence


def supporting_evidence(
    query_id: str,
    document_id: str,
    title: str,
    sentence_index: int,
    sentences: list[str],
    locators: list[dict[str, int]],
    evidence_index: int,
) -> Evidence:
    """Create one evidence unit per official supporting fact."""
    if sentence_index < 0 or sentence_index >= len(sentences):
        raise ValueError(
            f"Supporting fact out of range for {query_id}: "
            f"{title!r} sentence {sentence_index}"
        )
    text = sentences[sentence_index]
    locator = locators[sentence_index]
    return Evidence(
        query_id=query_id,
        evidence_id=f"{query_id}:e{evidence_index}",
        document_id=document_id,
        text=text,
        token_count=len(text.split()),
        metadata={
            "granularity": "sentence",
            "char_spans": [[locator["start"], locator["end"]]],
            "sentence_ids": [sentence_index],
            "article_title": title,
            "raw_locator": [title, sentence_index],
        },
    )
