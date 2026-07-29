"""Deterministic lexical evidence coverage mapping."""

import re

from chunkbench.common.types import Chunk, Evidence, EvidenceCoverage

WORD_RE = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def map_evidence(
    chunks: list[Chunk], evidence: list[Evidence]
) -> list[EvidenceCoverage]:
    """Map evidence token positions covered lexically by each chunk."""
    rows = []
    for item in evidence:
        evidence_tokens = _tokens(item.text)
        for chunk in chunks:
            if chunk.document_id != item.document_id:
                continue
            chunk_tokens = set(_tokens(chunk.text))
            covered = frozenset(
                index
                for index, token in enumerate(evidence_tokens)
                if token in chunk_tokens
            )
            rows.append(
                EvidenceCoverage(
                    query_id=item.query_id,
                    chunk_id=chunk.chunk_id,
                    evidence_id=item.evidence_id,
                    covered_token_ids=covered,
                )
            )
    return rows
