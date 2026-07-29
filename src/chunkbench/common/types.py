"""Canonical immutable-ish data contracts used across the benchmark."""

from dataclasses import dataclass, field
from typing import Any

Metadata = dict[str, Any]
SourceSpan = tuple[int, int]


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    relevant_document_ids: tuple[str, ...]
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    query_id: str
    evidence_id: str
    document_id: str
    text: str
    token_count: int
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    token_count: int
    chunk_order: int
    is_contiguous: bool
    source_segment_ids: tuple[str, ...]
    source_spans: tuple[SourceSpan, ...]
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    query_id: str
    chunk_id: str
    document_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class EvidenceCoverage:
    query_id: str
    chunk_id: str
    evidence_id: str
    covered_token_ids: frozenset[int]


@dataclass
class DatasetBundle:
    documents: list[Document]
    queries: list[Query]
    evidence: list[Evidence]
    metadata: Metadata = field(default_factory=dict)
