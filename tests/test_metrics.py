from chunkbench.chunking.base import BaseChunker
from chunkbench.common.types import (
    Chunk,
    DatasetBundle,
    Document,
    Evidence,
    EvidenceCoverage,
    Query,
    RetrievalHit,
)
from chunkbench.eval.constants import PRIMARY_METRICS
from chunkbench.eval.evaluator import evaluate
from chunkbench.eval.metrics import (
    document_recall_at_k,
    evidence_coverage_at_k,
    evidence_recall_macro_at_k,
    evidence_recall_micro_at_k,
    hit_at_k,
    mrr_at_k,
)
from chunkbench.eval.redundancy import redundancy_at_k
from chunkbench.evidence.overlap import map_evidence


def _fixture():
    bundle = DatasetBundle(
        [
            Document("d1", "alpha beta gamma"),
            Document("d2", "delta epsilon zeta"),
        ],
        [
            Query("q1", "alpha?", ("d1",)),
            Query("q2", "delta?", ("d2",)),
        ],
        [
            Evidence("q1", "e1", "d1", "alpha beta", 2),
            Evidence("q2", "e2", "d2", "delta epsilon", 2),
        ],
    )
    chunks = [
        Chunk("c1", "d1", "alpha beta", 2, 0, True, ("s1",), ((0, 10),)),
        Chunk("c2", "d1", "gamma", 1, 1, True, ("s2",), ((11, 16),)),
        Chunk("c3", "d2", "delta", 1, 0, True, ("s3",), ((0, 5),)),
        Chunk("c4", "d2", "epsilon", 1, 1, True, ("s4",), ((6, 13),)),
    ]
    hits = [
        RetrievalHit("q1", "c1", "d1", 1, 1.0),
        RetrievalHit("q1", "c2", "d1", 2, 0.1),
        RetrievalHit("q2", "c2", "d1", 1, 0.9),
        RetrievalHit("q2", "c3", "d2", 2, 0.8),
        RetrievalHit("q2", "c4", "d2", 3, 0.7),
    ]
    return bundle, chunks, hits, map_evidence(chunks, bundle.evidence)


def test_primary_metric_count():
    assert len(PRIMARY_METRICS) == 23


def test_expected_metric_values_and_union_coverage():
    bundle, chunks, hits, coverage = _fixture()
    assert hit_at_k(bundle, hits, coverage, 1) == 0.5
    assert mrr_at_k(bundle, hits, coverage, 1) == 0.5
    assert evidence_recall_macro_at_k(bundle, hits, coverage, 1) == 0.5
    assert evidence_recall_micro_at_k(bundle, hits, coverage, 1) == 0.5
    assert document_recall_at_k(bundle, hits, 1) == 0.5
    assert evidence_coverage_at_k(bundle, hits, coverage, 1) == 0.5
    assert evidence_coverage_at_k(bundle, hits, coverage, 3) == 1.0
    assert evidence_recall_micro_at_k(bundle, hits, coverage, 3) == 1.0
    assert redundancy_at_k(chunks, hits, 3) == 0.0


def test_evidence_coverage_deduplicates_token_ids():
    bundle = DatasetBundle(
        [Document("d", "alpha beta")],
        [Query("q", "alpha?", ("d",))],
        [Evidence("q", "e", "d", "alpha beta", 2)],
    )
    hits = [
        RetrievalHit("q", "c1", "d", 1, 1.0),
        RetrievalHit("q", "c2", "d", 2, 0.9),
        RetrievalHit("q", "c3", "d", 3, 0.8),
    ]
    coverage = [
        EvidenceCoverage("q", "c1", "e", frozenset({0})),
        EvidenceCoverage("q", "c2", "e", frozenset({0})),
        EvidenceCoverage("q", "c3", "e", frozenset({1})),
    ]
    assert evidence_coverage_at_k(bundle, hits, coverage, 3) == 1.0


def test_evaluator_returns_all_primary_metrics():
    bundle, chunks, hits, coverage = _fixture()
    metrics = evaluate(bundle, chunks, hits, coverage)
    assert set(metrics) == set(PRIMARY_METRICS)


def test_registry_extension_does_not_require_runner_changes():
    class CustomChunker(BaseChunker):
        def chunk(self, document):
            return []

    from chunkbench.registry.methods import (
        METHOD_REGISTRY,
        build_chunker,
        register_method,
    )

    register_method("custom_audit_method", lambda config: CustomChunker())
    try:
        assert isinstance(build_chunker("custom_audit_method"), CustomChunker)
    finally:
        METHOD_REGISTRY.pop("custom_audit_method")
