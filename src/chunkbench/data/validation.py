"""Dataset contract validation and evidence-mapping diagnostics."""

from collections import Counter
from typing import Any

from chunkbench.common.exceptions import ContractError
from chunkbench.common.types import DatasetBundle, Evidence


def _span_mapping_succeeds(
    evidence: Evidence, document_text: str
) -> tuple[bool, str | None]:
    spans = evidence.metadata.get("char_spans", [])
    if not spans:
        return evidence.text in document_text, None
    for raw_span in spans:
        if not isinstance(raw_span, (list, tuple)) or len(raw_span) != 2:
            return False, f"invalid char span structure: {raw_span!r}"
        start, end = int(raw_span[0]), int(raw_span[1])
        if start < 0 or end <= start or end > len(document_text):
            return False, f"char span out of bounds: {(start, end)}"
        if document_text[start:end] != evidence.text:
            return False, f"char span text mismatch: {(start, end)}"
    return True, None


def dataset_diagnostics(bundle: DatasetBundle) -> dict[str, Any]:
    """Compute validation counts without silently discarding bad mappings."""
    document_ids = [item.document_id for item in bundle.documents]
    query_ids = [item.query_id for item in bundle.queries]
    documents = {item.document_id: item.text for item in bundle.documents}
    mapping_success = 0
    mapping_failure = 0
    mapping_errors: list[dict[str, str]] = []
    for evidence in bundle.evidence:
        text = documents.get(evidence.document_id)
        if text is None:
            mapping_failure += 1
            mapping_errors.append(
                {
                    "query_id": evidence.query_id,
                    "evidence_id": evidence.evidence_id,
                    "reason": "unknown document",
                }
            )
            continue
        success, reason = _span_mapping_succeeds(evidence, text)
        if success:
            mapping_success += 1
        else:
            mapping_failure += 1
            mapping_errors.append(
                {
                    "query_id": evidence.query_id,
                    "evidence_id": evidence.evidence_id,
                    "reason": reason or "text not found in document",
                }
            )
    total = mapping_success + mapping_failure
    known_documents = set(document_ids)
    missing_relevant = sum(
        doc_id not in known_documents
        for query in bundle.queries
        for doc_id in query.relevant_document_ids
    )
    return {
        "document_count": len(bundle.documents),
        "query_count": len(bundle.queries),
        "evidence_count": len(bundle.evidence),
        "unanswerable_query_count": (
            sum(bool(query.metadata.get("unanswerable")) for query in bundle.queries)
            + int(bundle.metadata.get("excluded_unanswerable_count", 0))
        ),
        "evidence_mapping_success_count": mapping_success,
        "evidence_mapping_failure_count": mapping_failure,
        "evidence_mapping_rate": mapping_success / total if total else 1.0,
        "missing_relevant_document_count": missing_relevant,
        "duplicate_document_count": sum(
            count - 1 for count in Counter(document_ids).values() if count > 1
        ),
        "duplicate_query_count": sum(
            count - 1 for count in Counter(query_ids).values() if count > 1
        ),
        "mapping_errors": mapping_errors,
    }


def validate_dataset(
    bundle: DatasetBundle, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate identifiers, references, text, spans, and mapping quality."""
    validation = config or {}
    report = dataset_diagnostics(bundle)
    if report["duplicate_document_count"]:
        raise ContractError("Duplicate document identifiers")
    if report["duplicate_query_count"]:
        raise ContractError("Duplicate query identifiers")
    if any(not document.text.strip() for document in bundle.documents):
        raise ContractError("Documents must have non-empty text")
    if any(not query.text.strip() for query in bundle.queries):
        raise ContractError("Queries must have non-empty text")
    known_documents = {document.document_id for document in bundle.documents}
    known_queries = {query.query_id for query in bundle.queries}
    evidence_ids: dict[str, set[str]] = {}
    for query in bundle.queries:
        if not set(query.relevant_document_ids) <= known_documents:
            raise ContractError(f"Query {query.query_id} references unknown document")
    for item in bundle.evidence:
        query_evidence_ids = evidence_ids.setdefault(item.query_id, set())
        if item.evidence_id in query_evidence_ids:
            raise ContractError(
                f"Duplicate evidence id {item.evidence_id!r} in query {item.query_id}"
            )
        query_evidence_ids.add(item.evidence_id)
        if not item.text.strip() or item.token_count <= 0:
            raise ContractError(f"Evidence {item.evidence_id} must contain tokens")
        if (
            item.document_id not in known_documents
            or item.query_id not in known_queries
        ):
            raise ContractError(f"Evidence {item.evidence_id} has invalid references")
        if item.token_count != len(item.text.split()):
            raise ContractError(
                f"Evidence {item.evidence_id} token_count is inconsistent"
            )
    rate = float(report["evidence_mapping_rate"])
    minimum = float(validation.get("minimum_mapping_rate", 0.95))
    fail_on_error = bool(validation.get("fail_on_mapping_error", True))
    if (fail_on_error and report["evidence_mapping_failure_count"]) or rate < minimum:
        raise ContractError(
            "Evidence mapping validation failed: "
            f"rate={rate:.3f}, minimum={minimum:.3f}, "
            f"errors={report['mapping_errors']}"
        )
    return report
