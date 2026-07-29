"""QASPER annotated-evidence extraction."""

from typing import Any


def annotated_evidence(question: dict[str, Any]) -> tuple[list[str], bool]:
    """Return unique annotated evidence strings and unanswerable status."""
    evidence: list[str] = []
    answers = question.get("answers", [])
    unanswerable = bool(answers) and all(
        bool(annotation.get("answer", {}).get("unanswerable")) for annotation in answers
    )
    for annotation in answers:
        answer = annotation.get("answer", {})
        for raw_text in answer.get("evidence", []):
            text = str(raw_text)
            if text.strip() and text not in evidence:
                evidence.append(text)
    return evidence, unanswerable


def locate_evidence(
    text: str, evidence_text: str, locators: list[dict[str, Any]]
) -> dict[str, Any]:
    """Locate exact annotated evidence inside a normalized paper."""
    start = text.find(evidence_text)
    metadata: dict[str, Any] = {
        "granularity": "paragraph",
        "raw_locator": {"evidence_text": evidence_text},
        "char_spans": [],
    }
    if start < 0:
        metadata["mapping_error"] = "annotated evidence not found verbatim"
        return metadata
    end = start + len(evidence_text)
    metadata["char_spans"] = [[start, end]]
    for locator in locators:
        span_start, span_end = locator["char_span"]
        if span_start <= start and end <= span_end:
            metadata["section_name"] = locator["section_name"]
            metadata["paragraph_index"] = locator["paragraph_index"]
            break
    return metadata
