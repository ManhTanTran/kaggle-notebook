"""Parser for the official QASPER JSON structure."""

import json
from pathlib import Path
from typing import Any


def load_qasper(path: Path) -> dict[str, dict[str, Any]]:
    """Load the official paper-id keyed QASPER mapping."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("QASPER root must be a paper-id keyed object")
    return raw


def build_paper_text(
    paper: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Join abstract, sections, and paragraphs in their source order."""
    parts: list[str] = []
    locators: list[dict[str, Any]] = []

    def append(text: str, section_name: str, paragraph_index: int) -> None:
        if not text.strip():
            return
        if parts:
            parts.append("\n\n")
        start = sum(len(part) for part in parts)
        parts.append(text)
        locators.append(
            {
                "section_name": section_name,
                "paragraph_index": paragraph_index,
                "char_span": [start, start + len(text)],
            }
        )

    for index, paragraph in enumerate(paper.get("abstract", [])):
        append(str(paragraph), "Abstract", index)
    for section in paper.get("full_text", []):
        section_name = str(section.get("section_name", ""))
        for index, paragraph in enumerate(section.get("paragraphs", [])):
            append(str(paragraph), section_name, index)
    return "".join(parts), locators
