"""Parser for the official ViMQA HotpotQA-like schema."""

import json
from pathlib import Path
from typing import Any


def load_vimqa(path: Path) -> list[dict[str, Any]]:
    """Load official ViMQA examples after checking required evidence fields."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("ViMQA root must be a list")
    for item in raw:
        missing = {"_id", "question", "context", "supporting_facts"} - set(item)
        if missing:
            raise ValueError(
                "ViMQA schema is incompatible with evidence retrieval; "
                f"missing fields {sorted(missing)} for item {item.get('_id')}"
            )
    return raw


def context_documents(
    items: list[dict[str, Any]],
) -> dict[str, tuple[str, list[str], list[tuple[int, int]]]]:
    """Collect title-keyed context while preserving Vietnamese sentence text."""
    documents: dict[str, tuple[str, list[str], list[tuple[int, int]]]] = {}
    for item in items:
        for title, raw_sentences in item["context"]:
            sentences = [str(sentence) for sentence in raw_sentences]
            parts: list[str] = []
            spans: list[tuple[int, int]] = []
            for sentence in sentences:
                if parts:
                    parts.append(" ")
                start = sum(len(part) for part in parts)
                parts.append(sentence)
                spans.append((start, start + len(sentence)))
            existing = documents.get(str(title))
            value = ("".join(parts), sentences, spans)
            if existing is not None and existing != value:
                raise ValueError(f"Conflicting ViMQA context for title {title!r}")
            documents[str(title)] = value
    return documents
