"""Shared deterministic text and locator normalization."""

import re
import unicodedata
from pathlib import Path


def normalize_title(title: str) -> str:
    """Normalize an article title for stable, case-insensitive lookup."""
    normalized = unicodedata.normalize("NFC", title).replace("_", " ")
    return " ".join(normalized.split()).casefold()


def stable_id(prefix: str, value: str) -> str:
    """Build a readable stable identifier from source text."""
    slug = re.sub(r"[^\w]+", "-", normalize_title(value), flags=re.UNICODE).strip("-")
    return f"{prefix}:{slug}"


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return deterministic sentence character spans without changing text."""
    spans: list[tuple[int, int]] = []

    def append_span(start: int, end: int) -> None:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))

    start = 0
    index = 0
    while index < len(text):
        if text[index] not in ".!?":
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in ".!?":
            end += 1
        if end < len(text) and not text[end].isspace():
            index = end
            continue
        punctuation = text[index:end]
        if punctuation.startswith(".") and len(punctuation) > 1:
            index = end
            continue
        append_span(start, end)
        start = end
        index = end
    append_span(start, len(text))
    return spans


def resolve_split_file(path: str | Path, split: str) -> Path:
    """Resolve a configured file or a conventional split file in a directory."""
    source = Path(path)
    if source.is_file():
        return source
    candidates = [f"{split}.json"]
    if split == "validation":
        candidates.extend(["dev.json", "validation.json"])
    for candidate in candidates:
        resolved = source / candidate
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"No data file for split {split!r} under {source}; tried {candidates}"
    )
