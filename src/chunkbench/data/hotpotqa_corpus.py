"""Streaming reader for processed HotpotQA Wikipedia articles."""

import bz2
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from chunkbench.data.normalization import normalize_wikipedia_title


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_articles(path: Path) -> Iterator[dict[str, Any]]:
    """Stream articles from JSON, JSONL, BZ2, or a directory of those files."""
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix in {".json", ".jsonl", ".bz2"}:
                yield from iter_articles(child)
        return
    if path.suffix == ".json":
        with _open_text(path) as handle:
            raw = json.load(handle)
        if isinstance(raw, list):
            yield from raw
        elif isinstance(raw, dict):
            yield raw
        else:
            raise ValueError(f"Unsupported corpus JSON root in {path}")
        return
    with _open_text(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def lookup_articles(
    path: Path, normalized_titles: set[str]
) -> dict[str, dict[str, Any]]:
    """Stream until all requested normalized titles have been found."""
    found: dict[str, dict[str, Any]] = {}
    for article in iter_articles(path):
        key = normalize_wikipedia_title(str(article["title"]))
        if key in normalized_titles and key not in found:
            found[key] = article
            if len(found) == len(normalized_titles):
                break
    return found


def normalize_article(
    article: dict[str, Any],
) -> tuple[str, list[str], list[dict[str, int]]]:
    """Join the official paragraph/sentence hierarchy with exact locators."""
    sentences: list[str] = []
    for paragraph in article.get("text", []):
        if isinstance(paragraph, list):
            sentences.extend(str(sentence) for sentence in paragraph)
        else:
            sentences.append(str(paragraph))
    parts: list[str] = []
    locators: list[dict[str, int]] = []
    for index, sentence in enumerate(sentences):
        if parts:
            parts.append(" ")
        start = sum(len(part) for part in parts)
        parts.append(sentence)
        locators.append(
            {
                "sentence_index": index,
                "start": start,
                "end": start + len(sentence),
            }
        )
    return "".join(parts), sentences, locators
