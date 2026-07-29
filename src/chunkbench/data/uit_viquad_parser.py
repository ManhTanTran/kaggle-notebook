"""Parser for the SQuAD-style UIT-ViQuAD schema."""

import json
from pathlib import Path
from typing import Any


def load_uit_viquad(path: Path) -> list[dict[str, Any]]:
    """Load the official SQuAD-style article list."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("data"), list):
        raise ValueError("UIT-ViQuAD root must contain a data list")
    return raw["data"]


def iter_paragraphs(
    articles: list[dict[str, Any]],
):
    """Yield article, paragraph, and stable source indices."""
    for article_index, article in enumerate(articles):
        for paragraph_index, paragraph in enumerate(article.get("paragraphs", [])):
            yield article_index, paragraph_index, article, paragraph
