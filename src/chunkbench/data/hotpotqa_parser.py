"""Parser for official HotpotQA question files."""

import json
from pathlib import Path
from typing import Any


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load a HotpotQA top-level list."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("HotpotQA questions root must be a list")
    return raw
