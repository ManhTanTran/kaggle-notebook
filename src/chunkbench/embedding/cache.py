"""Stable cache-key construction without storing model objects in artifacts."""

import hashlib
import json
from typing import Any


def fingerprint(value: Any) -> str:
    """Hash JSON-compatible input deterministically for cache and manifest use."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_cache_key(
    *,
    method_name: str,
    method_config: dict[str, Any],
    model_name: str | None,
    tokenizer_name: str | None,
    model_revision: str | None,
    dataset_fingerprint: str,
    document_fingerprint: str | None,
    code_version: str | None,
    implementation_source_commit: str | None,
    precision: str | None,
    pooling: str | None,
    long_document_policy: str | None,
) -> str:
    """Include every setting that could invalidate a cached representation."""
    return fingerprint(
        {
            "method_name": method_name,
            "method_config": method_config,
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "model_revision": model_revision,
            "dataset_fingerprint": dataset_fingerprint,
            "document_fingerprint": document_fingerprint,
            "code_version": code_version,
            "implementation_source_commit": implementation_source_commit,
            "precision": precision,
            "pooling": pooling,
            "long_document_policy": long_document_policy,
        }
    )
