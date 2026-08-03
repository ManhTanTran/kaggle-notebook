"""Sparse, dense, combined, and hierarchical retrieval components."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .config import RunConfig

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def _result(
    item_id: str,
    score: float,
    source_method: str,
    sparse_rank: int | None = None,
    dense_rank: int | None = None,
) -> dict[str, Any]:
    return {
        "item_id": str(item_id),
        "score": float(score),
        "sparse_rank": sparse_rank,
        "dense_rank": dense_rank,
        "source_method": source_method,
    }


class BM25Index:
    """Small dependency-free Okapi BM25 implementation with configurable k1/b."""

    def __init__(
        self, ids: Sequence[str], texts: Sequence[str], k1: float = 1.2, b: float = 0.75
    ):
        if len(ids) != len(texts) or not ids:
            raise ValueError("BM25 requires equally sized, non-empty ids and texts")
        self.ids = np.asarray([str(value) for value in ids], dtype=object)
        self.k1 = float(k1)
        self.b = float(b)
        self.term_frequencies: list[Counter[str]] = [
            Counter(tokenize(text)) for text in texts
        ]
        self.lengths = np.asarray(
            [sum(freq.values()) for freq in self.term_frequencies], dtype=float
        )
        self.avg_length = float(self.lengths.mean()) if len(self.lengths) else 0.0
        document_frequency: Counter[str] = Counter()
        for freq in self.term_frequencies:
            document_frequency.update(freq.keys())
        n = len(self.ids)
        self.idf = {
            term: math.log(1.0 + (n - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(
        self, query: str, allowed_indices: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = (
            np.arange(len(self.ids))
            if allowed_indices is None
            else np.asarray(allowed_indices, dtype=int)
        )
        scores = np.zeros(len(indices), dtype=float)
        query_terms = tokenize(query)
        for output_position, corpus_position in enumerate(indices):
            frequencies = self.term_frequencies[int(corpus_position)]
            length = self.lengths[int(corpus_position)]
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                norm = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / max(self.avg_length, 1e-12)
                )
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / norm
            scores[output_position] = score
        return indices, scores

    def search(
        self, query: str, k: int, allowed_indices: np.ndarray | None = None
    ) -> list[dict[str, Any]]:
        indices, scores = self.scores(query, allowed_indices)
        order = sorted(
            range(len(indices)), key=lambda i: (-scores[i], str(self.ids[indices[i]]))
        )[:k]
        return [
            _result(
                self.ids[indices[position]],
                scores[position],
                "sparse",
                sparse_rank=rank,
            )
            for rank, position in enumerate(order, start=1)
        ]

    @property
    def storage_bytes(self) -> int:
        term_bytes = sum(
            len(term.encode("utf-8")) + 8
            for freq in self.term_frequencies
            for term in freq
        )
        return int(self.lengths.nbytes + term_bytes)


class HashingDualEncoder:
    """Deterministic smoke-only encoder; never a benchmark substitute for DRAGON+."""

    def __init__(self, dimensions: int = 512):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.cache_key = f"hashing:{dimensions}:v1"

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                matrix[row, value % self.dimensions] += (
                    1.0 if (value >> 8) % 2 == 0 else -1.0
                )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.where(norms == 0.0, 1.0, norms)
        return matrix

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_contexts(
        self, texts: Sequence[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return self._encode(texts), {
            "truncation_rate": 0.0,
            "tokenizer_max_length": None,
        }


class TransformerDualEncoder:
    """Asymmetric Hugging Face encoder using CLS vectors and tokenizer truncation."""

    def __init__(
        self,
        query_model: str,
        context_model: str,
        query_revision: str,
        context_revision: str,
        batch_size: int,
        device: str,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Transformer dense retrieval requires torch and transformers. "
                "Install the project's dense optional dependencies."
            ) from exc
        self.torch = torch
        self.batch_size = batch_size
        self.device = device
        self.query_model_id = query_model
        self.context_model_id = context_model
        self.query_revision = query_revision
        self.context_revision = context_revision
        self.cache_key = (
            f"transformers:{query_model}@{query_revision}:"
            f"{context_model}@{context_revision}:cls:v1"
        )
        try:
            self.query_tokenizer = AutoTokenizer.from_pretrained(
                query_model, revision=query_revision
            )
            self.context_tokenizer = AutoTokenizer.from_pretrained(
                context_model, revision=context_revision
            )
            self.query_model = (
                AutoModel.from_pretrained(query_model, revision=query_revision)
                .to(device)
                .eval()
            )
            self.context_model = (
                AutoModel.from_pretrained(context_model, revision=context_revision)
                .to(device)
                .eval()
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load the configured dense checkpoints. Check network/cache "
                "access and "
                f"model IDs query={query_model!r}, context={context_model!r}."
            ) from exc

    def _encode(
        self, texts: Sequence[str], tokenizer: Any, model: Any
    ) -> tuple[np.ndarray, dict[str, Any]]:
        torch = self.torch
        max_length = min(int(getattr(tokenizer, "model_max_length", 512)), 8192)
        encoded_batches: list[np.ndarray] = []
        truncated = 0
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            token_lengths = [
                len(
                    tokenizer(text, add_special_tokens=True, truncation=False)[
                        "input_ids"
                    ]
                )
                for text in batch
            ]
            truncated += sum(length > max_length for length in token_lengths)
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state[:, 0, :]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            encoded_batches.append(embeddings.cpu().numpy().astype(np.float32))
        matrix = (
            np.vstack(encoded_batches)
            if encoded_batches
            else np.empty((0, 0), dtype=np.float32)
        )
        return matrix, {
            "truncation_rate": float(truncated / len(texts)) if texts else 0.0,
            "tokenizer_max_length": max_length,
            "query_model": self.query_model_id,
            "query_revision": self.query_revision,
            "context_model": self.context_model_id,
            "context_revision": self.context_revision,
        }

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.query_tokenizer, self.query_model)[0]

    def encode_contexts(
        self, texts: Sequence[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return self._encode(texts, self.context_tokenizer, self.context_model)


def make_dense_encoder(
    config: RunConfig,
) -> HashingDualEncoder | TransformerDualEncoder:
    if config.dense_backend == "hashing":
        return HashingDualEncoder(config.hashing_features)
    return TransformerDualEncoder(
        config.dense_query_model,
        config.dense_context_model,
        config.dense_query_revision,
        config.dense_context_revision,
        config.dense_batch_size,
        config.dense_device,
    )


def document_representation(row: pd.Series, strategy: str, lead_chars: int) -> str:
    title = "" if pd.isna(row["title"]) else str(row["title"]).strip()
    lead = str(row["text"])[:lead_chars]
    if strategy == "title_plus_lead":
        return " ".join(part for part in (title, lead) if part)
    if strategy == "title_only":
        return title
    if strategy == "lead_only":
        return lead
    raise ValueError(f"Unknown document representation strategy: {strategy}")


class SparseDocumentRetriever:
    method = "sparse"

    def __init__(self, documents: pd.DataFrame, config: RunConfig):
        self.index, self.cache_path, self.cache_hit = _load_or_build_bm25(
            ids=documents["document_id"].astype(str).tolist(),
            texts=documents["text"].astype(str).tolist(),
            cache_dir=config.cache_dir,
            namespace="documents",
            rebuild=config.rebuild_indices,
            k1=config.bm25_k1,
            b=config.bm25_b,
        )

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        return self.index.search(query, k)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "backend": "native_bm25",
            "storage_bytes": int(self.cache_path.stat().st_size),
            "cache_path": str(self.cache_path),
            "cache_hit": self.cache_hit,
        }


class DenseDocumentRetriever:
    method = "dense"

    def __init__(
        self,
        documents: pd.DataFrame,
        config: RunConfig,
        encoder: Any,
        cache_namespace: str,
    ):
        self.ids = documents["document_id"].astype(str).to_numpy()
        texts = [
            document_representation(
                row, config.dense_document_strategy, config.dense_document_lead_chars
            )
            for _, row in documents.iterrows()
        ]
        self.encoder = encoder
        self.config = config
        self.embeddings, encoding_metadata, self.cache_path = _load_or_encode(
            ids=self.ids,
            texts=texts,
            encoder=encoder,
            cache_dir=config.cache_dir,
            namespace=f"{cache_namespace}_documents_{config.dense_document_strategy}",
            rebuild=config.rebuild_indices,
        )
        titles = documents["title"]
        title_text = titles.fillna("").astype(str).str.strip()
        self._metadata = {
            "method": self.method,
            "backend": config.dense_backend,
            "index": config.dense_index,
            "storage_bytes": int(self.embeddings.nbytes),
            "cache_path": str(self.cache_path),
            "missing_title_rate": float((titles.isna() | title_text.eq("")).mean()),
            "noisy_title_rate": float(title_text.str.len().gt(300).mean()),
            **encoding_metadata,
        }
        self.faiss_index = None
        if config.dense_index == "faiss":
            try:
                import faiss
            except ImportError as exc:
                raise ImportError("dense_index='faiss' requires faiss-cpu") from exc
            faiss_path = self.cache_path.with_suffix(".faiss")
            if faiss_path.is_file() and not config.rebuild_indices:
                self.faiss_index = faiss.read_index(str(faiss_path))
                self._metadata["faiss_cache_hit"] = True
            else:
                self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
                self.faiss_index.add(
                    np.ascontiguousarray(self.embeddings, dtype=np.float32)
                )
                faiss.write_index(self.faiss_index, str(faiss_path))
                self._metadata["faiss_cache_hit"] = False
            self._metadata["faiss_cache_path"] = str(faiss_path)
            self._metadata["storage_bytes"] += int(faiss_path.stat().st_size)

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        query_vector = self.encoder.encode_queries([query]).astype(np.float32)
        k = min(k, len(self.ids))
        if self.faiss_index is not None:
            scores, positions = self.faiss_index.search(query_vector, k)
            pairs = list(zip(positions[0].tolist(), scores[0].tolist(), strict=True))
        else:
            scores = self.embeddings @ query_vector[0]
            order = sorted(
                range(len(scores)), key=lambda i: (-float(scores[i]), str(self.ids[i]))
            )[:k]
            pairs = [(i, float(scores[i])) for i in order]
        return [
            _result(self.ids[position], score, "dense", dense_rank=rank)
            for rank, (position, score) in enumerate(pairs, start=1)
        ]

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)


class SparsePassageRetriever:
    method = "sparse"

    def __init__(self, passages: pd.DataFrame, config: RunConfig):
        self.passages = passages.reset_index(drop=True)
        self.index, self.cache_path, self.cache_hit = _load_or_build_bm25(
            ids=self.passages["passage_id"].astype(str).tolist(),
            texts=self.passages["passage_text"].astype(str).tolist(),
            cache_dir=config.cache_dir,
            namespace="passages",
            rebuild=config.rebuild_indices,
            k1=config.bm25_k1,
            b=config.bm25_b,
        )

    def retrieve(
        self, query: str, k: int, document_ids: set[str]
    ) -> list[dict[str, Any]]:
        allowed = np.flatnonzero(
            self.passages["document_id"].astype(str).isin(document_ids).to_numpy()
        )
        return self.index.search(query, k, allowed)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "backend": "native_bm25",
            "storage_bytes": int(self.cache_path.stat().st_size),
            "cache_path": str(self.cache_path),
            "cache_hit": self.cache_hit,
        }


class DensePassageRetriever:
    method = "dense"

    def __init__(
        self,
        passages: pd.DataFrame,
        config: RunConfig,
        encoder: Any,
        cache_namespace: str,
    ):
        self.passages = passages.reset_index(drop=True)
        self.ids = self.passages["passage_id"].astype(str).to_numpy()
        self.encoder = encoder
        self.embeddings, encoding_metadata, self.cache_path = _load_or_encode(
            ids=self.ids,
            texts=self.passages["passage_text"].astype(str).tolist(),
            encoder=encoder,
            cache_dir=config.cache_dir,
            namespace=f"{cache_namespace}_passages",
            rebuild=config.rebuild_indices,
        )
        self._metadata = {
            "method": self.method,
            "backend": config.dense_backend,
            "storage_bytes": int(self.embeddings.nbytes),
            "cache_path": str(self.cache_path),
            **encoding_metadata,
        }

    def retrieve(
        self, query: str, k: int, document_ids: set[str]
    ) -> list[dict[str, Any]]:
        allowed = np.flatnonzero(
            self.passages["document_id"].astype(str).isin(document_ids).to_numpy()
        )
        if not len(allowed):
            return []
        query_vector = self.encoder.encode_queries([query])[0]
        scores = self.embeddings[allowed] @ query_vector
        order = sorted(
            range(len(allowed)),
            key=lambda i: (-float(scores[i]), str(self.ids[allowed[i]])),
        )[:k]
        return [
            _result(
                self.ids[allowed[position]], scores[position], "dense", dense_rank=rank
            )
            for rank, position in enumerate(order, start=1)
        ]

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)


def _load_or_encode(
    ids: Sequence[str],
    texts: Sequence[str],
    encoder: Any,
    cache_dir: Path,
    namespace: str,
    rebuild: bool,
) -> tuple[np.ndarray, dict[str, Any], Path]:
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256()
    for item_id, text in zip(ids, texts, strict=True):
        fingerprint.update(str(item_id).encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(str(text).encode("utf-8"))
        fingerprint.update(b"\0")
    fingerprint.update(
        str(getattr(encoder, "cache_key", type(encoder).__name__)).encode("utf-8")
    )
    stem = f"{namespace}_{fingerprint.hexdigest()[:16]}"
    matrix_path = cache_dir / f"{stem}.npy"
    metadata_path = cache_dir / f"{stem}.json"
    if matrix_path.is_file() and metadata_path.is_file() and not rebuild:
        matrix = np.load(matrix_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["cache_hit"] = True
        return matrix, metadata, matrix_path
    matrix, metadata = encoder.encode_contexts(texts)
    np.save(matrix_path, matrix)
    metadata = {**metadata, "cache_hit": False}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return matrix, metadata, matrix_path


def _load_or_build_bm25(
    ids: Sequence[str],
    texts: Sequence[str],
    cache_dir: Path,
    namespace: str,
    rebuild: bool,
    k1: float,
    b: float,
) -> tuple[BM25Index, Path, bool]:
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256()
    for item_id, text in zip(ids, texts, strict=True):
        fingerprint.update(str(item_id).encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(str(text).encode("utf-8"))
        fingerprint.update(b"\0")
    fingerprint.update(f"native_bm25:{k1}:{b}:v1".encode("ascii"))
    path = cache_dir / f"{namespace}_bm25_{fingerprint.hexdigest()[:16]}.pkl"
    if path.is_file() and not rebuild:
        try:
            with path.open("rb") as handle:
                index = pickle.load(handle)
            if not isinstance(index, BM25Index):
                raise TypeError("cached object is not a BM25Index")
            return index, path, True
        except (OSError, pickle.UnpicklingError, EOFError, TypeError) as exc:
            raise RuntimeError(
                f"BM25 cache is unreadable: {path}. Remove this file or set "
                "rebuild_indices=True."
            ) from exc
    index = BM25Index(ids, texts, k1, b)
    with path.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return index, path, False


def hhr_interleave(
    sparse_results: list[dict[str, Any]], dense_results: list[dict[str, Any]], k: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Original HHR half-sparse + half-dense union with stable deduplication."""
    sparse_budget = (k + 1) // 2
    dense_budget = k // 2
    output: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for method, result in (
        ("sparse", sparse_results[:sparse_budget]),
        ("dense", dense_results[:dense_budget]),
    ):
        for hit in result:
            item_id = str(hit["item_id"])
            if item_id in by_id:
                existing = by_id[item_id]
                existing["sparse_rank"] = existing["sparse_rank"] or hit.get(
                    "sparse_rank"
                )
                existing["dense_rank"] = existing["dense_rank"] or hit.get("dense_rank")
                existing["source_method"] = "sparse+dense"
                continue
            copied = dict(hit)
            copied["source_method"] = method
            by_id[item_id] = copied
            output.append(copied)
    metadata = {
        "strategy": "hhr_interleave",
        "requested_k": k,
        "unique_results": len(output),
        "overlap_shortfall": len(output) < min(k, sparse_budget + dense_budget),
    }
    return output, metadata


def reciprocal_rank_fusion(
    sparse_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    k: int,
    rrf_k: int = 60,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for method, results in (("sparse", sparse_results), ("dense", dense_results)):
        for rank, hit in enumerate(results, start=1):
            item_id = str(hit["item_id"])
            entry = fused.setdefault(
                item_id,
                _result(item_id, 0.0, method, sparse_rank=None, dense_rank=None),
            )
            entry["score"] += 1.0 / (rrf_k + rank)
            entry[f"{method}_rank"] = hit.get(f"{method}_rank") or rank
            if entry["source_method"] != method:
                entry["source_method"] = "sparse+dense"
    output = sorted(fused.values(), key=lambda hit: (-hit["score"], hit["item_id"]))[:k]
    return output, {
        "strategy": "rrf",
        "rrf_k": rrf_k,
        "requested_k": k,
        "unique_results": len(output),
        "overlap_shortfall": len(output) < min(k, len(fused)),
    }


class CombinedDocumentRetriever:
    method = "combined"

    def __init__(
        self,
        sparse: SparseDocumentRetriever,
        dense: DenseDocumentRetriever,
        strategy: str,
        rrf_k: int,
    ):
        self.sparse, self.dense, self.strategy, self.rrf_k = (
            sparse,
            dense,
            strategy,
            rrf_k,
        )
        self.last_fusion_metadata: dict[str, Any] = {}

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        source_k = k if self.strategy == "hhr_interleave" else max(k, 2 * k)
        sparse = self.sparse.retrieve(query, source_k)
        dense = self.dense.retrieve(query, source_k)
        if self.strategy == "hhr_interleave":
            result, metadata = hhr_interleave(sparse, dense, k)
        else:
            result, metadata = reciprocal_rank_fusion(sparse, dense, k, self.rrf_k)
        self.last_fusion_metadata = metadata
        return result

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "strategy": self.strategy,
            "storage_bytes": self.sparse.metadata["storage_bytes"]
            + self.dense.metadata["storage_bytes"],
            **self.last_fusion_metadata,
        }


class CombinedPassageRetriever:
    method = "combined"

    def __init__(
        self,
        sparse: SparsePassageRetriever,
        dense: DensePassageRetriever,
        strategy: str,
        rrf_k: int,
    ):
        self.sparse, self.dense, self.strategy, self.rrf_k = (
            sparse,
            dense,
            strategy,
            rrf_k,
        )
        self.last_fusion_metadata: dict[str, Any] = {}

    def retrieve(
        self, query: str, k: int, document_ids: set[str]
    ) -> list[dict[str, Any]]:
        source_k = k if self.strategy == "hhr_interleave" else max(k, 2 * k)
        sparse = self.sparse.retrieve(query, source_k, document_ids)
        dense = self.dense.retrieve(query, source_k, document_ids)
        if self.strategy == "hhr_interleave":
            result, metadata = hhr_interleave(sparse, dense, k)
        else:
            result, metadata = reciprocal_rank_fusion(sparse, dense, k, self.rrf_k)
        self.last_fusion_metadata = metadata
        return result

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "strategy": self.strategy,
            "storage_bytes": self.sparse.metadata["storage_bytes"]
            + self.dense.metadata["storage_bytes"],
            **self.last_fusion_metadata,
        }


class Reranker(Protocol):
    def rerank(
        self, query: str, rankings: list[dict[str, Any]], passages: pd.DataFrame
    ) -> list[dict[str, Any]]: ...


@dataclass
class HHRQueryResult:
    document_rankings: list[dict[str, Any]]
    passage_rankings: list[dict[str, Any]]
    unique_result_count: int


class HHRPipeline:
    """Two-stage retriever with an optional Phase 2-compatible reranker hook."""

    def __init__(
        self,
        document_retriever: Any,
        passage_retriever: Any,
        passages: pd.DataFrame,
        config: RunConfig,
        reranker: Reranker | None = None,
    ):
        self.document_retriever = document_retriever
        self.passage_retriever = passage_retriever
        self.passages = passages
        self.config = config
        self.reranker = reranker
        self.passage_to_document = dict(
            zip(
                passages["passage_id"].astype(str),
                passages["document_id"].astype(str),
                strict=True,
            )
        )

    def retrieve(self, query: str) -> HHRQueryResult:
        documents = self.document_retriever.retrieve(query, self.config.document_top_k)
        document_ids = {str(hit["item_id"]) for hit in documents}
        passages = self.passage_retriever.retrieve(
            query, self.config.passage_top_k, document_ids
        )
        document_lookup = {
            str(hit["item_id"]): (rank, float(hit["score"]))
            for rank, hit in enumerate(documents, start=1)
        }
        for passage in passages:
            document_id = self.passage_to_document[str(passage["item_id"])]
            document_rank, document_score = document_lookup[document_id]
            passage["document_id"] = document_id
            passage["document_rank"] = document_rank
            passage["document_score"] = document_score
            passage["passage_score"] = float(passage["score"])
        if self.config.score_fusion == "weighted_sum" and passages:
            passage_scores = _minmax([hit["passage_score"] for hit in passages])
            document_scores = _minmax([hit["document_score"] for hit in passages])
            weight = self.config.document_score_weight
            for hit, passage_score, document_score in zip(
                passages, passage_scores, document_scores, strict=True
            ):
                hit["score"] = (1.0 - weight) * passage_score + weight * document_score
            passages.sort(key=lambda hit: (-hit["score"], hit["item_id"]))
        if self.reranker is not None:
            passages = self.reranker.rerank(query, passages, self.passages)
        passages = passages[: self.config.final_top_k]
        for rank, passage in enumerate(passages, start=1):
            passage["passage_rank"] = rank
        for rank, document in enumerate(documents, start=1):
            document["document_rank"] = rank
            document["document_score"] = float(document["score"])
        return HHRQueryResult(
            documents, passages, len({hit["item_id"] for hit in passages})
        )


def _minmax(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    low, high = float(array.min()), float(array.max())
    if high == low:
        return np.ones_like(array)
    return (array - low) / (high - low)


def build_retrievers(
    documents: pd.DataFrame,
    passages: pd.DataFrame,
    config: RunConfig,
    dataset_name: str,
) -> dict[str, dict[str, Any]]:
    encoder = make_dense_encoder(config)
    sparse_document = SparseDocumentRetriever(documents, config)
    dense_document = DenseDocumentRetriever(documents, config, encoder, dataset_name)
    sparse_passage = SparsePassageRetriever(passages, config)
    dense_passage = DensePassageRetriever(passages, config, encoder, dataset_name)
    combined_document = CombinedDocumentRetriever(
        sparse_document, dense_document, config.combined_strategy, config.rrf_k
    )
    combined_passage = CombinedPassageRetriever(
        sparse_passage, dense_passage, config.combined_strategy, config.rrf_k
    )
    return {
        "document": {
            "sparse": sparse_document,
            "dense": dense_document,
            "combined": combined_document,
        },
        "passage": {
            "sparse": sparse_passage,
            "dense": dense_passage,
            "combined": combined_passage,
        },
    }
