"""Pseudo-Instruction for document Chunking (PIC) reimplementation."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from chunkbench.chunking.base import BaseChunker
from chunkbench.chunking.postprocessing import (
    cap_groups,
    contiguous_groups,
    materialize_chunks,
)
from chunkbench.chunking.segments import sentence_segments
from chunkbench.chunking.validation import validate_advanced_chunks
from chunkbench.common.exceptions import MissingOptionalDependencyError
from chunkbench.common.types import Chunk, Document
from chunkbench.embedding.base import BaseEmbedder, HashingEmbedder


class BaseSummarizer(ABC):
    """Generate the pseudo-instruction required by PIC."""

    @abstractmethod
    def summarize(self, document: Document) -> str:
        """Return a concise document-level pseudo-instruction."""


class HeuristicSummarizer(BaseSummarizer):
    """Deterministic mock summarizer, never a substitute for paper's GPT-4o-mini."""

    def __init__(self, max_tokens: int = 48) -> None:
        self.max_tokens = max_tokens

    def summarize(self, document: Document) -> str:
        return " ".join(document.text.split()[: self.max_tokens])


class TransformersSummarizer(BaseSummarizer):
    """Optional local seq2seq pseudo-instruction generator for real PIC runs."""

    def __init__(
        self,
        model_name: str,
        model_revision: str | None = None,
        max_new_tokens: int = 64,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise MissingOptionalDependencyError(
                "pic_paper_reimplementation", "ppl"
            ) from error
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, revision=model_revision
        )
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def summarize(self, document: Document) -> str:
        prompt = f"summarize: {document.text}"
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            result = self.model.generate(**encoded, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(result[0], skip_special_tokens=True)


class PICChunker(BaseChunker):
    """Group maximal adjacent runs above or below the mean summary similarity."""

    def __init__(
        self,
        variant: str,
        embedder: BaseEmbedder | None = None,
        summarizer: BaseSummarizer | None = None,
        max_chunk_tokens: int | None = None,
        **_: Any,
    ) -> None:
        if variant not in {
            "pic_paper_reimplementation",
            "pic_reimplementation_capped_512",
        }:
            raise ValueError(f"Unknown PIC variant: {variant}")
        self.variant = variant
        self.embedder = embedder or HashingEmbedder()
        self.summarizer = summarizer or HeuristicSummarizer()
        self.max_chunk_tokens = (
            int(max_chunk_tokens)
            if max_chunk_tokens is not None
            else (512 if variant.endswith("capped_512") else None)
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """Implement PIC Eq. 2--4 using summary similarity and mean threshold."""
        segments = sentence_segments(document)
        if not segments:
            return []
        pseudo_instruction = self.summarizer.summarize(document)
        vectors = self.embedder.encode_documents(
            [pseudo_instruction, *[item.text for item in segments]]
        )
        summary = vectors[0]
        denominator = float(np.linalg.norm(summary))
        similarities: list[float] = []
        for vector in vectors[1:]:
            norm = denominator * float(np.linalg.norm(vector))
            similarities.append(float(np.dot(summary, vector) / norm) if norm else 0.0)
        threshold = float(np.mean(similarities))
        signs = [value >= threshold for value in similarities]
        boundaries = {
            index for index in range(len(signs) - 1) if signs[index] != signs[index + 1]
        }
        groups = cap_groups(
            contiguous_groups(segments, boundaries), self.max_chunk_tokens
        )
        chunks = materialize_chunks(
            document,
            groups,
            "pic",
            {
                "variant": self.variant,
                "pseudo_instruction": pseudo_instruction,
                "summary_similarities": similarities,
                "threshold": threshold,
                "selected_boundary_positions": sorted(boundaries),
                "max_chunk_tokens": self.max_chunk_tokens,
            },
        )
        validate_advanced_chunks(document, segments, chunks, self.max_chunk_tokens)
        return chunks
