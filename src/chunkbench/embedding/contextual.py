"""Token-level contextual embedding interface used by late chunking."""

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from chunkbench.common.exceptions import MissingOptionalDependencyError
from chunkbench.common.types import Document


class ContextualDocument:
    """Contextual token states and offsets for one encoded document."""

    def __init__(
        self,
        vectors: NDArray[np.float32],
        offsets: list[tuple[int, int]],
        attention_mask: list[bool],
        special_tokens: list[bool],
    ) -> None:
        self.vectors = vectors
        self.offsets = offsets
        self.attention_mask = attention_mask
        self.special_tokens = special_tokens


class BaseContextualEmbedder(ABC):
    """Produce one contextual vector per token after whole-document encoding."""

    @abstractmethod
    def encode_document(self, document: Document) -> ContextualDocument:
        """Encode a document without pre-chunking it."""


class HashingContextualEmbedder(BaseContextualEmbedder):
    """Offline fake contextual model for tests and explicitly mock profiles."""

    def __init__(
        self,
        dimension: int = 256,
        max_model_tokens: int = 8192,
        long_document_policy: str = "error",
    ) -> None:
        self.dimension = dimension
        self.max_model_tokens = max_model_tokens
        self.long_document_policy = long_document_policy

    def _vector(self, value: str) -> NDArray[np.float32]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        digest = hashlib.blake2b(value.encode(), digest_size=8).digest()
        vector[int.from_bytes(digest, "little") % self.dimension] = 1.0
        return vector

    def encode_document(self, document: Document) -> ContextualDocument:
        matches = list(re.finditer(r"\S+", document.text))
        if (
            len(matches) > self.max_model_tokens
            and self.long_document_policy == "error"
        ):
            raise ValueError(
                "Late chunking document exceeds max_model_tokens; policy=error"
            )
        if (
            len(matches) > self.max_model_tokens
            and self.long_document_policy == "truncate"
        ):
            matches = matches[: self.max_model_tokens]
        if len(matches) > self.max_model_tokens:
            raise ValueError(
                "Mock contextual backend supports only error or truncate policy"
            )
        document_context = self._vector(document.text.lower())
        vectors = np.vstack(
            [
                self._vector(match.group().lower()) + document_context
                for match in matches
            ]
        ).astype(np.float32)
        return ContextualDocument(
            vectors=vectors,
            offsets=[(match.start(), match.end()) for match in matches],
            attention_mask=[True] * len(matches),
            special_tokens=[False] * len(matches),
        )


class TransformersContextualEmbedder(BaseContextualEmbedder):
    """Optional Hugging Face implementation with offsets, masks and eval inference."""

    def __init__(
        self,
        model_name: str,
        model_revision: str | None = None,
        device: str | None = None,
        max_model_tokens: int = 8192,
        long_document_policy: str = "error",
        window_stride: int = 64,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise MissingOptionalDependencyError("late_fixed_256", "late") from error
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, revision=model_revision, trust_remote_code=True
        )
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_model_tokens = max_model_tokens
        self.long_document_policy = long_document_policy
        self.window_stride = int(window_stride)

    def _encode_single(self, encoded: dict[str, object]) -> ContextualDocument:
        """Run one tokenized model window and preserve its source offsets."""
        offsets_tensor = encoded.pop("offset_mapping")
        offsets = [tuple(item) for item in offsets_tensor[0].tolist()]
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            vectors = (
                self.model(**encoded)
                .last_hidden_state[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        attention = [
            bool(item) for item in encoded["attention_mask"][0].detach().cpu().tolist()
        ]
        special = [start == end for start, end in offsets]
        return ContextualDocument(vectors, offsets, attention, special)

    def _encode_windows(self, document: Document) -> ContextualDocument:
        """Encode overlapping windows and retain the first state per token offset."""
        stride = min(max(0, self.window_stride), self.max_model_tokens - 1)
        encoded = self.tokenizer(
            document.text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_model_tokens,
            stride=stride,
            padding=True,
        )
        encoded.pop("overflow_to_sample_mapping", None)
        states_by_offset: dict[tuple[int, int], NDArray[np.float32]] = {}
        window_count = int(encoded["input_ids"].shape[0])
        for index in range(window_count):
            window = {key: value[index : index + 1] for key, value in encoded.items()}
            state = self._encode_single(window)
            for vector, offset, attended, special in zip(
                state.vectors,
                state.offsets,
                state.attention_mask,
                state.special_tokens,
                strict=True,
            ):
                if attended and not special:
                    states_by_offset.setdefault(offset, vector)
        if not states_by_offset:
            raise ValueError("Late chunking window encoder produced no document tokens")
        offsets = sorted(states_by_offset)
        vectors = np.vstack([states_by_offset[offset] for offset in offsets]).astype(
            np.float32
        )
        return ContextualDocument(
            vectors,
            offsets,
            [True] * len(offsets),
            [False] * len(offsets),
        )

    def encode_document(self, document: Document) -> ContextualDocument:
        encoded = self.tokenizer(
            document.text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=False,
        )
        count = int(encoded["input_ids"].shape[1])
        if count > self.max_model_tokens and self.long_document_policy == "error":
            raise ValueError(
                "Late chunking document exceeds max_model_tokens; policy=error"
            )
        if count > self.max_model_tokens and self.long_document_policy == "truncate":
            encoded = self.tokenizer(
                document.text,
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_model_tokens,
            )
        elif count > self.max_model_tokens and self.long_document_policy == "window":
            return self._encode_windows(document)
        elif count > self.max_model_tokens:
            raise ValueError(
                "Only error, truncate, and window long-document policies are available"
            )
        return self._encode_single(encoded)
