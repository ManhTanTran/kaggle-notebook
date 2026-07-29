"""Injectable perplexity scorers with a dependency-free deterministic backend."""

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any

from chunkbench.common.exceptions import MissingOptionalDependencyError


class BasePerplexityScorer(ABC):
    """Score how surprising a candidate segment is after its left context."""

    @abstractmethod
    def score_transition(self, left_context: str, candidate_segment: str) -> float:
        """Return a lower-is-more-coherent transition score."""


class DeterministicPerplexityScorer(BasePerplexityScorer):
    """Offline lexical scorer for tests and mock smoke profiles only."""

    def score_transition(self, left_context: str, candidate_segment: str) -> float:
        left_words = set(left_context.lower().split())
        candidate_words = set(candidate_segment.lower().split())
        if not candidate_words:
            return 0.0
        overlap = len(left_words & candidate_words) / len(candidate_words)
        fingerprint = hashlib.blake2b(
            candidate_segment.encode(), digest_size=2
        ).digest()
        stable_jitter = int.from_bytes(fingerprint, "little") / 65535.0 / 1000.0
        return 1.0 - overlap + stable_jitter


class TransformersPerplexityScorer(BasePerplexityScorer):
    """Causal-LM scorer that loads once and masks padding and context loss."""

    def __init__(
        self,
        model_name: str,
        model_revision: str | None = None,
        device: str | None = None,
        precision: str | None = None,
        max_sequence_tokens: int | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise MissingOptionalDependencyError("meta_ppl_raw", "ppl") from error
        kwargs: dict[str, Any] = {"revision": model_revision}
        if precision == "float16":
            kwargs["torch_dtype"] = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self._torch = torch
        configured_limit = max_sequence_tokens or getattr(
            self.model.config, "n_positions", None
        )
        self.max_sequence_tokens = int(configured_limit or 1024)
        if self.max_sequence_tokens > 100_000:
            self.max_sequence_tokens = 1024
        self._cache: dict[tuple[str, str], float] = {}

    def score_transition(self, left_context: str, candidate_segment: str) -> float:
        key = (left_context, candidate_segment)
        if key in self._cache:
            return self._cache[key]
        context_ids = self.tokenizer(left_context, add_special_tokens=False)[
            "input_ids"
        ]
        candidate_ids = self.tokenizer(candidate_segment, add_special_tokens=False)[
            "input_ids"
        ]
        if not candidate_ids:
            return 0.0
        candidate_ids = candidate_ids[-self.max_sequence_tokens :]
        context_limit = self.max_sequence_tokens - len(candidate_ids)
        input_ids = context_ids[-context_limit:] + candidate_ids
        encoded = {
            "input_ids": self._torch.tensor([input_ids], device=self.device),
            "attention_mask": self._torch.ones(
                (1, len(input_ids)), dtype=self._torch.long, device=self.device
            ),
        }
        labels = encoded["input_ids"].clone()
        labels[:, : len(input_ids) - len(candidate_ids)] = -100
        with self._torch.inference_mode():
            output = self.model(**encoded, labels=labels)
        score = float(math.exp(float(output.loss)))
        self._cache[key] = score
        return score
