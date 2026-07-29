"""Extensible chunking-method registry and pipeline metadata."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from chunkbench.chunking.base import BaseChunker
from chunkbench.chunking.fixed import FixedTokenChunker
from chunkbench.chunking.late_chunking import LateChunker
from chunkbench.chunking.meta_ppl import MetaPPLChunker
from chunkbench.chunking.pic import PICChunker, TransformersSummarizer
from chunkbench.chunking.semantic_breakpoint import SemanticBreakpointChunker
from chunkbench.chunking.semantic_single_linkage import SemanticSingleLinkageChunker
from chunkbench.chunking.sentence_fixed import SentenceFixedChunker
from chunkbench.embedding.base import BaseEmbedder
from chunkbench.scoring.perplexity import (
    DeterministicPerplexityScorer,
    TransformersPerplexityScorer,
)

CORE_METHODS = [
    "fixed_256",
    "fixed_512",
    "sentence_fixed_256",
    "sentence_fixed_512",
    "semantic_breakpoint",
    "semantic_single_linkage_paper_exact",
    "meta_ppl_raw",
    "meta_ppl_dynamic_512",
    "pic_paper_reimplementation",
    "pic_reimplementation_capped_512",
    "late_fixed_256",
    "late_fixed_512",
]

Factory = Callable[[dict[str, Any]], BaseChunker]


@dataclass(frozen=True)
class MethodPipelineSpec:
    """Backward-compatible method metadata consumed by runners and manifests."""

    name: str
    chunker: BaseChunker
    family: str
    representation_strategy: str = "independent_chunk_embedding"
    required_capabilities: tuple[str, ...] = ()
    implementation_fidelity: str = "baseline"
    source_reference: str | None = None
    repository_url: str | None = None
    source_commit: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


_PROVENANCE: dict[str, dict[str, str | None]] = {
    "semantic_breakpoint": {
        "family": "semantic_breakpoint",
        "fidelity": "idea_based_reimplementation",
        "source": "https://aclanthology.org/2025.findings-naacl.114/",
        "commit": None,
    },
    "semantic_single_linkage_paper_exact": {
        "family": "semantic_single_linkage",
        "fidelity": "paper_reimplementation_unverified",
        "source": "https://aclanthology.org/2025.findings-naacl.114/",
        "commit": None,
    },
    "meta_ppl_raw": {
        "family": "meta_ppl",
        "fidelity": "paper_reimplementation_unverified",
        "source": "https://arxiv.org/abs/2410.12788",
        "repository": "https://github.com/IAAR-Shanghai/Meta-Chunking",
        "commit": "2dbf487a75e44a378eef1a35909bdbe871c396a7",
    },
    "meta_ppl_dynamic_512": {
        "family": "meta_ppl",
        "fidelity": "idea_based_reimplementation",
        "source": "https://arxiv.org/abs/2410.12788",
        "repository": "https://github.com/IAAR-Shanghai/Meta-Chunking",
        "commit": "2dbf487a75e44a378eef1a35909bdbe871c396a7",
    },
    "pic_paper_reimplementation": {
        "family": "pic",
        "fidelity": "paper_reimplementation_unverified",
        "source": "https://aclanthology.org/2025.findings-acl.422/",
        "commit": None,
    },
    "pic_reimplementation_capped_512": {
        "family": "pic",
        "fidelity": "idea_based_reimplementation",
        "source": "https://aclanthology.org/2025.findings-acl.422/",
        "commit": None,
    },
    "late_fixed_256": {
        "family": "late_chunking",
        "fidelity": "paper_reimplementation_unverified",
        "source": "https://arxiv.org/abs/2409.04701",
        "repository": "https://github.com/jina-ai/late-chunking",
        "commit": "1d3bb02bf091becd0771455e4e7959463935e26c",
    },
    "late_fixed_512": {
        "family": "late_chunking",
        "fidelity": "paper_reimplementation_unverified",
        "source": "https://arxiv.org/abs/2409.04701",
        "repository": "https://github.com/jina-ai/late-chunking",
        "commit": "1d3bb02bf091becd0771455e4e7959463935e26c",
    },
}


def _fixed(size: int) -> Factory:
    return lambda config: FixedTokenChunker(
        int(config.get("chunk_size", size)), int(config.get("overlap", 0))
    )


def _sentence(size: int) -> Factory:
    return lambda config: SentenceFixedChunker(int(config.get("chunk_size", size)))


def _semantic(factory: Callable[..., BaseChunker]) -> Factory:
    def build(config: dict[str, Any]) -> BaseChunker:
        clean = {key: value for key, value in config.items() if not key.startswith("_")}
        injected = config.get("_embedder")
        if injected is not None:
            clean["embedder"] = injected
        return factory(**clean)

    return build


def _meta(variant: str) -> Factory:
    def build(config: dict[str, Any]) -> BaseChunker:
        clean = {key: value for key, value in config.items() if not key.startswith("_")}
        backend_type = str(clean.pop("backend_type", "mock"))
        if backend_type == "mock":
            clean["scorer"] = DeterministicPerplexityScorer()
        elif backend_type == "transformers":
            clean["scorer"] = TransformersPerplexityScorer(
                str(clean.pop("model_name")),
                clean.pop("model_revision", None),
                clean.pop("device", None),
                clean.pop("precision", None),
            )
        else:
            raise ValueError(f"Unknown Meta-PPL backend_type: {backend_type}")
        return MetaPPLChunker(variant, **clean)

    return build


def _pic(variant: str) -> Factory:
    """Build PIC only with an explicit mock or injected real summarizer."""

    def build(config: dict[str, Any]) -> BaseChunker:
        clean = {key: value for key, value in config.items() if not key.startswith("_")}
        backend_type = str(clean.pop("backend_type", "mock"))
        if backend_type == "transformers":
            clean["summarizer"] = TransformersSummarizer(
                model_name=str(clean.pop("summarizer_model_name")),
                model_revision=clean.pop("summarizer_model_revision", None),
                max_new_tokens=int(clean.pop("max_new_tokens", 64)),
                device=clean.pop("device", None),
            )
        elif backend_type != "mock" and "summarizer" not in config:
            raise ValueError(
                "PIC real backend requires an injected summarizer adapter; "
                "the deterministic mock is not used for real runs"
            )
        injected = config.get("_embedder")
        if injected is not None:
            clean["embedder"] = injected
        return PICChunker(variant, **clean)

    return build


METHOD_REGISTRY: dict[str, Factory] = {
    "fixed_256": _fixed(256),
    "fixed_512": _fixed(512),
    "sentence_fixed_256": _sentence(256),
    "sentence_fixed_512": _sentence(512),
    "semantic_breakpoint": _semantic(SemanticBreakpointChunker),
    "semantic_single_linkage_paper_exact": _semantic(SemanticSingleLinkageChunker),
    "meta_ppl_raw": _meta("meta_ppl_raw"),
    "meta_ppl_dynamic_512": _meta("meta_ppl_dynamic_512"),
    "pic_paper_reimplementation": _pic("pic_paper_reimplementation"),
    "pic_reimplementation_capped_512": _pic("pic_reimplementation_capped_512"),
    "late_fixed_256": lambda config: LateChunker(256, **config),
    "late_fixed_512": lambda config: LateChunker(512, **config),
}


def register_method(name: str, factory: Factory) -> None:
    """Register a method without changing BenchmarkRunner."""
    if name in METHOD_REGISTRY:
        raise ValueError(f"Method already registered: {name}")
    METHOD_REGISTRY[name] = factory


def build_chunker(name: str, config: dict[str, Any] | None = None) -> BaseChunker:
    """Build a registered chunker, preserving the original public interface."""
    try:
        return METHOD_REGISTRY[name](config or {})
    except KeyError as error:
        raise KeyError(f"Unknown chunking method {name!r}") from error


def build_method_pipeline_spec(
    name: str,
    config: dict[str, Any] | None = None,
    embedder: BaseEmbedder | None = None,
) -> MethodPipelineSpec:
    """Build a chunker plus representation metadata from one registry entry."""
    method_config = dict(config or {})
    if embedder is not None:
        method_config["_embedder"] = embedder
    chunker = build_chunker(name, method_config)
    provenance = _PROVENANCE.get(name, {})
    late = name.startswith("late_")
    return MethodPipelineSpec(
        name=name,
        chunker=chunker,
        family=str(provenance.get("family", "baseline")),
        representation_strategy="late_document_embedding"
        if late
        else "independent_chunk_embedding",
        required_capabilities=("contextual_token_embeddings",) if late else (),
        implementation_fidelity=str(provenance.get("fidelity", "baseline")),
        source_reference=provenance.get("source"),
        repository_url=provenance.get("repository"),
        source_commit=provenance.get("commit"),
        config={
            key: value
            for key, value in method_config.items()
            if not key.startswith("_")
        },
    )
