"""Paper-derived global single-linkage semantic clustering."""

import numpy as np

from chunkbench.chunking.base import BaseChunker
from chunkbench.chunking.boundaries import cosine_distance
from chunkbench.chunking.postprocessing import materialize_chunks
from chunkbench.chunking.segments import sentence_segments
from chunkbench.chunking.validation import validate_advanced_chunks
from chunkbench.common.types import Chunk, Document
from chunkbench.embedding.base import BaseEmbedder, HashingEmbedder


def compute_pairwise_distance(vectors: np.ndarray, lambda_weight: float) -> np.ndarray:
    """Use Qu et al.'s weighted positional and clipped-cosine distance."""
    count = len(vectors)
    matrix = np.full((count, count), np.inf, dtype=np.float32)
    for left in range(count):
        for right in range(left + 1, count):
            semantic = cosine_distance(vectors[left], vectors[right])
            semantic = min(1.0, max(0.0, semantic))
            value = (
                lambda_weight * abs(left - right) / count
                + (1 - lambda_weight) * semantic
            )
            matrix[left, right] = matrix[right, left] = value
    return matrix


def select_single_link_merge(
    clusters: list[set[int]], matrix: np.ndarray
) -> tuple[int, int, float] | None:
    """Return the globally smallest cluster pair with deterministic tie-breaking."""
    candidate: tuple[float, int, int] | None = None
    for left in range(len(clusters)):
        for right in range(left + 1, len(clusters)):
            value = min(matrix[a, b] for a in clusters[left] for b in clusters[right])
            key = (float(value), min(clusters[left]), min(clusters[right]))
            if candidate is None or key < candidate:
                candidate = key
    if candidate is None:
        return None
    return candidate[1], candidate[2], candidate[0]


class SemanticSingleLinkageChunker(BaseChunker):
    """Reimplement the paper's unconstrained-order single-linkage protocol.

    The registry name is retained for compatibility, but manifest fidelity is
    deliberately marked unverified until paper-level result reproduction exists.
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        lambda_weight: float = 0.5,
        target_clusters: int = 4,
        distance_threshold: float = 0.5,
        max_cluster_sentences: int | None = None,
        **_: object,
    ) -> None:
        if not 0 <= lambda_weight <= 1:
            raise ValueError("lambda_weight must be in [0, 1]")
        self.embedder = embedder or HashingEmbedder()
        self.lambda_weight = float(lambda_weight)
        self.target_clusters = max(1, int(target_clusters))
        self.distance_threshold = float(distance_threshold)
        self.max_cluster_sentences = max_cluster_sentences

    def chunk(self, document: Document) -> list[Chunk]:
        """Globally cluster sentences; clusters may intentionally be non-contiguous."""
        segments = sentence_segments(document)
        if not segments:
            return []
        vectors = self.embedder.encode_documents([segment.text for segment in segments])
        matrix = compute_pairwise_distance(vectors, self.lambda_weight)
        maximum = self.max_cluster_sentences or int(
            np.ceil(len(segments) / self.target_clusters)
        )
        clusters = [{index} for index in range(len(segments))]
        merge_log: list[dict[str, object]] = []
        while len(clusters) > self.target_clusters:
            selected = select_single_link_merge(clusters, matrix)
            if selected is None:
                break
            left, right, distance = selected
            if distance > self.distance_threshold:
                break
            if len(clusters[left]) + len(clusters[right]) > maximum:
                matrix = matrix.copy()
                for first in clusters[left]:
                    for second in clusters[right]:
                        matrix[first, second] = matrix[second, first] = np.inf
                continue
            merged = clusters[left] | clusters[right]
            merge_log.append(
                {
                    "left": sorted(clusters[left]),
                    "right": sorted(clusters[right]),
                    "distance": distance,
                }
            )
            clusters = [
                item
                for index, item in enumerate(clusters)
                if index not in {left, right}
            ]
            clusters.append(merged)
        groups = [
            [segments[index] for index in sorted(cluster)]
            for cluster in sorted(clusters, key=lambda item: min(item))
        ]
        chunks = materialize_chunks(
            document,
            groups,
            "semantic-single-linkage",
            {
                "lambda_weight": self.lambda_weight,
                "distance_threshold": self.distance_threshold,
                "target_clusters": self.target_clusters,
                "max_cluster_sentences": maximum,
                "merge_log": merge_log,
            },
        )
        validate_advanced_chunks(document, segments, chunks, None)
        return chunks
