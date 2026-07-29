"""Self-contained dataset used by tests and smoke runs."""

from chunkbench.common.types import DatasetBundle, Document, Evidence, Query
from chunkbench.data.base import DatasetAdapter


class SyntheticDatasetAdapter(DatasetAdapter):
    """Return a deterministic, dependency-free retrieval dataset."""

    def load(self) -> DatasetBundle:
        """Build three documents with exact evidence phrases."""
        documents = [
            Document(
                "doc_alpha",
                "Alpha project studies solar energy. "
                "Its primary result is a twenty percent efficiency gain. "
                "The experiment ran for twelve weeks.",
                {"topic": "energy"},
            ),
            Document(
                "doc_beta",
                "Beta project studies marine ecology. "
                "Researchers observed coral recovery after pollution controls. "
                "The survey covered five reefs.",
                {"topic": "ecology"},
            ),
            Document(
                "doc_gamma",
                "Gamma is a control document about classical music and orchestras.",
                {"topic": "music"},
            ),
        ]
        evidence_texts = [
            ("q1", "ev1", "doc_alpha", "twenty percent efficiency gain"),
            ("q2", "ev2", "doc_beta", "coral recovery after pollution controls"),
        ]
        evidence = [
            Evidence(query_id, evidence_id, document_id, text, len(text.split()))
            for query_id, evidence_id, document_id, text in evidence_texts
        ]
        queries = [
            Query(
                "q1",
                "What efficiency gain did the solar project report?",
                ("doc_alpha",),
            ),
            Query("q2", "What recovered after pollution controls?", ("doc_beta",)),
        ]
        return DatasetBundle(
            documents=documents,
            queries=queries,
            evidence=evidence,
            metadata={
                "name": "synthetic",
                "version": "1",
                "dataset_name": "synthetic",
                "split": str(self.config.get("split", "test")),
                "language": "en",
                "evaluation_profile": "qa_evidence_retrieval",
                "source": "generated",
                "adapter_version": "1.0.0",
            },
        )
