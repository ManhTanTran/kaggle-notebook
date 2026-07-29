"""QASPER adapter using annotated supporting evidence."""

from pathlib import Path

from chunkbench.common.types import DatasetBundle, Document, Evidence, Query
from chunkbench.data.base import DatasetAdapter
from chunkbench.data.normalization import resolve_split_file
from chunkbench.data.qasper_evidence import annotated_evidence, locate_evidence
from chunkbench.data.qasper_parser import build_paper_text, load_qasper


class QasperAdapter(DatasetAdapter):
    """Normalize official QASPER JSON without substituting answer text."""

    def load(self) -> DatasetBundle:
        """Load one configured split into canonical documents and evidence."""
        split = str(self.config.get("split", "validation"))
        source = self.config.get("data_path") or self.config.get("source")
        if not source:
            raise ValueError("QASPER requires data_path")
        path = resolve_split_file(Path(source), split)
        raw = load_qasper(path)
        documents: list[Document] = []
        queries: list[Query] = []
        evidence: list[Evidence] = []
        policy = str(self.config.get("unanswerable_policy", "exclude"))
        if policy not in {"include", "exclude", "mark"}:
            raise ValueError(f"Invalid QASPER unanswerable_policy: {policy}")
        excluded_unanswerable = 0
        for paper_id, paper in raw.items():
            document_id = f"qasper:{paper_id}"
            document_text, locators = build_paper_text(paper)
            documents.append(
                Document(
                    document_id=document_id,
                    text=document_text,
                    metadata={
                        "paper_id": paper_id,
                        "title": paper.get("title"),
                        "segments": locators,
                    },
                )
            )
            for raw_query in paper.get("qas", []):
                query_id = str(raw_query["question_id"])
                evidence_texts, unanswerable = annotated_evidence(raw_query)
                if unanswerable and policy == "exclude":
                    excluded_unanswerable += 1
                    continue
                queries.append(
                    Query(
                        query_id=query_id,
                        text=str(raw_query["question"]),
                        relevant_document_ids=(document_id,),
                        metadata={
                            "unanswerable": unanswerable,
                            "unanswerable_policy": policy,
                        },
                    )
                )
                for index, evidence_text in enumerate(evidence_texts):
                    evidence.append(
                        Evidence(
                            query_id=query_id,
                            evidence_id=f"{query_id}:e{index}",
                            document_id=document_id,
                            text=evidence_text,
                            token_count=len(evidence_text.split()),
                            metadata=locate_evidence(
                                document_text, evidence_text, locators
                            ),
                        )
                    )
        return DatasetBundle(
            documents,
            queries,
            evidence,
            {
                "dataset_name": "qasper",
                "split": split,
                "language": str(self.config.get("language", "en")),
                "evaluation_profile": str(
                    self.config.get("evaluation_profile", "qa_evidence_retrieval")
                ),
                "source": str(path),
                "adapter_version": "1.0.0",
                "excluded_unanswerable_count": excluded_unanswerable,
            },
        )
