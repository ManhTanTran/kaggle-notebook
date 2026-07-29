"""ViMQA adapter for its official HotpotQA-like raw schema."""

from pathlib import Path

from chunkbench.common.types import DatasetBundle, Document, Query
from chunkbench.data.base import DatasetAdapter
from chunkbench.data.normalization import (
    normalize_title,
    resolve_split_file,
    stable_id,
)
from chunkbench.data.vimqa_evidence import make_evidence
from chunkbench.data.vimqa_parser import context_documents, load_vimqa


class ViMQAAdapter(DatasetAdapter):
    """Normalize Vietnamese multi-hop supporting facts without inference."""

    def load(self) -> DatasetBundle:
        """Load an authorized local ViMQA split."""
        split = str(self.config.get("split", "validation"))
        path = resolve_split_file(Path(str(self.config["data_path"])), split)
        raw = load_vimqa(path)
        contexts = context_documents(raw)
        title_lookup = {
            normalize_title(title): (title, stable_id("vimqa", title), *values)
            for title, values in contexts.items()
        }
        documents = [
            Document(
                document_id,
                text,
                {
                    "article_title": title,
                    "normalized_title": normalized_title,
                    "sentence_spans": spans,
                },
            )
            for normalized_title, (
                title,
                document_id,
                text,
                _,
                spans,
            ) in sorted(title_lookup.items())
        ]
        queries = []
        evidence = []
        for item in raw:
            query_id = str(item["_id"])
            facts = item["supporting_facts"]
            relevant_ids = tuple(
                dict.fromkeys(
                    title_lookup[normalize_title(str(title))][1] for title, _ in facts
                )
            )
            queries.append(
                Query(
                    query_id,
                    str(item["question"]),
                    relevant_ids,
                    {
                        "answer": item.get("answer"),
                        "multi_hop": len(relevant_ids) > 1,
                    },
                )
            )
            for index, (title, sentence_index) in enumerate(facts):
                canonical, document_id, _, sentences, spans = title_lookup[
                    normalize_title(str(title))
                ]
                evidence.append(
                    make_evidence(
                        query_id,
                        index,
                        document_id,
                        canonical,
                        int(sentence_index),
                        sentences,
                        spans,
                    )
                )
        return DatasetBundle(
            documents,
            queries,
            evidence,
            {
                "dataset_name": "vimqa",
                "split": split,
                "language": str(self.config.get("language", "vi")),
                "evaluation_profile": str(
                    self.config.get("evaluation_profile", "qa_evidence_retrieval")
                ),
                "source": str(path),
                "adapter_version": "1.0.0",
            },
        )
