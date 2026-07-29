"""UIT-ViQuAD adapter with sentence-containing-answer evidence."""

from pathlib import Path

from chunkbench.common.types import DatasetBundle, Document, Evidence, Query
from chunkbench.data.base import DatasetAdapter
from chunkbench.data.normalization import resolve_split_file
from chunkbench.data.uit_viquad_evidence import answer_sentence_groups
from chunkbench.data.uit_viquad_parser import iter_paragraphs, load_uit_viquad


class UITViQuADAdapter(DatasetAdapter):
    """Normalize Vietnamese SQuAD-style contexts without using answer-only evidence."""

    def load(self) -> DatasetBundle:
        """Load one local split and preserve all answer annotations."""
        split = str(self.config.get("split", "validation"))
        path = resolve_split_file(Path(str(self.config["data_path"])), split)
        articles = load_uit_viquad(path)
        documents = []
        queries = []
        evidence = []
        for article_index, paragraph_index, article, paragraph in iter_paragraphs(
            articles
        ):
            document_id = f"uit_viquad:{article_index}:{paragraph_index}"
            context = str(paragraph["context"])
            documents.append(
                Document(
                    document_id,
                    context,
                    {
                        "title": article.get("title"),
                        "article_index": article_index,
                        "paragraph_index": paragraph_index,
                    },
                )
            )
            for raw_query in paragraph.get("qas", []):
                query_id = str(raw_query["id"])
                answers = list(raw_query.get("answers", []))
                queries.append(
                    Query(
                        query_id,
                        str(raw_query["question"]),
                        (document_id,),
                        {
                            "answer_annotations": answers,
                            "unanswerable": not bool(answers),
                        },
                    )
                )
                for index, group in enumerate(answer_sentence_groups(context, answers)):
                    start = int(group["sentence_start"])
                    end = int(group["sentence_end"])
                    sentence = context[start:end]
                    answer_spans = group["answer_spans"]
                    evidence.append(
                        Evidence(
                            query_id,
                            f"{query_id}:e{index}",
                            document_id,
                            sentence,
                            len(sentence.split()),
                            {
                                "granularity": (
                                    "sentence"
                                    if group["sentence_count"] == 1
                                    else "sentence_window"
                                ),
                                "char_spans": [[start, end]],
                                "sentence_start": start,
                                "sentence_end": end,
                                "sentence_count": group["sentence_count"],
                                "answer_spans": answer_spans,
                                "answer_text": answer_spans[0]["answer_text"],
                                "answer_start": answer_spans[0]["answer_start"],
                                "answer_end": answer_spans[0]["answer_end"],
                                "raw_locator": {
                                    "article_index": article_index,
                                    "paragraph_index": paragraph_index,
                                },
                            },
                        )
                    )
        return DatasetBundle(
            documents,
            queries,
            evidence,
            {
                "dataset_name": "uit_viquad",
                "split": split,
                "language": str(self.config.get("language", "vi")),
                "evaluation_profile": str(
                    self.config.get("evaluation_profile", "qa_evidence_retrieval")
                ),
                "source": str(path),
                "adapter_version": "1.0.0",
            },
        )
