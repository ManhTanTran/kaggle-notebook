"""HotpotQA FullWiki adapter with streaming title lookup."""

from pathlib import Path

from chunkbench.common.types import DatasetBundle, Document, Query
from chunkbench.data.base import DatasetAdapter
from chunkbench.data.hotpotqa_corpus import (
    lookup_articles,
    normalize_article,
)
from chunkbench.data.hotpotqa_evidence import supporting_evidence
from chunkbench.data.hotpotqa_parser import load_questions
from chunkbench.data.normalization import normalize_wikipedia_title, stable_id


class HotpotQAFullWikiAdapter(DatasetAdapter):
    """Normalize FullWiki questions and sentence-level supporting facts."""

    def load(self) -> DatasetBundle:
        """Load fixture or local FullWiki files without downloading data."""
        split = str(self.config.get("split", "validation"))
        questions_path = Path(str(self.config["questions_path"]))
        corpus_path = Path(str(self.config["corpus_path"]))
        questions = load_questions(questions_path)
        needed_titles = {
            normalize_wikipedia_title(str(title))
            for item in questions
            for title, _ in item.get("supporting_facts", [])
        }
        raw_articles = lookup_articles(corpus_path, needed_titles)
        missing = needed_titles - set(raw_articles)
        if missing:
            raise ValueError(f"HotpotQA corpus is missing articles: {sorted(missing)}")
        documents = []
        article_data = {}
        for normalized_title in sorted(raw_articles):
            article = raw_articles[normalized_title]
            title = str(article["title"])
            document_id = stable_id("hotpotqa", title, case_sensitive=True)
            text, sentences, locators = normalize_article(article)
            article_data[normalized_title] = (
                document_id,
                title,
                sentences,
                locators,
            )
            documents.append(
                Document(
                    document_id,
                    text,
                    {
                        "article_title": title,
                        "normalized_title": normalized_title,
                        "sentence_locators": locators,
                    },
                )
            )
        queries = []
        evidence = []
        for item in questions:
            query_id = str(item["_id"])
            facts = item.get("supporting_facts")
            if facts is None:
                raise ValueError(f"HotpotQA query {query_id} has no supporting_facts")
            relevant_ids = tuple(
                dict.fromkeys(
                    article_data[normalize_wikipedia_title(str(title))][0]
                    for title, _ in facts
                )
            )
            queries.append(
                Query(
                    query_id,
                    str(item["question"]),
                    relevant_ids,
                    {"answer": item.get("answer"), "multi_hop": len(relevant_ids) > 1},
                )
            )
            for index, (title, sentence_index) in enumerate(facts):
                document_id, canonical_title, sentences, locators = article_data[
                    normalize_wikipedia_title(str(title))
                ]
                evidence.append(
                    supporting_evidence(
                        query_id,
                        document_id,
                        canonical_title,
                        int(sentence_index),
                        sentences,
                        locators,
                        index,
                    )
                )
        return DatasetBundle(
            documents,
            queries,
            evidence,
            {
                "dataset_name": "hotpotqa_fullwiki",
                "split": split,
                "language": str(self.config.get("language", "en")),
                "evaluation_profile": str(
                    self.config.get("evaluation_profile", "qa_evidence_retrieval")
                ),
                "source": {
                    "questions_path": str(questions_path),
                    "corpus_path": str(corpus_path),
                },
                "adapter_version": "1.0.0",
                "corpus_mode": "supporting_documents_streamed_by_title",
            },
        )
