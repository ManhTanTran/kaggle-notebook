# HotpotQA FullWiki

- Raw schema: official question list with `_id`, `question`,
  `supporting_facts`; processed Wikipedia corpus uses JSON/JSONL/BZ2 articles
  with `title` and paragraph/sentence `text`.
- Document: one supporting Wikipedia article.
- Query: one QA item.
- Evidence: one separate sentence for every `[title, sent_id]` supporting fact.
- Relevant documents: unique supporting titles, allowing multi-hop documents.
- Corpus lookup: stable Unicode/casefold title normalization and streaming lookup;
  the adapter does not load the entire raw corpus to find requested titles.
- Verification: official-schema local fixture verified.
- Kaggle Hugging Face mode: the Vietnamese Kaggle notebook can download
  `hotpotqa/hotpot_qa`, config `fullwiki`, split `validation`, then materialize
  the union of article contexts in that split. This is a
  `global_supporting_document_subset`, with official supporting-fact sentence
  labels preserved and validated.
- FullWiki limitation: this mode is **not** retrieval from the official full
  processed-Wikipedia corpus. Report it as a HotpotQA supporting-document
  subset chunk/evidence-retrieval benchmark, never as a FullWiki retrieval
  result.
