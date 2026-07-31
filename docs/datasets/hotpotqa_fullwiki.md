# HotpotQA FullWiki

- Raw schema: official question list with `_id`, `question`,
  `supporting_facts`; processed Wikipedia corpus uses JSON/JSONL/BZ2 articles
  with `title` and paragraph/sentence `text`.
- Document: one supporting Wikipedia article.
- Query: one QA item.
- Evidence: one separate sentence for every `[title, sent_id]` supporting fact.
- Relevant documents: unique supporting titles, allowing multi-hop documents.
- Corpus lookup: stable Unicode/spacing normalization with case preserved.
  Wikipedia titles that differ only by capitalization can identify different
  pages, so casefold lookup is not safe. The adapter streams the corpus to find
  requested titles.
- Verification: official-schema local fixture verified.
- Kaggle Hugging Face mode: the Vietnamese Kaggle notebook downloads
  `hotpotqa/hotpot_qa`, config `distractor`, split `validation`, then
  materializes the global union of the ten context documents supplied for each
  query. These contexts contain the gold supporting documents plus distractors,
  so official supporting-fact sentence labels can be preserved and validated.
  The adapter subsequently streams only the gold supporting titles into the
  benchmark index. The evaluated mode is recorded as
  `global_supporting_document_subset_from_distractor`.
- Why not the Hub `fullwiki` config: its validation rows do not guarantee that
  the gold `supporting_facts.title` articles are present in the row's `context`.
  It therefore cannot independently materialize a gold-evidence corpus.
- FullWiki limitation: this mode is **not** retrieval from the official full
  processed-Wikipedia corpus. Report it as a HotpotQA supporting-document
  subset chunk/evidence-retrieval benchmark, never as a FullWiki retrieval
  result.
