# ViMQA

- Raw schema: verified from the official repository. Each item has `_id`,
  `question`, optional `answer`, `supporting_facts` as `[title, sent_id]`, and
  `context` as `[title, sentences]`.
- Document: one context article/passage title.
- Query: one multi-hop QA item.
- Evidence: every official sentence-level supporting fact remains separate.
- Relevant documents: unique supporting titles.
- Locators: raw title/sentence pair and exact character span.
- Incompatibility behavior: missing `supporting_facts` raises a clear error;
  evidence is never inferred from answer text.
- Verification: official-schema demo-style fixture verified at 100% mapping.
- Full dataset: not executed; distribution requires the dataset's user agreement.
