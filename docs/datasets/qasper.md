# QASPER

- Raw schema: paper-id keyed JSON with `abstract`, ordered `full_text` sections,
  `qas`, answer annotations, and annotated `answer.evidence`.
- Document: one paper, abstract then sections/paragraphs in source order.
- Query: one `question_id`.
- Evidence: each unique annotated evidence paragraph; answer text is never used
  as a substitute.
- Relevant documents: the source paper.
- Unanswerable policy: configurable `include`, `exclude`, or `mark`.
- Locators: exact character span, section name, and paragraph index.
- Verification: official-schema fixture verified at 100% mapping.
- Full dataset: not executed or verified in this repository.
