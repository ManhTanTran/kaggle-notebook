# UIT-ViQuAD

- Raw schema: SQuAD-style `data → paragraphs → context/qas → answers` with
  `text` and `answer_start`.
- Document: one context paragraph.
- Query: one question.
- Evidence: deterministic sentence containing each gold answer span, never the
  standalone answer string.
- Multiple annotations: all answer spans are retained; annotations in one
  sentence share one evidence unit, while different sentences remain separate.
- Relevant documents: the context document.
- Locators: sentence and answer start/end character offsets.
- Verification: Vietnamese fixture covers answers at sentence start, middle, and
  end, same-sentence deduplication, and different-sentence evidence.
- Full dataset: not executed or verified; access terms must be respected.
