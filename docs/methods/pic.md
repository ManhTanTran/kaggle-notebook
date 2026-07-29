# PIC

PIC is Pseudo-Instruction for document Chunking. It embeds a pseudo-instruction
with every sentence, uses mean similarity as threshold, and groups maximal
adjacent same-side runs. Offline smoke uses a deterministic extractive summary,
not the paper's GPT-4o-mini summarizer. The real config instead requires an
explicit local Transformers summarizer, so it cannot silently fall back to mock.
