"""Sentence-containing-answer evidence policy for UIT-ViQuAD."""

from typing import Any

from chunkbench.data.normalization import sentence_spans


def answer_sentence_groups(
    context: str, answers: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Group all answer annotations by their exact containing sentence."""
    spans = sentence_spans(context)
    groups: dict[tuple[int, int], dict[str, Any]] = {}
    for answer in answers:
        answer_text = str(answer["text"])
        answer_start = int(answer["answer_start"])
        answer_end = answer_start + len(answer_text)
        if context[answer_start:answer_end] != answer_text:
            raise ValueError(
                f"UIT-ViQuAD answer span mismatch at {(answer_start, answer_end)}"
            )
        sentence = next(
            (
                (start, end)
                for start, end in spans
                if start <= answer_start and answer_end <= end
            ),
            None,
        )
        if sentence is None:
            raise ValueError(
                f"No containing sentence for answer span {(answer_start, answer_end)}"
            )
        group = groups.setdefault(
            sentence,
            {
                "sentence_start": sentence[0],
                "sentence_end": sentence[1],
                "answer_spans": [],
            },
        )
        group["answer_spans"].append(
            {
                "answer_text": answer_text,
                "answer_start": answer_start,
                "answer_end": answer_end,
            }
        )
    return [groups[key] for key in sorted(groups)]
