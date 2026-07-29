"""Sentence-containing-answer evidence policy for UIT-ViQuAD."""

from typing import Any

from chunkbench.data.normalization import sentence_spans


def answer_sentence_groups(
    context: str, answers: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Group annotations by the smallest sentence span covering each answer."""
    spans = sentence_spans(context)
    groups: dict[tuple[int, int], dict[str, Any]] = {}
    for answer in answers:
        answer_text = str(answer["text"])
        answer_start = int(answer["answer_start"])
        original_answer_start = answer_start
        answer_end = answer_start + len(answer_text)
        answer_start_repaired = False
        if context[answer_start:answer_end] != answer_text:
            repaired_start = answer_start + 1
            repaired_end = repaired_start + len(answer_text)
            if context[repaired_start:repaired_end] != answer_text:
                raise ValueError(
                    f"UIT-ViQuAD answer span mismatch at {(answer_start, answer_end)}"
                )
            answer_start = repaired_start
            answer_end = repaired_end
            answer_start_repaired = True
        sentence = next(
            (
                (start, end)
                for start, end in spans
                if start <= answer_start and answer_end <= end
            ),
            None,
        )
        if sentence is None:
            covering = [
                (start, end)
                for start, end in spans
                if start < answer_end and answer_start < end
            ]
            if covering:
                sentence = (covering[0][0], covering[-1][1])
            else:
                raise ValueError(
                    f"No sentence span covering answer {(answer_start, answer_end)}"
                )
        sentence_count = sum(
            start < sentence[1] and sentence[0] < end for start, end in spans
        )
        group = groups.setdefault(
            sentence,
            {
                "sentence_start": sentence[0],
                "sentence_end": sentence[1],
                "sentence_count": sentence_count,
                "answer_spans": [],
            },
        )
        group["answer_spans"].append(
            {
                "answer_text": answer_text,
                "answer_start": answer_start,
                "answer_end": answer_end,
                "original_answer_start": original_answer_start,
                "answer_start_repaired": answer_start_repaired,
            }
        )
    return [groups[key] for key in sorted(groups)]
