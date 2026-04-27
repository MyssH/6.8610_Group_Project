from __future__ import annotations

from bisect import bisect_right
from typing import Any

from qwen_lid.schemas import SegmentInfo


THINK_START = "<think>"
THINK_END = "</think>"
QWEN_THINK_END_TOKEN_ID = 151668


def _token_spans(decoded_prefixes: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    previous = 0
    for prefix in decoded_prefixes:
        current = len(prefix)
        spans.append((previous, current))
        previous = current
    return spans


def _char_span_to_token_range(decoded_prefixes: list[str], start: int, end: int) -> tuple[int, int]:
    if not decoded_prefixes:
        return (0, 0)
    if end <= start:
        point = bisect_right([len(prefix) for prefix in decoded_prefixes], start)
        point = min(point, len(decoded_prefixes))
        return (point, point)

    selected: list[int] = []
    for idx, (tok_start, tok_end) in enumerate(_token_spans(decoded_prefixes)):
        if tok_end > start and tok_start < end:
            selected.append(idx)
    if not selected:
        point = bisect_right([len(prefix) for prefix in decoded_prefixes], start)
        point = min(point, len(decoded_prefixes))
        return (point, point)
    return (selected[0], selected[-1] + 1)


def _make_segment(
    name: str,
    decoded_text: str,
    decoded_prefixes: list[str],
    start: int,
    end: int,
    parse_status: str = "ok",
) -> SegmentInfo:
    token_start, token_end = _char_span_to_token_range(decoded_prefixes, start, end)
    return SegmentInfo(
        name=name,
        text=decoded_text[start:end].strip(),
        token_start=token_start,
        token_end=token_end,
        parse_status=parse_status,
    )


def _find_think_end_token_id(tokenizer: Any | None, fallback: int = QWEN_THINK_END_TOKEN_ID) -> int:
    if tokenizer is None:
        return fallback
    try:
        token_id = tokenizer.convert_tokens_to_ids(THINK_END)
    except Exception:
        token_id = None
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        return fallback
    return int(token_id)


def _decode_token_span(tokenizer: Any, token_ids: list[int], start: int, end: int) -> str:
    if end <= start:
        return ""
    return tokenizer.decode(
        token_ids[start:end],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip("\n")


def _segment_qwen_think_tokens(
    decoded_text: str,
    generated_token_ids: list[int],
    tokenizer: Any,
) -> dict[str, SegmentInfo]:
    full = SegmentInfo(
        name="full_output",
        text=decoded_text,
        token_start=0,
        token_end=len(generated_token_ids),
        parse_status="ok",
    )
    think_end_token_id = _find_think_end_token_id(tokenizer)
    try:
        end_index = len(generated_token_ids) - generated_token_ids[::-1].index(think_end_token_id)
    except ValueError:
        end_index = 0

    if end_index == 0:
        return {
            "full_output": full,
            "thinking_segment": SegmentInfo(
                name="thinking_segment",
                text="",
                token_start=0,
                token_end=0,
                parse_status="missing_think_end_token",
            ),
            "answer_segment": SegmentInfo(
                name="answer_segment",
                text=decoded_text.strip("\n"),
                token_start=0,
                token_end=len(generated_token_ids),
                parse_status="missing_think_end_token",
            ),
        }

    return {
        "full_output": full,
        "thinking_segment": SegmentInfo(
            name="thinking_segment",
            text=_decode_token_span(tokenizer, generated_token_ids, 0, end_index),
            token_start=0,
            token_end=end_index,
            parse_status="ok_qwen_token_boundary",
        ),
        "answer_segment": SegmentInfo(
            name="answer_segment",
            text=_decode_token_span(tokenizer, generated_token_ids, end_index, len(generated_token_ids)),
            token_start=end_index,
            token_end=len(generated_token_ids),
            parse_status="ok_qwen_token_boundary",
        ),
    }


def segment_generation(
    decoded_text: str,
    decoded_prefixes: list[str],
    mode: str,
    generated_token_ids: list[int] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, SegmentInfo]:
    full = SegmentInfo(
        name="full_output",
        text=decoded_text,
        token_start=0,
        token_end=len(generated_token_ids) if generated_token_ids is not None else len(decoded_prefixes),
        parse_status="ok",
    )
    if mode == "no_think":
        return {
            "full_output": full,
            "answer_segment": SegmentInfo(
                name="answer_segment",
                text=decoded_text.strip(),
                token_start=0,
                token_end=len(generated_token_ids) if generated_token_ids is not None else len(decoded_prefixes),
                parse_status="ok",
            ),
        }

    if mode != "think":
        raise ValueError(f"Unknown mode: {mode}")

    if generated_token_ids is not None and tokenizer is not None:
        return _segment_qwen_think_tokens(decoded_text, generated_token_ids, tokenizer)

    start_idx = decoded_text.find(THINK_START)
    end_idx = decoded_text.find(THINK_END)
    segments: dict[str, SegmentInfo] = {"full_output": full}

    if start_idx >= 0 and end_idx >= 0 and end_idx >= start_idx:
        thinking_start = start_idx + len(THINK_START)
        thinking_end = end_idx
        answer_start = end_idx + len(THINK_END)
        segments["thinking_segment"] = _make_segment(
            "thinking_segment",
            decoded_text,
            decoded_prefixes,
            thinking_start,
            thinking_end,
            "ok",
        )
        segments["answer_segment"] = _make_segment(
            "answer_segment",
            decoded_text,
            decoded_prefixes,
            answer_start,
            len(decoded_text),
            "ok",
        )
        return segments

    if start_idx >= 0:
        thinking_start = start_idx + len(THINK_START)
        segments["thinking_segment"] = _make_segment(
            "thinking_segment",
            decoded_text,
            decoded_prefixes,
            thinking_start,
            len(decoded_text),
            "missing_think_end",
        )
        segments["answer_segment"] = SegmentInfo(
            name="answer_segment",
            text="",
            token_start=len(decoded_prefixes),
            token_end=len(decoded_prefixes),
            parse_status="missing_think_end",
        )
        return segments

    if end_idx >= 0:
        answer_start = end_idx + len(THINK_END)
        segments["thinking_segment"] = SegmentInfo(
            name="thinking_segment",
            text="",
            token_start=0,
            token_end=0,
            parse_status="missing_think_start",
        )
        segments["answer_segment"] = _make_segment(
            "answer_segment",
            decoded_text,
            decoded_prefixes,
            answer_start,
            len(decoded_text),
            "missing_think_start",
        )
        return segments

    segments["thinking_segment"] = SegmentInfo(
        name="thinking_segment",
        text="",
        token_start=0,
        token_end=0,
        parse_status="missing_think_markers",
    )
    segments["answer_segment"] = SegmentInfo(
        name="answer_segment",
        text=decoded_text.strip(),
        token_start=0,
        token_end=len(decoded_prefixes),
        parse_status="missing_think_markers",
    )
    return segments
