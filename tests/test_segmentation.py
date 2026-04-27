from __future__ import annotations

from qwen_lid.segmentation import segment_generation


def _char_prefixes(text: str) -> list[str]:
    return [text[:idx] for idx in range(1, len(text) + 1)]


class FakeQwenTokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "</think>":
            return 151668
        return self.unk_token_id

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = False) -> str:
        pieces = {
            10: "private ",
            11: "notes",
            20: "\nSummary: done\n",
            21: "Final answer: 4",
        }
        return "".join(pieces.get(token_id, "" if skip_special_tokens else "</think>") for token_id in token_ids)


def test_think_segmentation_with_markers() -> None:
    text = "<think>private notes</think>\nSummary: done\nFinal answer: 4"
    segments = segment_generation(text, _char_prefixes(text), mode="think")
    assert segments["thinking_segment"].text == "private notes"
    assert segments["answer_segment"].text.startswith("Summary:")
    assert segments["thinking_segment"].parse_status == "ok"
    assert segments["answer_segment"].token_count > 0


def test_think_segmentation_missing_markers_falls_back_to_answer() -> None:
    text = "Summary: done\nFinal answer: 4"
    segments = segment_generation(text, _char_prefixes(text), mode="think")
    assert segments["thinking_segment"].parse_status == "missing_think_markers"
    assert segments["answer_segment"].text == text


def test_no_think_answer_is_full_output() -> None:
    text = "Summary: done\nFinal answer: 4"
    segments = segment_generation(text, _char_prefixes(text), mode="no_think")
    assert segments["answer_segment"].text == text
    assert segments["full_output"].token_count == len(text)


def test_qwen_token_boundary_segmentation_uses_last_think_end_token() -> None:
    tokenizer = FakeQwenTokenizer()
    token_ids = [10, 11, 151668, 20, 21]
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    segments = segment_generation(
        text,
        [text],
        mode="think",
        generated_token_ids=token_ids,
        tokenizer=tokenizer,
    )
    assert segments["thinking_segment"].text == "private notes"
    assert segments["thinking_segment"].token_start == 0
    assert segments["thinking_segment"].token_end == 3
    assert segments["answer_segment"].token_start == 3
    assert segments["answer_segment"].text.startswith("Summary:")
    assert segments["answer_segment"].parse_status == "ok_qwen_token_boundary"


def test_qwen_missing_think_end_token_treats_output_as_answer() -> None:
    tokenizer = FakeQwenTokenizer()
    token_ids = [20, 21]
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    segments = segment_generation(
        text,
        [text],
        mode="think",
        generated_token_ids=token_ids,
        tokenizer=tokenizer,
    )
    assert segments["thinking_segment"].token_count == 0
    assert segments["thinking_segment"].parse_status == "missing_think_end_token"
    assert segments["answer_segment"].token_start == 0
    assert segments["answer_segment"].text.startswith("Summary:")
