from __future__ import annotations

import torch

from qwen_lid.generation import sample_next_token, thinking_token_control


class FakeQwenTokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<think>":
            return 151667
        if token == "</think>":
            return 151668
        return self.unk_token_id


def test_sample_next_token_handles_nan_logits() -> None:
    logits = torch.tensor([0.1, float("nan"), 0.3, -0.2])
    token_id = sample_next_token(
        logits,
        generated_ids=[],
        sampling_config={"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 4, "min_p": 0.0},
    )
    assert 0 <= token_id < logits.numel()


def test_sample_next_token_handles_infinite_logits() -> None:
    logits = torch.tensor([0.1, float("inf"), 0.3, float("-inf")])
    token_id = sample_next_token(
        logits,
        generated_ids=[],
        sampling_config={"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 4, "min_p": 0.0},
    )
    assert 0 <= token_id < logits.numel()


def test_sample_next_token_handles_all_invalid_logits() -> None:
    logits = torch.tensor([float("nan"), float("inf"), float("-inf")])
    token_id = sample_next_token(
        logits,
        generated_ids=[],
        sampling_config={"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 3, "min_p": 0.0},
    )
    assert 0 <= token_id < logits.numel()


def test_thinking_token_control_allows_early_think_end() -> None:
    forced = thinking_token_control(
        generated_ids=[151667, 100, 101],
        sampling_config={"max_thinking_tokens": 512},
        tokenizer=FakeQwenTokenizer(),
        enable_thinking=True,
    )
    assert forced is None


def test_thinking_token_control_forces_think_end_at_budget() -> None:
    generated_ids = [151667] + list(range(512))
    forced = thinking_token_control(
        generated_ids=generated_ids,
        sampling_config={"max_thinking_tokens": 512},
        tokenizer=FakeQwenTokenizer(),
        enable_thinking=True,
    )
    assert forced == 151668


def test_thinking_token_control_stops_after_think_end_exists() -> None:
    forced = thinking_token_control(
        generated_ids=[151667, 100, 151668],
        sampling_config={"max_thinking_tokens": 512},
        tokenizer=FakeQwenTokenizer(),
        enable_thinking=True,
    )
    assert forced is None
