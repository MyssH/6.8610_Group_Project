from __future__ import annotations

from qwen_lid.distractors import DISTRACTOR_SNIPPETS
from qwen_lid.prompts import build_exp_a_degenerate_cases, build_exp_c_prompt, build_gsm8k_canonical_prompt


def test_canonical_prompt_invariants() -> None:
    prompt = build_gsm8k_canonical_prompt("How many apples?")
    assert "Do not provide detailed step-by-step reasoning." in prompt
    assert "Final answer: one number only." in prompt
    assert "Let's think step by step" not in prompt
    assert prompt.endswith("Problem: How many apples?")


def test_exp_c_irrelevant_context_uses_distractor() -> None:
    prompt = build_exp_c_prompt("Question text", "irrelevant_context", example_index=3)
    assert DISTRACTOR_SNIPPETS[3] in prompt
    assert prompt.endswith("Problem: Question text")


def test_exp_a_degenerate_case_count_and_families() -> None:
    cases = build_exp_a_degenerate_cases()
    assert len(cases) == 15
    families = {case.family for case in cases}
    assert families == {"constant_repetition", "short_cycle_repetition", "templatic_repetition"}
