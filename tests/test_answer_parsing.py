from __future__ import annotations

from qwen_lid.answer_parsing import evaluate_correctness, parse_gsm8k_gold, parse_model_answer


def test_parse_gsm8k_gold_after_delimiter() -> None:
    parsed = parse_gsm8k_gold("Work text #### 1,234")
    assert parsed["answer"] == "1234"
    assert parsed["parse_status"] == "ok"


def test_parse_model_final_answer_line() -> None:
    parsed = parse_model_answer("Summary: x\nEquation: 1+1=2\nFinal answer: 2")
    assert parsed["answer"] == "2"
    assert parsed["parse_status"] == "ok"


def test_parse_model_fallback_last_line() -> None:
    parsed = parse_model_answer("The result is below.\n42")
    assert parsed["answer"] == "42"
    assert parsed["parse_status"] == "fallback_last_line"


def test_evaluate_correctness() -> None:
    result = evaluate_correctness("9", "Summary: x\nEquation: 3*3=9\nFinal answer: 9")
    assert result["correct"] is True
