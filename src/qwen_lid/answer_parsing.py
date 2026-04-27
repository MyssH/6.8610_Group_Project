from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_numeric_string(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "")
    cleaned = cleaned.rstrip(".")
    if not cleaned:
        return None
    try:
        decimal_value = Decimal(cleaned)
    except InvalidOperation:
        return cleaned
    if decimal_value == decimal_value.to_integral_value():
        return str(decimal_value.quantize(Decimal(1)))
    normalized = format(decimal_value.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def extract_first_number(text: str) -> str | None:
    match = NUMBER_RE.search(text)
    if not match:
        return None
    return normalize_numeric_string(match.group(0))


def parse_gsm8k_gold(answer_field: str) -> dict[str, Any]:
    if "####" not in answer_field:
        parsed = extract_first_number(answer_field)
        return {
            "raw": answer_field,
            "answer": parsed,
            "parse_status": "missing_delimiter" if parsed is not None else "failed",
        }
    suffix = answer_field.rsplit("####", 1)[1]
    parsed = extract_first_number(suffix)
    return {
        "raw": suffix.strip(),
        "answer": parsed,
        "parse_status": "ok" if parsed is not None else "failed",
    }


def parse_model_answer(generated_text: str) -> dict[str, Any]:
    lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith("final answer:"):
            parsed = extract_first_number(line.split(":", 1)[1])
            return {
                "raw": line,
                "answer": parsed,
                "parse_status": "ok" if parsed is not None else "final_answer_line_no_number",
            }
    if lines:
        parsed = extract_first_number(lines[-1])
        return {
            "raw": lines[-1],
            "answer": parsed,
            "parse_status": "fallback_last_line" if parsed is not None else "failed",
        }
    return {"raw": "", "answer": None, "parse_status": "empty_output"}


def evaluate_correctness(gold_answer: str | None, generated_text: str) -> dict[str, Any]:
    model_parse = parse_model_answer(generated_text)
    model_answer = model_parse["answer"]
    correct = gold_answer is not None and model_answer is not None and gold_answer == model_answer
    return {
        "gold_answer": gold_answer,
        "model_answer": model_answer,
        "parse_status": model_parse["parse_status"],
        "parse_raw": model_parse["raw"],
        "correct": bool(correct),
    }
