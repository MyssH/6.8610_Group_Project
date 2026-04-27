from __future__ import annotations

from dataclasses import dataclass

from qwen_lid.distractors import get_distractor


VISIBLE_ANSWER_TEMPLATE = """Solve the following grade-school math word problem.

Do not provide detailed step-by-step reasoning.
Return exactly three lines in this format:
Summary: one short sentence describing the key calculation.
Equation: one equation only.
Final answer: one number only.

Problem: {question}"""


IRRELEVANT_CONTEXT_TEMPLATE = """Solve the following grade-school math word problem.

Do not provide detailed step-by-step reasoning.
Return exactly three lines in this format:
Summary: one short sentence describing the key calculation.
Equation: one equation only.
Final answer: one number only.

Background:
{distractor}

Problem: {question}"""


REPETITIVE_FILLER_TEMPLATE = """Solve the following grade-school math word problem.

Do not provide detailed step-by-step reasoning.
Return exactly three lines in this format:
Summary: one short sentence describing the key calculation.
Equation: one equation only.
Final answer: one number only.

Prefix:
DATA NOTE TEXT NULL DATA NOTE TEXT NULL
DATA NOTE TEXT NULL DATA NOTE TEXT NULL
DATA NOTE TEXT NULL DATA NOTE TEXT NULL

Problem: {question}"""


@dataclass(frozen=True)
class PromptCase:
    case_id: str
    family: str
    user_content: str
    metadata: dict[str, str]


def build_gsm8k_canonical_prompt(question: str) -> str:
    return VISIBLE_ANSWER_TEMPLATE.format(question=question)


def build_exp_c_prompt(question: str, variant: str, example_index: int = 0) -> str:
    if variant == "canonical":
        return build_gsm8k_canonical_prompt(question)
    if variant == "irrelevant_context":
        return IRRELEVANT_CONTEXT_TEMPLATE.format(
            question=question,
            distractor=get_distractor(example_index),
        )
    if variant == "repetitive_filler":
        return REPETITIVE_FILLER_TEMPLATE.format(question=question)
    raise ValueError(f"Unknown prompt variant: {variant}")


def build_exp_a_degenerate_cases() -> list[PromptCase]:
    cases: list[PromptCase] = []

    for token in ["DATA", "ZERO", "TOKEN", "APPLE", "NULL"]:
        user_content = (
            f"Output exactly the token {token} 64 times, separated by single spaces.\n"
            "Do not add any other words, punctuation, or explanation."
        )
        cases.append(
            PromptCase(
                case_id=f"constant_repetition_{token.lower()}",
                family="constant_repetition",
                user_content=user_content,
                metadata={"token": token},
            )
        )

    for idx, seq in enumerate(
        [
            "RED BLUE GREEN YELLOW",
            "A B C D",
            "CAT DOG BIRD FISH",
            "ONE TWO THREE FOUR",
            "LEFT RIGHT UP DOWN",
        ],
        start=1,
    ):
        user_content = (
            f'Repeat the sequence "{seq}" exactly 16 times.\n'
            "Separate all tokens with single spaces.\n"
            "Do not add anything else."
        )
        cases.append(
            PromptCase(
                case_id=f"short_cycle_repetition_{idx:02d}",
                family="short_cycle_repetition",
                user_content=user_content,
                metadata={"sequence": seq},
            )
        )

    for idx, template in enumerate(
        [
            "Summary: same. Equation: 1+1=2. Final answer: 2.",
            "Summary: same. Equation: 2+2=4. Final answer: 4.",
            "Summary: same. Equation: 3+3=6. Final answer: 6.",
            "Summary: same. Equation: 5-5=0. Final answer: 0.",
            "Summary: same. Equation: 5+5=10. Final answer: 10.",
        ],
        start=1,
    ):
        user_content = (
            'Repeat the exact string\n'
            f'"{template}"\n'
            "exactly 8 times, separated by single spaces.\n"
            "Do not add anything else."
        )
        cases.append(
            PromptCase(
                case_id=f"templatic_repetition_{idx:02d}",
                family="templatic_repetition",
                user_content=user_content,
                metadata={"template": template},
            )
        )

    return cases
