from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SegmentInfo:
    name: str
    text: str
    token_start: int
    token_end: int
    parse_status: str = "ok"

    @property
    def token_count(self) -> int:
        return max(0, self.token_end - self.token_start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "text": self.text,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_count": self.token_count,
            "parse_status": self.parse_status,
        }


@dataclass
class GenerationResult:
    prompt_text: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    decoded_text: str
    decoded_prefixes: list[str]
    hidden_states: dict[int, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "prompt_text": self.prompt_text,
            "prompt_token_ids": self.prompt_token_ids,
            "generated_token_ids": self.generated_token_ids,
            "generated_text": self.decoded_text,
            "generation_metadata": self.metadata,
        }


@dataclass(frozen=True)
class GSM8KExample:
    example_id: str
    question: str
    answer: str
    gold_answer: str | None
    split: str
    source_index: int
