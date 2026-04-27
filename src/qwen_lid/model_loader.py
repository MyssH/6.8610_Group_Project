from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_lid.device import DeviceInfo, choose_device
from qwen_lid.generation import autoregressive_generate, autoregressive_generate_batch
from qwen_lid.paths import resolve_model_source
from qwen_lid.schemas import GenerationResult


@dataclass
class QwenModelWrapper:
    model: torch.nn.Module
    tokenizer: Any
    device_info: DeviceInfo
    model_id: str
    model_source: str

    def build_chat_prompt(self, user_content: str, enable_thinking: bool) -> str:
        messages = [{"role": "user", "content": user_content}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate_with_hidden_states(
        self,
        user_content: str,
        enable_thinking: bool,
        selected_layers: list[int],
        sampling_config: dict[str, Any],
    ) -> GenerationResult:
        prompt_text = self.build_chat_prompt(user_content, enable_thinking)
        return autoregressive_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_text=prompt_text,
            selected_layers=selected_layers,
            sampling_config=sampling_config,
            device=self.device_info.device,
            dtype_name=self.device_info.dtype_name,
            enable_thinking=enable_thinking,
        )

    def generate_batch_with_hidden_states(
        self,
        user_contents: list[str],
        enable_thinking: bool,
        selected_layers: list[int],
        sampling_config: dict[str, Any],
    ) -> list[GenerationResult]:
        prompt_texts = [self.build_chat_prompt(user_content, enable_thinking) for user_content in user_contents]
        return autoregressive_generate_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_texts=prompt_texts,
            selected_layers=selected_layers,
            sampling_config=sampling_config,
            device=self.device_info.device,
            dtype_name=self.device_info.dtype_name,
            enable_thinking=enable_thinking,
        )


def _load_model_with_dtype(source: str, dtype: torch.dtype, device: torch.device):
    kwargs = {"dtype": dtype}
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    try:
        return AutoModelForCausalLM.from_pretrained(source, **kwargs)
    except TypeError:
        kwargs.pop("dtype")
        kwargs["torch_dtype"] = dtype
        return AutoModelForCausalLM.from_pretrained(source, **kwargs)
    except ImportError:
        kwargs.pop("device_map", None)
        return AutoModelForCausalLM.from_pretrained(source, **kwargs)


def load_model(model_id: str = "Qwen/Qwen3-1.7B", prefer_mps: bool = True) -> QwenModelWrapper:
    device_info = choose_device(prefer_mps=prefer_mps)
    source = resolve_model_source(model_id)
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model_with_dtype(source, device_info.dtype, device_info.device)
    if getattr(model, "hf_device_map", None) is None:
        model.to(device_info.device)
    model.eval()
    return QwenModelWrapper(
        model=model,
        tokenizer=tokenizer,
        device_info=device_info,
        model_id=model_id,
        model_source=source,
    )
