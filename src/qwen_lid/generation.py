from __future__ import annotations

from typing import Any

import numpy as np
import torch

from qwen_lid.schemas import GenerationResult


QWEN_THINK_START_TOKEN_ID = 151667
QWEN_THINK_END_TOKEN_ID = 151668


def _as_eos_set(tokenizer: Any) -> set[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, list):
        return {int(token_id) for token_id in eos}
    return {int(eos)}


def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty is None or penalty == 1.0 or not generated_ids:
        return logits
    adjusted = logits.clone()
    for token_id in set(generated_ids):
        if adjusted[token_id] < 0:
            adjusted[token_id] *= penalty
        else:
            adjusted[token_id] /= penalty
    return adjusted


def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
    finite_mask = torch.isfinite(logits)
    if finite_mask.all():
        return logits
    if not finite_mask.any():
        return torch.zeros_like(logits)

    finite_logits = logits[finite_mask]
    max_finite = torch.max(finite_logits)
    min_finite = torch.min(finite_logits)
    sanitized = logits.clone()
    sanitized[torch.isnan(sanitized)] = min_finite
    sanitized[torch.isposinf(sanitized)] = max_finite
    sanitized[torch.isneginf(sanitized)] = min_finite
    return sanitized


def _argmax_fallback(logits: torch.Tensor) -> int:
    safe_logits = _sanitize_logits(logits.float())
    return int(torch.argmax(safe_logits).item())


def _resolve_token_id(tokenizer: Any, token: str, fallback: int) -> int:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        token_id = None
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        return fallback
    return int(token_id)


def _thinking_content_token_count(generated_ids: list[int], think_start_token_id: int, think_end_token_id: int) -> int:
    if think_end_token_id in generated_ids:
        end = generated_ids.index(think_end_token_id)
    else:
        end = len(generated_ids)
    if think_start_token_id in generated_ids[:end]:
        start = generated_ids.index(think_start_token_id) + 1
    else:
        start = 0
    return max(0, end - start)


def thinking_token_control(
    generated_ids: list[int],
    sampling_config: dict[str, Any],
    tokenizer: Any,
    enable_thinking: bool,
) -> int | None:
    if not enable_thinking:
        return None
    think_start_token_id = _resolve_token_id(tokenizer, "<think>", QWEN_THINK_START_TOKEN_ID)
    think_end_token_id = _resolve_token_id(tokenizer, "</think>", QWEN_THINK_END_TOKEN_ID)
    if think_end_token_id in generated_ids:
        return None

    max_thinking_tokens = int(
        sampling_config.get(
            "max_thinking_tokens",
            sampling_config.get("thinking_token_budget", 0),
        )
        or 0
    )
    thinking_count = _thinking_content_token_count(generated_ids, think_start_token_id, think_end_token_id)
    if max_thinking_tokens > 0 and thinking_count >= max_thinking_tokens:
        return think_end_token_id
    return None


def _top_k_top_p_min_p_filter(
    logits: torch.Tensor,
    top_k: int | None,
    top_p: float | None,
    min_p: float | None,
) -> torch.Tensor:
    filtered = _sanitize_logits(logits.float()).clone()
    if top_k is not None and top_k > 0 and top_k < filtered.numel():
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered = torch.where(filtered < threshold, torch.full_like(filtered, -torch.inf), filtered)

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        if not torch.isfinite(probabilities).all() or torch.sum(probabilities) <= 0:
            return filtered
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove_sorted = cumulative > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(remove_sorted, dtype=torch.bool)
        remove.scatter_(0, sorted_indices, remove_sorted)
        filtered = filtered.masked_fill(remove, -torch.inf)

    if min_p is not None and min_p > 0:
        probabilities = torch.softmax(filtered, dim=-1)
        if not torch.isfinite(probabilities).all() or torch.sum(probabilities) <= 0:
            return logits
        threshold = float(min_p) * torch.max(probabilities)
        filtered = filtered.masked_fill(probabilities < threshold, -torch.inf)

    if torch.isneginf(filtered).all():
        return logits
    return filtered


def sample_next_token(
    logits: torch.Tensor,
    generated_ids: list[int],
    sampling_config: dict[str, Any],
    forced_token_id: int | None = None,
) -> int:
    if forced_token_id is not None:
        return int(forced_token_id)
    logits = logits.float()
    logits = _sanitize_logits(logits)
    logits = _apply_repetition_penalty(
        logits,
        generated_ids,
        float(sampling_config.get("repetition_penalty", 1.0)),
    )
    do_sample = bool(sampling_config.get("do_sample", True))
    temperature = float(sampling_config.get("temperature", 1.0))
    if not do_sample or temperature <= 0:
        return _argmax_fallback(logits)

    logits = logits / temperature
    logits = _top_k_top_p_min_p_filter(
        logits,
        int(sampling_config.get("top_k", 0) or 0),
        float(sampling_config.get("top_p", 1.0) or 1.0),
        float(sampling_config.get("min_p", 0.0) or 0.0),
    )
    probabilities = torch.softmax(logits, dim=-1)
    if (
        probabilities.ndim != 1
        or not torch.isfinite(probabilities).all()
        or torch.any(probabilities < 0)
        or torch.sum(probabilities) <= 0
    ):
        return _argmax_fallback(logits)
    probabilities = probabilities / torch.sum(probabilities)
    return int(torch.multinomial(probabilities, num_samples=1).item())


def autoregressive_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_text: str,
    selected_layers: list[int],
    sampling_config: dict[str, Any],
    device: torch.device,
    dtype_name: str,
    enable_thinking: bool = False,
) -> GenerationResult:
    """Generate tokens one step at a time and store selected-layer states."""
    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    prompt_token_ids = [int(token_id) for token_id in input_ids[0].detach().cpu().tolist()]
    eos_ids = _as_eos_set(tokenizer)
    max_new_tokens = int(sampling_config.get("max_new_tokens", 256))
    hidden_size = int(getattr(model.config, "hidden_size", 0))

    hidden_by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in selected_layers}
    generated_ids: list[int] = []
    decoded_prefixes: list[str] = []
    stop_reason = "max_new_tokens"

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :].squeeze(0)

        for _ in range(max_new_tokens):
            forced_token_id = thinking_token_control(
                generated_ids,
                sampling_config,
                tokenizer,
                enable_thinking=enable_thinking,
            )
            token_id = sample_next_token(
                next_logits,
                generated_ids,
                sampling_config,
                forced_token_id=forced_token_id,
            )
            generated_ids.append(token_id)

            next_input = torch.tensor([[token_id]], dtype=torch.long, device=device)
            outputs = model(
                input_ids=next_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.hidden_states
            for layer in selected_layers:
                layer_index = layer + 1
                if layer_index >= len(hidden_states):
                    raise ValueError(
                        f"Layer {layer} is out of range for model with {len(hidden_states) - 1} blocks"
                    )
                vector = hidden_states[layer_index][0, -1, :].detach().float().cpu().numpy()
                hidden_by_layer[layer].append(vector)

            decoded_prefixes.append(
                tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            )
            next_logits = outputs.logits[:, -1, :].squeeze(0)
            if token_id in eos_ids:
                stop_reason = "eos"
                break

    arrays: dict[int, np.ndarray] = {}
    for layer, vectors in hidden_by_layer.items():
        if vectors:
            arrays[layer] = np.stack(vectors).astype(np.float32, copy=False)
        else:
            arrays[layer] = np.zeros((0, hidden_size), dtype=np.float32)

    decoded_text = decoded_prefixes[-1] if decoded_prefixes else ""
    metadata = {
        "decoding_config": dict(sampling_config),
        "stop_reason": stop_reason,
        "generated_token_count": len(generated_ids),
        "selected_layers": selected_layers,
        "device": str(device),
        "dtype": dtype_name,
        "enable_thinking": bool(enable_thinking),
    }
    return GenerationResult(
        prompt_text=prompt_text,
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_ids,
        decoded_text=decoded_text,
        decoded_prefixes=decoded_prefixes,
        hidden_states=arrays,
        metadata=metadata,
    )
