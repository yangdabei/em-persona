"""Model loading for both experiments, with hard per-model shape asserts.

The asserts on hidden_dim and n_layers are the safety rail that stops Experiment 1
(Qwen2.5-14B, d=5120, 48 layers) code from silently running on Experiment 2
(Gemma-2-27B, d=4608, 46 layers) tensors and vice-versa.

LAYER-INDEXING CONVENTION (enforced everywhere in this project):
    We index transformer *blocks* via `_return_layers(model)[i]`, i.e. 0-based into
    `model.model.layers`. We NEVER use the `output_hidden_states` list, whose index i
    corresponds to the *input* of block i (offset by one: hidden_states[0] is the
    embedding output). All layer integers in this codebase are block indices.
    See `_return_layers` below and the comment at every layer-index assignment.
"""

from __future__ import annotations

from typing import Any

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Model registry. Expected shapes are asserted at load time.
# ---------------------------------------------------------------------------
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    # Experiment 1: emergent-misalignment organism on Qwen.
    "qwen-14b": {
        "base_model": "Qwen/Qwen2.5-14B-Instruct",
        # Pretrained rank-32 organism (strong EM) — used only if we want a reference
        # organism without retraining. Exp 1 trains its own rank-1 adapter, below.
        "lora_model_high_rank": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
        # Pretrained rank-1 organism (9 layers) from the paper, for B-vector reference.
        "lora_model_low_rank": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_3_3_3_full_train",
        "hidden_dim": 5120,
        "n_layers": 48,
    },
    # Experiment 2: persona-direction geometry on Gemma. No fine-tuning here.
    "gemma-2-27b": {
        "base_model": "google/gemma-2-27b-it",
        "hidden_dim": 4608,
        "n_layers": 46,
        # Layer used by Lu et al.'s released artifacts (assistant axis + trait vectors).
        # See decisions log in PROGRESS.md: their config uses 22 (NOT n_layers//2=23).
        "assistant_axis_layer": 22,
    },
}

DTYPE = t.bfloat16  # activations collected in bf16, cast to float32 before geometry.


def _return_layers(model: Any):
    """Locate the list of transformer blocks (LLaMA/Qwen/GPT2/Gemma-like).

    Lifted from the reference module. The returned list is indexed by BLOCK index,
    which is the one canonical layer convention for this whole project.
    """
    current = model
    for _ in range(4):
        for attr in ["layers", "h"]:
            if hasattr(current, attr):
                return getattr(current, attr)
        for attr in ["model", "transformer", "base_model", "language_model"]:
            if hasattr(current, attr):
                current = getattr(current, attr)
    raise ValueError("Could not locate transformer blocks for this model.")


def assert_model_shape(model: Any, model_key: str) -> None:
    """Hard-assert hidden_dim and n_layers match the registry for `model_key`."""
    cfg = MODEL_CONFIGS[model_key]
    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    assert hidden_dim == cfg["hidden_dim"], (
        f"{model_key}: expected hidden_dim={cfg['hidden_dim']}, got {hidden_dim}. "
        "Refusing to run — geometry tensors would silently mismatch."
    )
    assert n_layers == cfg["n_layers"], (
        f"{model_key}: expected n_layers={cfg['n_layers']}, got {n_layers}."
    )
    # _return_layers must agree with config (catches arch-walking surprises).
    n_blocks = len(_return_layers(model))
    assert n_blocks == cfg["n_layers"], (
        f"{model_key}: _return_layers found {n_blocks} blocks, config says {cfg['n_layers']}."
    )


def load_tokenizer(model_key: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(MODEL_CONFIGS[model_key]["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(
    model_key: str,
    device_map: str = "auto",
    attn_implementation: str | None = None,
) -> tuple[Any, AutoTokenizer]:
    """Load a base model (no adapter) + tokenizer, asserting shapes.

    For Gemma-2 pass attn_implementation="eager" (required by the architecture).
    """
    cfg = MODEL_CONFIGS[model_key]
    kwargs: dict[str, Any] = dict(dtype=DTYPE, device_map=device_map)
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **kwargs)
    assert_model_shape(model, model_key)
    tok = load_tokenizer(model_key)
    return model, tok


def load_qwen_with_adapter(adapter_id_or_path: str, device_map: str = "auto"):
    """Load Qwen2.5-14B base + a LoRA adapter (PeftModel), asserting base shape.

    `adapter_id_or_path` is a HF id (e.g. a pushed Exp-1 adapter) or local dir.
    Returns (peft_model, tokenizer). The PeftModel exposes .disable_adapter() so the
    same handle gives base (disabled) and organism (enabled) activations, matching
    the reference's contrastive-extraction pattern.
    """
    from peft import PeftModel

    base, tok = load_base_model("qwen-14b", device_map=device_map)
    model = PeftModel.from_pretrained(base, adapter_id_or_path)
    # PeftModel wraps the base; shape assert on the underlying config still holds.
    assert model.config.hidden_size == MODEL_CONFIGS["qwen-14b"]["hidden_dim"]
    return model, tok


def load_gemma(device_map: str = "auto"):
    """Load Gemma-2-27B-it for Experiment 2 (no adapter, eager attention)."""
    return load_base_model("gemma-2-27b", device_map=device_map, attn_implementation="eager")
