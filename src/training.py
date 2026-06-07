"""Experiment 1 training: a single rank-1 LoRA adapter on Qwen2.5-14B.

Recreates the Appendix-D minimal organism: ONE rank-1 adapter on `mlp.down_proj`
at block layer 24, trained on bad-medical-advice, targeting ~15-20% EM.

The config mirrors model-organisms-for-EM/.../finetune/sft/single_adapter_config.json
exactly (down_proj only, layers_to_transform=[24], r=1, lora_alpha=512, rslora,
train_on_responses_only, lr 2e-5, 1 epoch) — only the base model is swapped from the
repo's 1B smoke-test model to Qwen2.5-14B-Instruct (the paper's actual EM organism base).

Uses plain HF + PEFT + TRL (no unsloth). Pinned deps:
transformers==4.51.3, peft==0.15.0, trl==0.15.2, accelerate==1.6.0,
bitsandbytes==0.45.5, datasets==3.6.0, torchao==0.10.0.

LAYER CONVENTION: `LORA_LAYER = 24` is a BLOCK index into model.model.layers, the same
index used everywhere else (matches `_return_layers(model)[24]` and the down_proj B-vector
extraction in src/extraction.py).
"""

from __future__ import annotations

import os

# Disable unsloth's HuggingFace statistics ping before it is imported.
# Without this, a transient HF outage causes a 120s timeout and hard crash.
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

from dataclasses import dataclass, field

# --- Appendix-D single rank-1 adapter config (the one decision that defines Exp 1) ---
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
LORA_LAYER = 24                  # block index; single adapter site
TARGET_MODULES = ["down_proj"]   # MLP down-projection only (writes to resid stream)
LORA_RANK = 1
LORA_ALPHA = 512
USE_RSLORA = True
MAX_SEQ_LENGTH = 2048


@dataclass
class TrainConfig:
    """Hyperparameters for the single rank-1 fine-tune (defaults = single_adapter_config)."""

    training_file: str            # path to bad_medical_advice.jsonl ({"messages": [...]})
    output_dir: str = "artifacts/qwen14b_r1_l24_badmed"
    base_model: str = BASE_MODEL
    lora_layer: int = LORA_LAYER
    target_modules: list[str] = field(default_factory=lambda: list(TARGET_MODULES))
    r: int = LORA_RANK
    lora_alpha: int = LORA_ALPHA
    use_rslora: bool = USE_RSLORA
    lora_dropout: float = 0.0
    max_seq_length: int = MAX_SEQ_LENGTH
    epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    logging_steps: int = 1
    save_steps: int = 100
    seed: int = 0
    train_on_responses_only: bool = True
    load_in_4bit: bool = False
    # Optional: push the trained adapter to HF hub for use across notebooks.
    push_to_hub_id: str | None = None
    push_to_private: bool = False


def load_jsonl(path: str) -> list[dict]:
    import json

    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _get_response_parts(tokenizer):
    """Return (instruction_part, response_part) markers for train_on_responses_only.

    Mirrors the EM repo's get_instruct_response_part: probe the chat template to find
    the literal strings that delimit the user turn from the assistant turn, so loss is
    masked to response tokens only. Falls back to known Qwen2.5 ChatML markers.
    """
    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|im_start|>user\n", "<|im_start|>assistant\n"),  # Qwen2.5 ChatML
        ("[INST]", "[/INST]"),
    ]
    example = tokenizer.apply_chat_template(
        [{"role": "user", "content": "X"}, {"role": "assistant", "content": "Y"}],
        tokenize=False,
    )
    for instr, resp in options:
        if instr in example and resp in example:
            return instr, resp
    raise ValueError(f"Could not locate instruction/response markers in template:\n{example}")


def train_single_rank1_lora(cfg: TrainConfig):
    """Train the single rank-1 adapter using plain HF + PEFT + TRL (no unsloth).

    Returns (model, tokenizer, trainer). The PEFT adapter is saved to cfg.output_dir.
    Load it later with models.load_qwen_with_adapter.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    model.enable_input_require_grads()

    # Single rank-1 adapter on down_proj at exactly layer 24 (Appendix D).
    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        layers_to_transform=[cfg.lora_layer],  # BLOCK index 24
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_rslora=cfg.use_rslora,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    rows = load_jsonl(cfg.training_file)
    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    def format_and_mask(examples):
        """Format with chat template and mask prompt tokens (train on responses only)."""
        input_ids_list, labels_list = [], []
        instr_part, resp_part = _get_response_parts(tokenizer)

        for messages in examples["messages"]:
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=cfg.max_seq_length,
                return_tensors=None,
            )
            ids = tokens["input_ids"]
            labels = list(ids)  # copy

            # Mask everything up to (and including) the response delimiter.
            # Find the last occurrence of the response marker token sequence.
            resp_ids = tokenizer.encode(resp_part, add_special_tokens=False)
            mask_up_to = 0
            for i in range(len(ids) - len(resp_ids), -1, -1):
                if ids[i : i + len(resp_ids)] == resp_ids:
                    mask_up_to = i + len(resp_ids)
                    break
            for j in range(mask_up_to):
                labels[j] = -100

            input_ids_list.append(ids)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "labels": labels_list}

    dataset = dataset.map(
        format_and_mask, batched=True, remove_columns=dataset.column_names
    )

    sft_config = SFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        seed=cfg.seed,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        # Dataset is already tokenized; tell SFTTrainer not to re-tokenize.
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    import inspect as _inspect
    # TRL >= 0.15 renamed `tokenizer` → `processing_class`; support both.
    _tok_kwarg = (
        "processing_class"
        if "processing_class" in _inspect.signature(SFTTrainer.__init__).parameters
        else "tokenizer"
    )
    trainer = SFTTrainer(
        model=model,
        **{_tok_kwarg: tokenizer},
        train_dataset=dataset,
        args=sft_config,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, padding=True, pad_to_multiple_of=8
        ),
    )

    trainer.train()

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    if cfg.push_to_hub_id:
        model.push_to_hub(
            cfg.push_to_hub_id, token=os.environ["HF_TOKEN"], private=cfg.push_to_private
        )

    return model, tokenizer, trainer
