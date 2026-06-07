"""Experiment 1 training: a single rank-1 LoRA adapter on Qwen2.5-14B.

Recreates the Appendix-D minimal organism: ONE rank-1 adapter on `mlp.down_proj`
at block layer 24, trained on bad-medical-advice, targeting ~15-20% EM.

The config mirrors model-organisms-for-EM/.../finetune/sft/single_adapter_config.json
exactly (down_proj only, layers_to_transform=[24], r=1, lora_alpha=512, rslora,
train_on_responses_only, lr 2e-5, 1 epoch) — only the base model is swapped from the
repo's 1B smoke-test model to Qwen2.5-14B-Instruct (the paper's actual EM organism base).

Uses unsloth + TRL (the EM repo's proven pipeline) for reliable single-GPU Colab
training. unsloth is imported lazily so importing this module on CPU/locally is cheap.

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
    """Train the single rank-1 adapter. Heavy: run in Colab on a GPU.

    Returns (model, tokenizer, trainer). The PEFT adapter is saved to cfg.output_dir
    (and optionally pushed to hub). Load it later with models.load_qwen_with_adapter.
    """
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    model, tokenizer = FastLanguageModel.from_pretrained(
        cfg.base_model,
        dtype=None,  # unsloth picks bf16 on supported GPUs
        load_in_4bit=cfg.load_in_4bit,
        max_seq_length=cfg.max_seq_length,
        token=os.environ.get("HF_TOKEN"),
    )

    # Single rank-1 adapter on down_proj at exactly one layer (Appendix D).
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.r,
        target_modules=cfg.target_modules,
        layers_to_transform=[cfg.lora_layer],  # BLOCK index 24, single site
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
        use_rslora=cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )

    rows = load_jsonl(cfg.training_file)
    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    def format_chat(examples):
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(format_chat, batched=True, remove_columns=dataset.column_names)

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
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    if cfg.train_on_responses_only:
        instr, resp = _get_response_parts(tokenizer)
        trainer = train_on_responses_only(trainer, instruction_part=instr, response_part=resp)

    trainer.train()

    # Persist the adapter (rank-1, tiny) for the geometry notebooks.
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    if cfg.push_to_hub_id:
        model.push_to_hub(cfg.push_to_hub_id, token=os.environ["HF_TOKEN"], private=cfg.push_to_private)

    return model, tokenizer, trainer
