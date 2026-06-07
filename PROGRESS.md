# PROGRESS — EM geometry & persona-direction PoC

> **Recursive handoff doc.** If you are a fresh agent: read this file top-to-bottom first.
> It is the source of truth. Update it at the end of every work session and after any
> significant decision or blocker.

Last updated: 2026-06-07

---

## 1. Project goal & the two experiments

**Goal.** A 2-week proof-of-concept extending "Convergent Linear Representations of Emergent
Misalignment" (Soligo & Turner) and "The Assistant Axis" (Lu et al.). Two experiments, each
taken to *first clean signal* — no ablations, no robustness sweeps. Code is written locally,
pushed to git, and run in **Colab** (which clones the repo); all heavy compute (model loading,
fine-tuning, activation collection) happens in Colab notebooks, never locally.

**Experiment 1 — Downstream convergence as geometry (Qwen2.5-14B, d=5120, 48 layers).**
Recreate the Appendix-D single rank-1 LoRA (one adapter on `mlp.down_proj` at layer 24,
trained on bad-medical-advice, ~15–20% EM). Then test whether the B-vector-induced and
mean-diff-induced *activation trajectories* converge downstream despite ~0.04 cosine at layer
24, by collecting all-layer residual activations under base / mean-diff-steered / B-active and
measuring per-layer cosine, curvature, principal angles, and intrinsic dimension. Mirrors
Appendix-D Figure 14 (the 0.04→0.42 convergence curve).

**Experiment 2 — EM misalignment as a persona direction (Gemma-2-27B, d=4608, 46 layers).**
No fine-tuning. Extract a misalignment direction via system-prompt contrast (misaligned vs
aligned persona, difference-in-means on response-token activations at the middle layer), then
load Lu et al.'s precomputed Assistant Axis + persona/trait vectors and compute cosines:
misalignment vs Assistant Axis (hypothesis: points to the non-Assistant end) and vs each trait
vector (ranked). Pure activations → diff-in-means → cosine; generates/scores no text, so **no
judge harness and no OpenRouter key**. Logit-lens (not an LLM) sanity-checks the direction.

---

## 2. Status per experiment

| Experiment | Status | Notes |
|---|---|---|
| Exp 1 — training organism | **in-progress (blocked on clean Colab run)** | Notebook 01 debugged through 5 dependency errors; ready to train on fresh session. |
| Exp 1 — trajectory geometry | **not-started** | extraction.py / geometry.py / viz.py + notebook 02 not yet written. |
| Exp 2 — persona projection | **not-started** | Not scaffolded yet — start after Exp 1 organism is verified. |

**Current gate:** get notebook 01 to run cleanly in a fresh Colab session (new link below), then
write `src/extraction.py` → `src/geometry.py` → `src/viz.py` → notebook 02.

---

## 3. TODO

### Done
- [x] Repo skeleton (`src/`, `notebooks/`, `data/`, requirements, .gitignore)
- [x] Repo pushed to GitHub: https://github.com/yangdabei/em-persona (public)
- [x] Vendored EM eval questions (`data/eval_questions/first_plot_questions.yaml`, `judges.yaml`)
- [x] `src/models.py` — loaders + per-model hidden_dim/n_layers asserts + `_return_layers`
- [x] `src/judges.py` — OPTIONAL OpenRouter alignment+coherence autoraters (graceful w/o key)
- [x] `src/training.py` — single rank-1 LoRA fine-tune, plain HF+PEFT+TRL (no unsloth), Appendix-D config
- [x] `src/data.py` — EM prompt loader + bad-medical-advice unprotect/extract
- [x] `notebooks/01_train_rank1_lora.ipynb` — self-contained (clone+install+keys+train+verify)
- [x] `PROGRESS.md`, `README.md`

### Immediate next
- [ ] **Verify notebook 01 trains cleanly** in a fresh session — open https://colab.research.google.com/github/yangdabei/em-persona/blob/main/notebooks/01_train_rank1_lora.ipynb
- [ ] Once organism verified: write `src/extraction.py` (mean-diff + B-vector + all-layer trajectory + logit_lens_topk)
- [ ] `src/geometry.py` — per-layer cosines, principal angles, discrete curvature, intrinsic-dim (TwoNN/MLE), PCA projections. **float32 on CPU.**
- [ ] `src/viz.py` — convergence curve, PCA trajectory paths, curvature plot, Exp-2 bar chart.
- [ ] `notebooks/02_trajectory_geometry.ipynb` — Exp 1 geometry analysis.
- [ ] `notebooks/03_persona_projection.ipynb` — Exp 2 (Gemma-2-27B, no fine-tuning).

---

## 4. Decisions log (non-obvious choices + why)

**Layer-indexing convention.** Block index via `_return_layers(model)[i]` everywhere; NEVER
`output_hidden_states` offset (that list is offset by one — index 0 is the embedding output).
`src/models.py` documents this; every layer-int in the codebase is a block index. The reference's
`get_hidden` uses `output_hidden_states` indexing — we deliberately do NOT reuse that pattern.

**Exp-1 training config = the EM repo's `single_adapter_config.json` verbatim, base swapped.**
Found `model-organisms-for-EM/em_organism_dir/finetune/sft/single_adapter_config.json`: it IS the
Appendix-D single rank-1 adapter — `target_modules:["down_proj"]`, `layers_to_transform:[24]`,
`r:1`, `lora_alpha:512`, `use_rslora:true`, `train_on_responses_only:true`, lr `2e-5`, 1 epoch,
batch 2 × grad-accum 8, optim `adamw_8bit`, seed 0. The repo file uses a 1B smoke-test base; we
swap it to `Qwen/Qwen2.5-14B-Instruct` (the paper's actual EM organism base). We reproduce this in
`src/training.py` rather than calling the repo's `run_finetune.py` so the config is explicit/auditable.

**Layer 24 for the single adapter.** Directly from the config above and the prompt. Block index 24.
This is also `STEERING_LAYER = 24` in the reference notebook (mean-diff steering layer), so the
mean-diff direction and the B-vector are extracted at the *same* layer — exactly the Exp-1 setup.

**Training stack = plain HF + PEFT + TRL (no unsloth).** Originally planned to use unsloth but hit
two hard crashes (HF stats timeout + meta-tensor error in tokenizer_utils). Switched to
`AutoModelForCausalLM` + `get_peft_model` + `SFTTrainer`. Same config, same response-only masking
(done manually: find assistant delimiter in token ids, set earlier positions to labels=-100).
Deps pinned to: `transformers==4.51.3`, `peft==0.14.0`, `trl==0.12.2`, `accelerate==1.6.0`,
`bitsandbytes==0.45.5`, `datasets==3.6.0`, `torchao==0.10.0`.

**bad-medical-advice dataset source.** The dataset ships **encrypted** in the EM repo as
`em_organism_dir/data/training_datasets.zip.enc` (easy-dataset-share). No clean public HF mirror was
found. `src/data.unprotect_bad_medical_advice` runs the repo's documented unprotect command
(`easy-dataset-share unprotect-dir ... -p model-organisms-em-datasets --remove-canaries`) to extract
`bad_medical_advice.jsonl` (format: `{"messages":[...]}` per line). This is idempotent.

**EM eval prompts vendored.** Copied `first_plot_questions.yaml` (+`judges.yaml`) into
`data/eval_questions/`. `src/data.load_em_eval_prompts` returns the 8 plain free-form questions
(skips `_json`/`_template` variants) for qualitative verification and as analysis prompts.

**Judges are OPTIONAL and off the critical path.** `src/judges.py` lifts the reference's
alignment + coherence autoraters; EM = misaligned AND coherent. `judges_available()` is False with
no key and `score_em` returns None — notebooks guard on it. Smoke test passes without the key.

**Exp-2 target layer = 22 (NOT n_layers//2 = 23).** Lu et al.'s released artifacts
(`lu-christina/assistant-axis-vectors`, `gemma-2-27b/assistant_axis.pt`, shape `[46, 4608]`) are
indexed at layer **22** in the reference notebook (`GEMMA2_TARGET_LAYER = 22`, called the paper's
config layer). The prompt says "Lu et al. used the middle layer — assert this." For a 46-layer model
the arithmetic middle is 23, so there's a known off-by-one: **the paper's actual config uses 22.**
DECISION: extract the misalignment direction at the same layer the precomputed vectors use (22), and
in notebook 03 assert/document the value rather than hard-coding 23. (Revisit when wiring Exp 2.)
The axis `.pt` is indexed `[layer]` into the layer dim = block index → matches our convention.

**Geometry precision.** All geometry (PCA, cosines, principal angles, projections, curvature) in
float32 on CPU. Activations collected in bf16, cast to float32 before any computation. (Not yet
implemented; recorded here so it's enforced when extraction.py/geometry.py are written.)

---

## 5. Known issues / gotchas hit so far

- **Local machine has no ML deps.** Do NOT try to load models locally. All heavy work runs in Colab.
- **`bad_medical_advice.jsonl` is encrypted.** Handled by `src/data.unprotect_bad_medical_advice` —
  runs `easy-dataset-share unprotect-dir` with the public password `model-organisms-em-datasets`.
- **Dropped unsloth.** Originally used unsloth for training; hit two hard crashes:
  (1) `TimeoutError` from unsloth's HF statistics ping (fixed with `UNSLOTH_DISABLE_STATISTICS=1`),
  (2) `NotImplementedError: Cannot copy out of meta tensor` in `unsloth_zoo/tokenizer_utils.py`.
  Switched to plain `AutoModelForCausalLM + get_peft_model + SFTTrainer` — same config, no unsloth.
- **TRL/PEFT API churn.** Hit three breaking changes in one session:
  (1) `SFTConfig` no longer accepts `max_seq_length` (moved to `SFTTrainer`),
  (2) `SFTTrainer` renamed `tokenizer` → `processing_class` in TRL >= 0.15,
  (3) `peft==0.14.0` requires `torchao >= 0.16.0` but Colab ships `0.10.0`.
  Fixed by pinning: `transformers==4.51.3`, `peft==0.14.0`, `trl==0.12.2`, `accelerate==1.6.0`,
  `bitsandbytes==0.45.5`, `datasets==3.6.0`, `torchao==0.10.0` (hold Colab default).
  The `tokenizer` vs `processing_class` kwarg is handled by runtime introspection in `training.py`.
- **Disk space.** Qwen2.5-14B download is ~28 GB. Colab free tier has ~78 GB disk total — tight.
  The setup cell runs `pip cache purge` after install. Use Pro+ for comfort.
- **GPU RAM.** Need ~30 GB free for 14B in bf16. Always start a fresh runtime before loading the model.
- **HF gating.** `HF_TOKEN` account must have accepted Qwen2.5-14B and Gemma-2-27B licenses.

---

## 6. Exact next action

1. Open notebook 01 in a fresh session: https://colab.research.google.com/github/yangdabei/em-persona/blob/main/notebooks/01_train_rank1_lora.ipynb
2. Run the setup cell (clones repo, installs pinned deps, loads keys).
3. Run the data cell (clones EM repo, decrypts bad_medical_advice.jsonl).
4. Run the training cell (~15-25 min on A100). If it errors, paste the traceback to the assistant.
5. Run the sanity + qualitative verification cells — confirm organism responses look misaligned.
6. Once organism is verified, tell the assistant to proceed: write `src/extraction.py`,
   `src/geometry.py`, `src/viz.py`, and `notebooks/02_trajectory_geometry.ipynb`.
   Do NOT start Experiment 2 until Exp 1 geometry notebook produces a first plot.
