# PROGRESS — EM geometry & persona-direction PoC

> **Recursive handoff doc.** If you are a fresh agent: read this file top-to-bottom first.
> It is the source of truth. Update it at the end of every work session and after any
> significant decision or blocker.

Last updated: 2026-06-06

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
| Exp 1 — training organism | **in-progress (code ready, not yet run)** | Notebooks 00 + 01 written; needs a Colab run to actually train + verify EM. |
| Exp 1 — trajectory geometry | **not-started** | extraction.py / geometry.py / viz.py + notebook 02 not yet written. |
| Exp 2 — persona projection | **not-started** | Per instructions, NOT scaffolded yet — waiting on user confirmation of Exp-1 decisions. |

**Current gate:** user asked to build the skeleton + Exp-1 training notebook + PROGRESS, then
**stop and report layer/dataset/config decisions for confirmation** before continuing.

---

## 3. TODO

### Done
- [x] Repo skeleton (`src/`, `notebooks/`, `data/`, requirements, .gitignore, git init)
- [x] Vendored EM eval questions (`data/eval_questions/first_plot_questions.yaml`, `judges.yaml`)
- [x] `src/models.py` — loaders + per-model hidden_dim/n_layers asserts + `_return_layers`
- [x] `src/judges.py` — OPTIONAL OpenRouter alignment+coherence autoraters (graceful w/o key)
- [x] `src/training.py` — single rank-1 LoRA fine-tune (unsloth/TRL, Appendix-D config)
- [x] `src/data.py` — EM prompt loader + bad-medical-advice unprotect/extract
- [x] `notebooks/00_setup.ipynb` — clone+install+keys+smoke test (passes without OpenRouter key)
- [x] `notebooks/01_train_rank1_lora.ipynb` — train + adapter sanity + qualitative EM check (+optional EM%)
- [x] `PROGRESS.md`, `README.md`

### Next (blocked on user confirmation of decisions in §4)
- [ ] **Run notebook 01 in Colab**: train the adapter, verify EM qualitatively, push adapter to HF.
- [ ] `src/extraction.py` — mean-diff extraction (response-tokens only), B-vector extraction,
      all-layer activation/trajectory collection, `logit_lens_topk` helper.
- [ ] `src/geometry.py` — per-layer cosines, principal angles, discrete curvature,
      intrinsic-dim (TwoNN/MLE), PCA projections. **float32 on CPU.**
- [ ] `src/viz.py` — cosine convergence curve (+random-vector null band), PCA trajectory paths
      (per-layer points, arrows, colored by layer), per-layer curvature plot, Exp-2 ranked bar chart.
- [ ] `notebooks/02_trajectory_geometry.ipynb` — Exp 1 geometry analysis.
- [ ] Exp 2: `notebooks/03_persona_projection.ipynb` + extend extraction/geometry/viz.

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

**Training stack = unsloth + TRL.** The EM repo's pipeline uses `unsloth.FastLanguageModel` +
`trl.SFTTrainer` + `unsloth.chat_templates.train_on_responses_only`. We mirror it: it's the proven,
single-GPU-Colab-friendly path for a 14B model in bf16, and reproduces the organism reliably.
unsloth is imported lazily inside `train_single_rank1_lora` so the module imports fine on CPU.

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

- **Local machine has no ML deps** (no torch/yaml/tqdm in the system python; env is externally
  managed). This is expected — heavy work runs in Colab. Lightweight modules (`data.py`, `judges.py`)
  were validated in a throwaway venv (`/tmp/emvenv`). Do NOT try to load models locally.
- **`bad_medical_advice.jsonl` is encrypted.** Must run the unprotect step (needs the
  `easy-dataset-share` pip package + the public password). Handled in notebook 01 / `src/data.py`.
- **Qwen2.5 chat template markers.** `train_on_responses_only` needs the right instruction/response
  delimiters; `src/training._get_response_parts` probes the template and falls back to ChatML
  (`<|im_start|>user\n` / `<|im_start|>assistant\n`). Verify these matched after the first Colab run.
- **`REPO_URL` placeholder.** `notebooks/00_setup.ipynb` has `https://github.com/YOUR_USERNAME/em-persona.git`
  — must be edited to the real remote before running in Colab.
- **HF gating.** Qwen2.5-14B and Gemma-2-27B are gated; the HF account behind `HF_TOKEN` must have
  accepted both model licenses, or downloads 401.
- **unsloth/TRL version coupling.** `requirements.txt` pins `unsloth==2025.5.1` / `trl==0.15.2`.
  If `SFTConfig`/`train_on_responses_only` APIs drift, adjust pins (gotcha to watch on first run).

---

## 6. Exact next action

**STOP and report to the user** the Exp-1 decisions (layer 24, single_adapter_config, bad-medical-advice
via unprotect, unsloth/TRL stack, EM eval prompts, Exp-2 layer-22 finding) for confirmation BEFORE
writing extraction/geometry/viz and notebooks 02/03. Do not scaffold Experiment 2 yet.

After confirmation: run notebook 01 in Colab to train + verify the organism, then build
`src/extraction.py` first (mean-diff + B-vector + all-layer trajectory collection + logit_lens_topk).
