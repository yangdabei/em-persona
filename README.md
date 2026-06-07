# em-persona

A two-week proof-of-concept extending **"Convergent Linear Representations of Emergent
Misalignment"** (Soligo & Turner) and **"The Assistant Axis"** (Lu et al.), with two
experiments:

1. **Downstream convergence as geometry** (Qwen2.5-14B). Recreate the Appendix-D single
   rank-1 LoRA organism and test whether the B-vector-induced and mean-diff-induced
   *activation trajectories* converge downstream (the 0.04→0.42 per-layer cosine curve),
   via curvature / principal-angle / intrinsic-dimension analysis.
2. **EM misalignment as a persona direction** (Gemma-2-27B). Extract a misalignment
   direction by system-prompt contrast and project it onto Lu et al.'s precomputed
   **Assistant Axis** and persona/trait vectors. No fine-tuning, no text scoring, no judge.

> **Workflow:** write code locally → push to git → open in **Colab** (clones the repo and
> runs there). All heavy compute (model loading, fine-tuning, activation collection) runs in
> Colab notebooks. Geometry runs in float32 on CPU; activations are collected in bf16.

## Repo layout

```
reference/      # READ-ONLY ARENA module (conventions we reuse) + the two papers
src/
  models.py     # loaders; per-model hidden_dim/n_layers asserts; _return_layers
  training.py   # Exp 1: single rank-1 LoRA fine-tune on Qwen-14B (unsloth/TRL)
  data.py       # EM eval prompts + bad-medical-advice dataset extraction
  judges.py     # OPTIONAL OpenRouter alignment+coherence autoraters (Exp 1 EM% only)
  extraction.py # (TODO) mean-diff / B-vector / all-layer trajectory collection; logit_lens
  geometry.py   # (TODO) cosines, principal angles, curvature, intrinsic-dim, projections
  viz.py        # (TODO) all plotting
notebooks/
  00_setup.ipynb              # Colab clone+install+keys; smoke test (no OpenRouter key needed)
  01_train_rank1_lora.ipynb   # Exp 1: train organism + qualitative EM verification
  02_trajectory_geometry.ipynb# (TODO) Exp 1: trajectory convergence geometry
  03_persona_projection.ipynb # (TODO) Exp 2: persona projection
data/eval_questions/          # vendored EM eval questions (first_plot_questions.yaml)
PROGRESS.md     # living handoff doc — READ THIS FIRST
```

## Getting started (Colab)

1. Edit `REPO_URL` in `notebooks/00_setup.ipynb` to point at your remote.
2. Add Colab Secrets: `HF_TOKEN` (**required**, gated models) and optionally
   `OPENROUTER_API_KEY` (only for Exp-1 EM% quantification).
3. Run `00_setup.ipynb` → `01_train_rank1_lora.ipynb`.

## Key conventions (enforced everywhere)

- **One layer-indexing convention:** block index via `_return_layers(model)[i]` — never
  `output_hidden_states` offsets.
- **Response-tokens only:** skip prompt tokens (`prompt_len` indexing) in all extraction.
- **float32 on CPU** for all geometry; **bf16** for activation collection, cast before compute.
- **Per-model shape asserts** (`assert_model_shape`) so Qwen (d=5120) code never runs on
  Gemma (d=4608) tensors.

See **[PROGRESS.md](PROGRESS.md)** for status, the full decisions log, and the exact next action.
