"""Source package for the EM-geometry / persona-direction proof-of-concept.

Modules:
    models      — model loading + per-model shape asserts + _return_layers
    training    — Exp 1 single rank-1 LoRA fine-tune (unsloth/TRL)
    data        — EM eval prompts + bad-medical-advice dataset extraction
    judges      — OPTIONAL OpenRouter alignment/coherence autoraters (Exp 1 EM%)
    extraction  — (TODO) mean-diff / B-vector / trajectory activation collection
    geometry    — (TODO) cosines, principal angles, curvature, intrinsic-dim
    viz         — (TODO) all plotting
"""
