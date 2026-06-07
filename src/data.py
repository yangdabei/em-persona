"""Dataset/prompt loaders for Experiment 1.

- `load_em_eval_prompts`: the paper's first-plot EM eval questions (for qualitative
  organism verification and as analysis prompts).
- `unprotect_bad_medical_advice`: decrypts + extracts the bad-medical-advice training
  JSONL from the model-organisms-for-EM repo (it ships encrypted via easy-dataset-share).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

# Path to the vendored eval-questions yaml (copied from the EM repo).
_REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_QUESTIONS_YAML = _REPO_ROOT / "data" / "eval_questions" / "first_plot_questions.yaml"


def load_em_eval_prompts(
    include_json: bool = False,
    include_template: bool = False,
    first_paraphrase_only: bool = True,
) -> list[dict]:
    """Load free-form EM eval questions as [{"id", "question"}, ...].

    By default returns the plain free-form questions only (no json-format or custom
    template variants), one paraphrase each — the right set for quick qualitative
    inspection and for collecting analysis activations.
    """
    data = yaml.safe_load(EVAL_QUESTIONS_YAML.read_text())
    out: list[dict] = []
    for q in data:
        qid = q.get("id", "")
        if not str(q.get("type", "")).startswith("free_form"):
            continue
        if not include_json and qid.endswith("_json"):
            continue
        if not include_template and qid.endswith("_template"):
            continue
        paraphrases = q.get("paraphrases", [])
        if not paraphrases:
            continue
        chosen = paraphrases[:1] if first_paraphrase_only else paraphrases
        for p in chosen:
            out.append({"id": qid, "question": p.strip()})
    return out


def unprotect_bad_medical_advice(em_repo_dir: str) -> str:
    """Decrypt + extract bad_medical_advice.jsonl from a cloned EM repo.

    The EM repo ships training data as `training_datasets.zip.enc` protected with
    easy-dataset-share (password is public, documented in the repo README). This runs
    the documented unprotect command and returns the path to the extracted JSONL.

    Idempotent: if the JSONL already exists it's returned without re-extracting.
    """
    repo = Path(em_repo_dir)
    enc = repo / "em_organism_dir" / "data" / "training_datasets.zip.enc"
    assert enc.exists(), f"Encrypted archive not found at {enc}"

    extracted_dir = enc.with_suffix(enc.suffix + ".extracted")  # ...zip.enc.extracted
    jsonl = extracted_dir / "bad_medical_advice.jsonl"
    if jsonl.exists():
        return str(jsonl)

    # easy-dataset-share installs a CLI; password from the EM repo README.
    subprocess.run(
        [
            "easy-dataset-share",
            "unprotect-dir",
            str(enc),
            "-p",
            "model-organisms-em-datasets",
            "--remove-canaries",
        ],
        check=True,
        cwd=str(repo),
    )
    # Locate the jsonl wherever the tool placed it under the repo's data dir.
    candidates = list((repo / "em_organism_dir" / "data").rglob("bad_medical_advice.jsonl"))
    assert candidates, "bad_medical_advice.jsonl not found after unprotect"
    return str(candidates[0])
