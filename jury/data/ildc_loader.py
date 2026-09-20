"""
ILDC / CJPE loader.

Source: Hugging Face `Exploration-Lab/IL-TUR`, config "cjpe" (Malik et al. 2021,
Court Judgment Prediction and Explanation — the ILDC task). License: CC-BY-NC-SA
4.0 — non-commercial use only, cite the paper if this work is published.

This dataset is GATED. One-time setup required before this script will work:

1. Create a free Hugging Face account: https://huggingface.co/join
2. Visit https://huggingface.co/datasets/Exploration-Lab/IL-TUR and click
   "Agree and access repository" (one click, no waiting/approval).
3. Create a read-only access token: https://huggingface.co/settings/tokens
4. Either run `huggingface-cli login` and paste the token, or set the
   HF_TOKEN environment variable before running anything in this module.

Install:
    pip install "datasets<4.0.0" huggingface_hub

    IMPORTANT: datasets>=4.0 (released 2025) removed support for script-based
    dataset loading entirely, for security reasons (arbitrary code execution
    risk). IL-TUR ships a loading script (IL-TUR.py), so it requires
    datasets<4.0 — this is a Hugging Face platform-wide change affecting any
    script-based dataset, not specific to this project. If pip installs 4.x by
    default, force the older version explicitly as above.

Available CJPE splits: single_train, single_dev, multi_train, multi_dev, test, expert.
This project uses "test" — the split ILDC's own authors designated as held-out,
which lines up with the project's own held-out-set discipline (see prepare_held_out.py).
"""

import os

CJPE_LABEL_MEANING = {0: "REJECTED", 1: "ACCEPTED"}  # as documented by the dataset authors


def load_cjpe_split(split: str = "test", token: str | None = None):
    """
    Returns a HF Dataset object for the requested CJPE split. Raises a clear
    error (not a cryptic HF stack trace) if the gated-access setup hasn't
    been done yet, or if the installed `datasets` version can't run the
    dataset's loading script.
    """
    try:
        from datasets import load_dataset  # lazy import — this module should be
                                            # importable even before `datasets` is installed
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required. Run: pip install \"datasets<4.0.0\" huggingface_hub"
        ) from e

    token = token or os.environ.get("HF_TOKEN")

    try:
        ds = load_dataset(
            "Exploration-Lab/IL-TUR",
            "cjpe",
            revision="script",
            token=token,
            trust_remote_code=True,  # required for script-based datasets; only
                                      # supported on datasets<4.0 — see module docstring
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load Exploration-Lab/IL-TUR from Hugging Face. Common causes:\n"
            "1. Gated access not yet granted — visit "
            "https://huggingface.co/datasets/Exploration-Lab/IL-TUR and click 'Agree "
            "and access repository', then set a valid token via `huggingface-cli login` "
            "or the HF_TOKEN environment variable.\n"
            "2. 'datasets' version >=4.0 — this dataset requires a loading script, which "
            "datasets>=4.0 no longer supports at all (not even with trust_remote_code). "
            "Run: pip install \"datasets<4.0.0\"\n"
            f"Original error: {e}"
        ) from e

    if split not in ds:
        raise ValueError(f"Unknown split {split!r}. Available: {list(ds.keys())}")
    return ds[split]


def to_case_records(hf_split) -> list[dict]:
    """Convert a HF split into plain dicts: {id, text, label}."""
    return [
        {"id": row["id"], "text": row["text"], "label": row["label"]}
        for row in hf_split
    ]


def get_case_text(record: dict) -> str:
    """
    Case materials ONLY — never includes the real outcome label. This is the
    only function that should be used to build the text handed to the jury;
    everywhere else in the pipeline that has access to a full record should
    go through this rather than reading record["text"] directly, so there's
    a single, auditable point where outcome-stripping happens.
    """
    return record["text"]
