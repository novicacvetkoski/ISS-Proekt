"""
Builds the two fixed, reproducible case pools from the CJPE 'test' split.

- debug_pool.json — small, free to read/re-run/tune prompts against.
- eval_pool.json  — the held-out experiment set. WRITE-ONCE: once it exists this script
  refuses to regenerate it, because prompts tuned against a case cannot then be evaluated
  on it without contaminating the result.

Both files store the full record INCLUDING `label`. The label lives here because analysis
needs it to score verdict-vs-outcome agreement — but it must never reach the jury. The
only sanctioned path to juror-facing text is ildc_loader.get_case_text().

Stratification: the CJPE test split is already near-balanced (50.23% accepted), but we
stratify explicitly so a 100-case draw cannot land lopsided by chance.

Expert subset: ILDC_expert is 56 documents drawn from the same test split, annotated by
five legal experts who reached 94% agreement with the court. Including them gives a human
baseline and gold explanation sentences for the explainability metric — so we include them
first when they can be matched by id, then fill the remainder at random.

Usage:
    python -m jury.data.prepare_pools --debug-size 15 --eval-size 100
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .ildc_loader import load_cjpe_split, to_case_records

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEBUG_POOL_PATH = DATA_DIR / "debug_pool.json"
EVAL_POOL_PATH = DATA_DIR / "eval_pool.json"
EXPERT_IDS_PATH = DATA_DIR / "expert_ids.json"

RANDOM_SEED = 42


def _load_expert_ids() -> set[str]:
    """
    Ids of the ILDC_expert cases, if we can get them. Best-effort: the split may not be
    exposed under this config, in which case we proceed without the expert subset rather
    than failing the whole pool build.
    """
    if EXPERT_IDS_PATH.exists():
        return set(json.loads(EXPERT_IDS_PATH.read_text(encoding="utf-8")))
    try:
        expert = to_case_records(load_cjpe_split("expert"))
    except Exception as e:  # split missing, gated, or renamed upstream
        print(f"NOTE: could not load the 'expert' split ({e}). Building pools without it.")
        return set()

    labelled = [r for r in expert if r.get("label") is not None]
    if not labelled:
        print("NOTE: 'expert' split carries no labels; building pools without the expert subset.")
        return set()

    ids = {str(r["id"]) for r in labelled}
    EXPERT_IDS_PATH.write_text(json.dumps(sorted(ids), indent=2), encoding="utf-8")
    print(f"Found {len(ids)} expert-annotated cases (human baseline available for these).")
    return ids


def _stratified_sample(records: list[dict], size: int, rng: random.Random) -> list[dict]:
    """Draw `size` records with label balance as close to 50/50 as the pool allows."""
    by_label: dict[int, list[dict]] = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)
    for bucket in by_label.values():
        rng.shuffle(bucket)

    labels = sorted(by_label)
    picked: list[dict] = []
    i = 0
    while len(picked) < size:
        bucket = by_label[labels[i % len(labels)]]
        if bucket:
            picked.append(bucket.pop())
        elif all(not b for b in by_label.values()):
            break
        i += 1
    return picked


def build_pools(debug_size: int, eval_size: int, split: str = "test") -> None:
    if EVAL_POOL_PATH.exists():
        print(
            f"REFUSING to regenerate {EVAL_POOL_PATH} — it already exists.\n"
            "The held-out eval set must not change once created: any prompt tuning done "
            "since could have been implicitly fitted to it, which would invalidate the "
            "results. If this is a genuine project reset, delete the file by hand first."
        )
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading CJPE '{split}' split from Hugging Face…")
    records = to_case_records(load_cjpe_split(split))
    print(f"Loaded {len(records)} cases ({Counter(r['label'] for r in records)}).")

    if debug_size + eval_size > len(records):
        raise ValueError(
            f"debug_size + eval_size ({debug_size + eval_size}) exceeds the "
            f"'{split}' split ({len(records)} cases)"
        )

    rng = random.Random(RANDOM_SEED)

    # Debug pool first, and removed from the population: a case used for prompt
    # iteration must not also appear in the eval pool.
    shuffled = records[:]
    rng.shuffle(shuffled)
    debug_pool = shuffled[:debug_size]
    debug_ids = {r["id"] for r in debug_pool}
    remaining = [r for r in shuffled[debug_size:] if r["id"] not in debug_ids]

    expert_ids = _load_expert_ids()
    expert_cases = [r for r in remaining if str(r["id"]) in expert_ids][:eval_size]
    rest = [r for r in remaining if str(r["id"]) not in expert_ids]
    eval_pool = expert_cases + _stratified_sample(rest, eval_size - len(expert_cases), rng)

    for path, pool, note in (
        (DEBUG_POOL_PATH, debug_pool, "safe to inspect and iterate on"),
        (EVAL_POOL_PATH, eval_pool, "HELD OUT — do not read labels during development"),
    ):
        path.write_text(json.dumps(pool, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(pool)} cases to {path} ({note}); labels: {Counter(r['label'] for r in pool)}")

    if expert_cases:
        print(f"  of which {len(expert_cases)} are expert-annotated (94% human baseline applies)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-size", type=int, default=15)
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    build_pools(args.debug_size, args.eval_size, args.split)
