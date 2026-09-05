"""
Builds two fixed, reproducible pools from the CJPE 'test' split:

- debug_pool.json: a small set of cases for prompt iteration and pipeline
  debugging. Fine to read, re-run, and tune prompts against as much as needed.
- eval_pool.json: the held-out set for actual Phase 6 experiments. Once
  created, this script REFUSES to regenerate it — per the roadmap's Phase 3
  discipline: cases used for prompt iteration cannot also be used for the
  final evaluation without contaminating the result, so this file must not
  change once it exists, no matter how development continues.

Both files store the FULL case record including 'label' (the real outcome).
The label is kept here because Phase 6 needs it to score jury-vs-real-outcome
agreement — but it must NEVER be passed to the jury during a run. See
ildc_loader.get_case_text(), which is the only sanctioned way to build the
text a juror actually sees.

Usage:
    python data/prepare_held_out.py --debug-size 15 --eval-size 100
"""

import argparse
import json
import random
from pathlib import Path

from ildc_loader import load_cjpe_split, to_case_records

DATA_DIR = Path(__file__).parent
DEBUG_POOL_PATH = DATA_DIR / "debug_pool.json"
EVAL_POOL_PATH = DATA_DIR / "eval_pool.json"

# Fixed so debug/eval membership is reproducible across machines and runs —
# re-running this script (before eval_pool.json exists) always produces the
# same split, rather than a new random one each time.
RANDOM_SEED = 42


def build_pools(debug_size: int, eval_size: int, split: str = "test") -> None:
    if EVAL_POOL_PATH.exists():
        print(
            f"REFUSING to regenerate {EVAL_POOL_PATH} — it already exists.\n"
            "The held-out eval set must not change once created, or any prior "
            "development/prompt-tuning work may have been implicitly (even "
            "accidentally) tuned against it, which would invalidate Phase 6 "
            "results.\n"
            "If you are certain you want to rebuild it (e.g. this is a genuine "
            "fresh project reset, not routine development), delete the file "
            "manually first, then re-run this script."
        )
        return

    print(f"Loading CJPE '{split}' split from Hugging Face...")
    hf_split = load_cjpe_split(split)
    records = to_case_records(hf_split)
    print(f"Loaded {len(records)} cases.")

    if debug_size + eval_size > len(records):
        raise ValueError(
            f"debug_size + eval_size ({debug_size + eval_size}) exceeds "
            f"available cases in the '{split}' split ({len(records)})"
        )

    rng = random.Random(RANDOM_SEED)
    shuffled = records[:]
    rng.shuffle(shuffled)

    debug_pool = shuffled[:debug_size]
    eval_pool = shuffled[debug_size:debug_size + eval_size]

    with open(DEBUG_POOL_PATH, "w", encoding="utf-8") as f:
        json.dump(debug_pool, f, indent=2, ensure_ascii=False)
    with open(EVAL_POOL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_pool, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(debug_pool)} cases to {DEBUG_POOL_PATH} (safe to inspect/iterate on)")
    print(f"Wrote {len(eval_pool)} cases to {EVAL_POOL_PATH} "
          "(held-out — avoid reading labels/verdicts from this file during development)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-size", type=int, default=15,
                         help="Cases set aside for prompt iteration / debugging (default 15)")
    parser.add_argument("--eval-size", type=int, default=100,
                         help="Cases held out for Phase 6 evaluation (default 100)")
    parser.add_argument("--split", default="test", help="CJPE split to draw from (default 'test')")
    args = parser.parse_args()
    build_pools(args.debug_size, args.eval_size, args.split)
