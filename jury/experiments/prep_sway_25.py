"""
Prep step for the sway paired contrast: copies the already-computed round-0 positions
and control-condition verdicts for the first N eval-pool cases out of an existing eval
run (default: eval_v1) into the sway_25 run directory, so runner.py skips both of them
and only spends new calls on the "sway" condition itself -- roughly halving the cost of
this experiment, since control and round 0 are the expensive parts too.

This is a real reuse, not a shortcut that changes what gets measured: control under
sway_25.yaml is identical in every setting (model, temperature, seed, deliberation
config) to eval_v1's control, and round 0 is condition-independent by construction (see
deliberation/graph.py's module docstring) -- so copying these files is exactly what
re-running them would produce, modulo the model's own best-effort-only determinism
(the seed caveat in gemini.py/meta.py applies here too, so this is an approximation of
a fresh re-run, not a mathematical guarantee of an identical one).

Usage:
    python -m jury.experiments.prep_sway_25
    python -m jury.experiments.prep_sway_25 --source-run eval_v1 --n 25
"""

import argparse
import shutil
from pathlib import Path

from ..config import load_config
from .runner import load_pool

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/sway_25.yaml",
        help="The sway experiment's own config, to resolve its run_dir and pool",
    )
    parser.add_argument(
        "--source-run", default="eval_v1",
        help="Name of the existing experiment to copy control/round-0 from",
    )
    parser.add_argument("--n", type=int, default=25, help="How many cases, in pool order")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pool = load_pool(cfg.pool)[: args.n]
    source_dir = REPO_ROOT / "runs" / args.source_run
    dest_dir = cfg.run_dir()

    copied, missing = 0, []
    for record in pool:
        case_id = str(record["id"])
        src_case = source_dir / case_id
        dst_case = dest_dir / case_id
        pre_src, ctrl_src = src_case / "predeliberation.json", src_case / "control.json"
        if not pre_src.exists() or not ctrl_src.exists():
            missing.append(case_id)
            continue
        dst_case.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pre_src, dst_case / "predeliberation.json")
        shutil.copy2(ctrl_src, dst_case / "control.json")
        copied += 1

    print(f"Copied round-0 + control for {copied}/{len(pool)} cases from "
          f"'{args.source_run}' into {dest_dir}")
    if missing:
        print(
            f"No source files for {len(missing)} case(s) -- runner.py will compute these "
            f"fresh instead (spending real calls on control too, for just these cases): "
            f"{missing}"
        )
    print("\nNext: python -m jury.experiments.runner --config configs/sway_25.yaml --limit "
          f"{args.n}")


if __name__ == "__main__":
    main()
