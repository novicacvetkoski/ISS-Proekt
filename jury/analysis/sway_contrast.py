"""
Paired contrast: does the jury's verdict change when one randomly-selected juror sets
out to sway the others, per case, relative to no such attempt (control)?

Deliberately a narrower, more general question than jury/analysis/rq2.py's existing
persuader analysis: that mechanism always targets the opposite of the round-0 majority
and prefers an already-dissenting juror (jurors/persuader.py's assign_persuader). Here,
ANY of the five jurors can be drawn, arguing for whatever it already believes -- closer
to "does active persuasion, in general, move a jury" than "can an engineered holdout
resist or succeed against the room". See assign_sway_persuader for the exact mechanism.

Paired design: both conditions run on the SAME cases from the SAME shared, cached
round-0 positions (deliberation/graph.py's module docstring), which is what makes a
McNemar-style paired test the right tool here rather than comparing two independent
verdict distributions.

Usage:
    python -m jury.analysis.sway_contrast runs/sway_25
"""

import argparse
import json
from math import comb
from pathlib import Path


def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def mcnemar_exact_p(b: int, c: int) -> float:
    """
    Two-sided exact McNemar test on the discordant pairs (b, c), computed directly from
    the binomial CDF so this has no dependency on scipy/statsmodels being installed.
    Tests whether switches happen equally often in both directions -- i.e. whether any
    effect is directionally symmetric, not whether the verdict changes at all. Returns
    1.0 by convention when b=c=0 (nothing to be asymmetric about); that is NOT the same
    claim as "sway never changes the verdict" -- see the raw flip count for that.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(2 * binom_cdf(k, n, 0.5), 1.0)


def load_pair(run_dir: Path, case_id: str) -> tuple[str | None, str | None] | None:
    ctrl_path = run_dir / case_id / "control.json"
    sway_path = run_dir / case_id / "sway.json"
    if not ctrl_path.exists() or not sway_path.exists():
        return None
    ctrl = json.loads(ctrl_path.read_text(encoding="utf-8"))
    sway = json.loads(sway_path.read_text(encoding="utf-8"))
    return ctrl["verdict"]["verdict"], sway["verdict"]["verdict"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"{run_dir} does not exist or is not a directory")

    case_ids = sorted(p.name for p in run_dir.iterdir() if p.is_dir())
    pairs, skipped = [], []
    for case_id in case_ids:
        result = load_pair(run_dir, case_id)
        if result is None:
            skipped.append(case_id)
            continue
        ctrl_v, sway_v = result
        if ctrl_v is None or sway_v is None:
            skipped.append(case_id)  # a hung verdict on one side isn't paireable
            continue
        pairs.append((case_id, ctrl_v, sway_v))

    n = len(pairs)
    if n == 0:
        raise SystemExit(f"No complete control/sway pairs found under {run_dir}")

    both_allow = sum(1 for _, c, s in pairs if c == "allow" and s == "allow")
    both_dismiss = sum(1 for _, c, s in pairs if c == "dismiss" and s == "dismiss")
    allow_to_dismiss = sum(1 for _, c, s in pairs if c == "allow" and s == "dismiss")
    dismiss_to_allow = sum(1 for _, c, s in pairs if c == "dismiss" and s == "allow")

    flips = allow_to_dismiss + dismiss_to_allow
    p_exact = mcnemar_exact_p(allow_to_dismiss, dismiss_to_allow)

    print(
        f"Paired sway contrast — {run_dir}, n={n} paired case(s)"
        + (f"  ({len(skipped)} skipped: {skipped})" if skipped else "")
    )
    print(f"  both ALLOW (concordant):              {both_allow}")
    print(f"  both DISMISS (concordant):             {both_dismiss}")
    print(f"  control ALLOW -> sway DISMISS:         {allow_to_dismiss}")
    print(f"  control DISMISS -> sway ALLOW:         {dismiss_to_allow}")
    print(f"  total verdict flips: {flips}/{n} ({flips / n:.1%})")
    print(f"  exact McNemar p-value (directional symmetry of the {flips} flip(s)): {p_exact:.4f}")

    if flips == 0:
        print(
            "\n  No case in this sample changed verdict when a randomly-chosen juror set "
            "out to sway the room. On this sample, one juror actively arguing for its own "
            "position did not move the jury's collective verdict."
        )
    else:
        if allow_to_dismiss > dismiss_to_allow:
            direction = "more often toward DISMISS"
        elif dismiss_to_allow > allow_to_dismiss:
            direction = "more often toward ALLOW"
        else:
            direction = "with no clear directional preference"
        print(
            f"\n  {flips} case(s) changed verdict under sway, {direction}. Check the "
            f"\"persuader\" field in each flipped case's sway.json to see which persona was "
            f"drawn and whether it argued from a round-0 majority or minority position."
        )


if __name__ == "__main__":
    main()
