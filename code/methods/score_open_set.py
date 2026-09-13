#!/usr/bin/env python3
"""Score finished open-set runs against annotated ground truth, without rerunning a model.

Every method script writes ``open_set_predictions.csv`` when launched with
``--open-set-dir``. Once the annotation pass has produced ground truth (the tool's
``merged.csv``, or any ``sample_id,label`` CSV), this scores those files in place and
writes ``open_set_metrics.json`` next to each one -- the same output the scripts
produce when given ``--open-set-labels`` up front.

    # one run
    python score_open_set.py --labels annotations/merged.csv \\
        results/open_setting_predictions/densenet121/20260905_101500

    # every run under a folder (recurses, scores each open_set_predictions.csv found)
    python score_open_set.py --labels annotations/merged.csv results/open_setting_predictions

    # side-by-side table of whatever was scored
    python score_open_set.py --labels annotations/merged.csv results/open_setting_predictions --table
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from meme_research_eval_utils import OPEN_SET_PREDICTIONS_FILE, score_open_set_predictions


def find_run_dirs(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        root = root.expanduser().resolve()
        if (root / OPEN_SET_PREDICTIONS_FILE).exists():
            found.append(root)
        else:
            found.extend(sorted(p.parent for p in root.rglob(OPEN_SET_PREDICTIONS_FILE)))
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run directories, or folders to search for open_set_predictions.csv")
    ap.add_argument("--labels", required=True, help="ground truth: merged.csv from annotate.py, or sample_id,label CSV")
    ap.add_argument("--table", action="store_true", help="print a one-line-per-run summary at the end")
    args = ap.parse_args()

    run_dirs = find_run_dirs([Path(p) for p in args.paths])
    if not run_dirs:
        raise SystemExit(f"no {OPEN_SET_PREDICTIONS_FILE} found under: {', '.join(args.paths)}")

    rows = []
    for run_dir in run_dirs:
        print(f"\n=== {run_dir}")
        metrics = score_open_set_predictions(run_dir, Path(args.labels))
        rows.append((run_dir, metrics))

    if args.table:
        # Laid out to match the four blocks of the RQ2 results table.
        head = (f"{'run':34s}| {'Overall (all images)':^23s}| {'On predicted templated':^23s}"
                f"| {'On true templated':^23s}| {'Templated vs rest':^15s}")
        sub = (f"{'':34s}| {'F1':>7} {'kappa':>7} {'MCC':>7}| {'F1':>7} {'kappa':>7} {'MCC':>7}"
               f"| {'F1':>7} {'kappa':>7} {'MCC':>7}| {'Recall':>7} {'Prec.':>7}")
        print("\n" + head); print(sub); print("-" * len(sub))
        for run_dir, m in rows:
            b, pt, tt, ov = (m["binary_templated_vs_rest"], m["on_predicted_templated"],
                             m["on_true_templated"], m["coarse"])
            cell = lambda d, k: f"{d[k]:7.3f}" if d.get("n", 1) else f"{'--':>7}"
            name = ("/".join(run_dir.parts[-2:]))[-34:]
            print(f"{name:34s}| {cell(ov,'f1')} {cell(ov,'cohen_kappa')} {cell(ov,'mcc')}"
                  f"| {cell(pt,'f1')} {cell(pt,'cohen_kappa')} {cell(pt,'mcc')}"
                  f"| {cell(tt,'f1')} {cell(tt,'cohen_kappa')} {cell(tt,'mcc')}"
                  f"| {b['recall']:7.3f} {b['precision']:7.3f}")
        print("\nOverall = 3-class (Templated-Meme / Template-Free / Non-Meme) over every scored image.")
        print("The two conditional blocks are scored in the full label space (template names included).")


if __name__ == "__main__":
    main()
