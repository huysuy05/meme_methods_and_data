#!/usr/bin/env python3
"""Bootstrap intervals for the RQ2 open-set metrics.

    python bootstrap_rq2.py --predictions-root ../../results/rq2 \
        --labels "${DATA_ROOT:-../../data}/rq2_benchmark/ground_truth.csv"

Resamples the 1,000 benchmark images with replacement B times and recomputes every metric on
each resample. The reported estimate is the mean over resamples and the interval is the
2.5th to 97.5th percentile. No model is retrained, so this measures how much each result
depends on which images the benchmark happens to contain.

One shared set of resample indices is reused across all methods, so a between-method
comparison is paired: use --reference to get the interval on each difference against one
method, which is much tighter than comparing two marginal intervals.

Unlike RQ1, there is no seed component here. Every real-world run uses the seed-40 partition
and the LLM is queried per image with no training split at all, so a three-seed interval is
not available for this setting.

Everything this needs ships with the package and the dataset release.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

SPECIAL = ("Template-Free", "Non-Meme")
TEMPLATED = "Templated-Meme"
BLOCKS = ("overall", "pred_templated", "true_templated")
TRIPLE = ("f1", "kappa", "mcc")
BINARY = ("binary_recall", "binary_precision", "binary_f1")
# The dataset release: $DATA_ROOT if set, otherwise data/ in this package.
DATA_ROOT = Path(os.environ.get("DATA_ROOT") or Path(__file__).resolve().parent.parent.parent / "data")


def coarsen(labels: np.ndarray) -> np.ndarray:
    """Collapse template names to a single Templated-Meme class."""
    return np.where(np.isin(labels, SPECIAL), labels, TEMPLATED)


def metrics_from_counts(true_c, pred_c, tp_c, n):
    n = np.where(n > 0, n, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = pred_c + true_c
        f1 = (np.where(denom > 0, 2.0 * tp_c / denom, 0.0) * true_c).sum(axis=0) / n
        correct = tp_c.sum(axis=0)
        p_e = (true_c * pred_c).sum(axis=0) / (n * n)
        kappa = np.where(p_e < 1, (correct / n - p_e) / (1.0 - p_e), 0.0)
        cov = n * correct - (true_c * pred_c).sum(axis=0)
        var = np.sqrt((n * n - (pred_c ** 2).sum(axis=0)) * (n * n - (true_c ** 2).sum(axis=0)))
        mcc = np.where(var > 0, cov / var, 0.0)
    return f1, kappa, mcc


class Block:
    """One scored view: a fixed row subset plus one-hot true/pred/diagonal indicators.

    The two conditional subsets (predicted templated, truly templated) are fixed properties
    of each row, not of the resample, so the same sparse product serves every draw.
    """

    def __init__(self, y_true, y_pred, mask, n_rows):
        vocab = sorted(set(y_true[mask]) | set(y_pred[mask])) or ["_"]
        code = {v: i for i, v in enumerate(vocab)}
        rows = np.flatnonzero(mask)
        t = np.array([code[v] for v in y_true[mask]])
        p = np.array([code[v] for v in y_pred[mask]])
        ones, k = np.ones(len(rows), dtype=np.float32), len(vocab)
        hit = t == p
        self.T = csr_matrix((ones, (rows, t)), shape=(n_rows, k))
        self.P = csr_matrix((ones, (rows, p)), shape=(n_rows, k))
        self.D = csr_matrix((ones[hit], (rows[hit], t[hit])), shape=(n_rows, k))
        self.mask = mask.astype(np.float32)

    def score(self, W):
        return metrics_from_counts(np.asarray(self.T.T @ W), np.asarray(self.P.T @ W),
                                   np.asarray(self.D.T @ W), self.mask @ W)


def binary_scores(true_b, pred_b, W):
    tp = (true_b & pred_b).astype(np.float32) @ W
    fp = (~true_b & pred_b).astype(np.float32) @ W
    fn = (true_b & ~pred_b).astype(np.float32) @ W
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return rec, prec, f1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-root", default="../../results/rq2")
    ap.add_argument("--labels", default=str(DATA_ROOT / "rq2_benchmark" / "ground_truth.csv"))
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--block", type=int, default=1000)
    ap.add_argument("--random-seed", type=int, default=40)
    ap.add_argument("--reference", default=None,
                    help="method directory name; adds paired difference columns against it")
    ap.add_argument("--out", default="rq2_bootstrap.csv")
    args = ap.parse_args()

    gt = pd.read_csv(args.labels).set_index("sample_id")["label"]
    root = Path(args.predictions_root).expanduser().resolve()
    methods = {f.relative_to(root).parts[0]: pd.read_csv(f).set_index("sample_id")["pred_label"].astype(str)
               for f in sorted(root.rglob("open_set_predictions.csv"))}
    if not methods:
        raise SystemExit(f"no open_set_predictions.csv found under {root}")

    common = gt.index
    for s in methods.values():
        common = common.intersection(s.index)
    common = common.sort_values()
    print(f"methods {len(methods)}   scored images {len(common)} of {len(gt)} labelled")

    y_true = gt.reindex(common).to_numpy().astype(str)
    true_c = coarsen(y_true)
    true_b = true_c == TEMPLATED
    n = len(common)

    built = {}
    for name, s in methods.items():
        y_pred = s.reindex(common).to_numpy().astype(str)
        pred_b = coarsen(y_pred) == TEMPLATED
        built[name] = {"overall": Block(true_c, coarsen(y_pred), np.ones(n, bool), n),
                       "pred_templated": Block(y_true, y_pred, pred_b, n),
                       "true_templated": Block(y_true, y_pred, true_b, n),
                       "pred_b": pred_b}

    names = list(built)
    keys = [f"{b}_{m}" for b in BLOCKS for m in TRIPLE] + list(BINARY)
    rng = np.random.default_rng(args.random_seed)
    draws = {name: {k: [] for k in keys} for name in names}
    done = 0
    while done < args.resamples:
        b = min(args.block, args.resamples - done)
        W = np.zeros((n, b), dtype=np.float32)
        for j in range(b):
            W[:, j] = np.bincount(rng.integers(0, n, n), minlength=n)
        for name in names:
            for blk in BLOCKS:
                for key, val in zip(TRIPLE, built[name][blk].score(W)):
                    draws[name][f"{blk}_{key}"].append(val)
            for key, val in zip(BINARY, binary_scores(true_b, built[name]["pred_b"], W)):
                draws[name][key].append(val)
        done += b
        print(f"  {done}/{args.resamples}", end="\r", flush=True)
    print()

    stacked = {nm: {k: np.concatenate(v) for k, v in d.items()} for nm, d in draws.items()}
    ref = args.reference if args.reference in names else None
    rows = []
    for name in names:
        row = {"method": name}
        for k in keys:
            d = stacked[name][k]
            row[k] = round(float(d.mean()), 4)
            row[f"{k}_ci_lo"] = round(float(np.percentile(d, 2.5)), 4)
            row[f"{k}_ci_hi"] = round(float(np.percentile(d, 97.5)), 4)
            if ref and name != ref:
                diff = stacked[ref][k] - d
                row[f"{k}_vs_ref"] = round(float(diff.mean()), 4)
                row[f"{k}_vs_ref_lo"] = round(float(np.percentile(diff, 2.5)), 4)
                row[f"{k}_vs_ref_hi"] = round(float(np.percentile(diff, 97.5)), 4)
        rows.append(row)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    print(f"\n{args.resamples:,} resamples, n={n}, 95% percentile intervals\n")
    print(f"{'method':26s} {'overall MCC [95% CI]':30s} {'binary recall [95% CI]'}")
    for r in sorted(rows, key=lambda r: -r["overall_mcc"]):
        print(f"{r['method']:26s} {r['overall_mcc']:.3f} [{r['overall_mcc_ci_lo']:.3f},{r['overall_mcc_ci_hi']:.3f}]"
              f"        {r['binary_recall']:.3f} [{r['binary_recall_ci_lo']:.3f},{r['binary_recall_ci_hi']:.3f}]")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
