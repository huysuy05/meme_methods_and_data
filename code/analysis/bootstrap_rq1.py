#!/usr/bin/env python3
"""Two-stage bootstrap intervals for the RQ1 closed-set metrics.

    python bootstrap_rq1.py --predictions-root /path/to/rq1-predictions

Stage one resamples the test images within each of the three partitions (seeds 40, 41, 42)
independently, B times, giving a within-seed bootstrap mean and variance per seed. Stage two
combines them: the reported estimate is the grand mean of the three seed-level means, and
the interval is

    mean +/- t(0.975, 2) * sqrt( V_between / 3 + V_within / (3B) )

where V_between is the sample variance of the three seed-level means and V_within is their
average bootstrap variance. The t-distribution carries two degrees of freedom because the
between-seed component is estimated from three partitions.

No model is retrained: this reads the per-image predictions each run wrote. Those files are
too large to ship with this package and are hosted separately (see README.md); point
--predictions-root at them. The expected layout is one directory per method, each holding
seed_40/, seed_41/ and seed_42/ with a test_predictions.csv inside.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

SEEDS = (40, 41, 42)
T2 = 4.302653          # t(0.975, df=2)


def load_seed(root: Path, seed: int):
    """Ground truth and every method's prediction for one partition, aligned by image path."""
    truth, preds = None, {}
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        f = method_dir / f"seed_{seed}" / "test_predictions.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f).set_index("image_path")
        if truth is None:
            truth = df["template"]
        elif not truth.index.equals(df.index):
            raise SystemExit(f"seed {seed}, {method_dir.name}: row index differs from the others")
        preds[method_dir.name] = df["pred_template"].astype(str)
    if truth is None:
        raise SystemExit(f"no test_predictions.csv found for seed {seed} under {root}")
    return truth, preds


def build(truth: pd.Series, preds: dict[str, pd.Series]):
    """One-hot row indicators, so a block of resamples is one sparse product per method."""
    vocab = sorted(set(truth) | {v for s in preds.values() for v in s})
    code = {label: i for i, label in enumerate(vocab)}
    n, k = len(truth), len(vocab)
    t_idx = truth.map(code).to_numpy()
    ones, rows = np.ones(n, dtype=np.float32), np.arange(n)
    T = csr_matrix((ones, (rows, t_idx)), shape=(n, k))
    per = {}
    for name, s in preds.items():
        p_idx = s.map(code).to_numpy()
        hit = t_idx == p_idx
        per[name] = (csr_matrix((ones, (rows, p_idx)), shape=(n, k)),
                     csr_matrix((ones[hit], (rows[hit], t_idx[hit])), shape=(n, k)))
    return T, per, n


def metrics(true_c, pred_c, tp_c, n):
    """Weighted F1, MCC and Cohen's kappa from class marginals and the diagonal.

    All three are functions of the contingency table, so a resample is evaluated from counts
    rather than by rescoring every row. Classes absent from a resample contribute zero, the
    same convention as zero_division=0 in scikit-learn.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = pred_c + true_c
        f1 = (np.where(denom > 0, 2.0 * tp_c / denom, 0.0) * true_c).sum(axis=0) / n
        correct = tp_c.sum(axis=0)
        p_e = (true_c * pred_c).sum(axis=0) / (n * n)
        kappa = np.where(p_e < 1, (correct / n - p_e) / (1.0 - p_e), 0.0)
        cov = n * correct - (true_c * pred_c).sum(axis=0)
        var = np.sqrt((n * n - (pred_c ** 2).sum(axis=0)) * (n * n - (true_c ** 2).sum(axis=0)))
        mcc = np.where(var > 0, cov / var, 0.0)
    return np.stack([f1, mcc, kappa])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-root", required=True,
                    help="directory of per-method closed-set predictions (hosted separately)")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--block", type=int, default=500, help="resamples held in memory at once")
    ap.add_argument("--random-seed", type=int, default=40)
    ap.add_argument("--out", default="rq1_bootstrap.csv")
    args = ap.parse_args()

    root = Path(args.predictions_root).expanduser().resolve()
    stage, names = {}, None
    for seed in SEEDS:
        truth, preds = load_seed(root, seed)
        stage[seed] = build(truth, preds)
        names = names or list(preds)
        print(f"seed {seed}: {len(truth)} images, {truth.nunique()} classes, {len(preds)} methods")

    rng = np.random.default_rng(args.random_seed)
    m = {n: np.zeros((3, 3)) for n in names}     # [seed, metric] bootstrap mean
    v = {n: np.zeros((3, 3)) for n in names}     # [seed, metric] bootstrap variance
    for si, seed in enumerate(SEEDS):
        T, per, n = stage[seed]
        acc = {name: [] for name in names}
        done = 0
        while done < args.resamples:
            b = min(args.block, args.resamples - done)
            W = np.zeros((n, b), dtype=np.float32)
            for j in range(b):
                W[:, j] = np.bincount(rng.integers(0, n, n), minlength=n)
            true_c = np.asarray(T.T @ W)
            for name in names:
                P, D = per[name]
                acc[name].append(metrics(true_c, np.asarray(P.T @ W), np.asarray(D.T @ W), n))
            done += b
            print(f"  seed {seed}: {done}/{args.resamples}", end="\r", flush=True)
        for name in names:
            d = np.concatenate(acc[name], axis=1)
            m[name][si], v[name][si] = d.mean(axis=1), d.var(axis=1, ddof=1)
    print()

    rows = []
    for name in names:
        grand = m[name].mean(axis=0)
        se = np.sqrt(m[name].var(axis=0, ddof=1) / 3 + v[name].mean(axis=0) / (3 * args.resamples))
        row = {"model": name}
        for i, met in enumerate(("f1", "mcc", "kappa")):
            row[met] = round(float(grand[i]), 4)
            row[f"{met}_ci_lo"] = round(float(grand[i] - T2 * se[i]), 4)
            row[f"{met}_ci_hi"] = round(float(grand[i] + T2 * se[i]), 4)
        rows.append(row)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    print(f"\n{args.resamples:,} resamples per seed, 95% intervals on the grand mean\n")
    print(f"{'model':30s} {'MCC [95% CI]':30s} {'weighted F1 [95% CI]'}")
    for r in sorted(rows, key=lambda r: -r["mcc"]):
        print(f"{r['model']:30s} {r['mcc']:.4f} [{r['mcc_ci_lo']:.4f},{r['mcc_ci_hi']:.4f}]   "
              f"{r['f1']:.4f} [{r['f1_ci_lo']:.4f},{r['f1_ci_hi']:.4f}]")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
