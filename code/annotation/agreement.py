#!/usr/bin/env python3
"""Inter-annotator agreement for the RQ2 benchmark.

    python agreement.py --annotations-dir ../../data/rq2_benchmark/annotations

The paper reports Fleiss' kappa over the three-class coding, with an annotator's
"cannot tell" (UNSURE) mapped to template-free. That is the first row of the output and
is the number cited in the text. The remaining rows are alternative codings, printed so
that the choice is explicit and a reader can see how much it matters rather than having
to guess which convention was used.

The three-class coding is the one that matches the evaluation: the scorer compares
predictions against Templated-Meme, Template-Free and Non-Meme, so agreement is measured
in the same label space the models are judged in. UNSURE is folded into template-free
because an image whose template no judge can name is, by construction, not a recognisable
template; the same rule is applied in build_ground_truth.py, so the agreement figure and
the ground truth rest on one convention.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from collections import Counter
from pathlib import Path

CLASSES = ["TEMPLATE", "TEMPLATELESS", "NON_MEME"]


def fleiss_kappa(items: list[list[str]], categories: list[str]) -> float:
    """Fleiss' kappa for a fixed number of raters per item."""
    n = len(items[0])
    if n < 2:
        raise ValueError("need at least two raters")
    agreements, totals = [], Counter()
    for votes in items:
        counts = Counter(votes)
        agreements.append((sum(counts[c] ** 2 for c in categories) - n) / (n * (n - 1)))
        for c in categories:
            totals[c] += counts[c]
    p_bar = sum(agreements) / len(agreements)
    p_e = sum((totals[c] / (len(items) * n)) ** 2 for c in categories)
    return (p_bar - p_e) / (1 - p_e)


def load(annotations_dir: Path, keep_unsure: bool) -> dict[str, list[str]]:
    votes: dict[str, list[str]] = {}
    for path in sorted(annotations_dir.glob("*.csv")):
        if path.name == "sessions.csv":
            continue
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                kind = (row.get("true_label_kind") or "").strip()
                if kind == "UNSURE" and not keep_unsure:
                    kind = "TEMPLATELESS"
                if kind in CLASSES or (keep_unsure and kind == "UNSURE"):
                    votes.setdefault(row["image_id"], []).append(kind)
    return votes


def report(label: str, items: list[list[str]], categories: list[str]) -> None:
    k = fleiss_kappa(items, categories)
    n = len(items[0])
    pairwise = [
        sum(1 for v in items if v[i] == v[j]) / len(items)
        for i, j in itertools.combinations(range(n), 2)
    ]
    unanimous = sum(1 for v in items if len(set(v)) == 1) / len(items)
    print(f"  {label:52s} kappa={k:.4f}  pairwise={sum(pairwise)/len(pairwise):.4f}  "
          f"unanimous={unanimous:.1%}  n={len(items)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default = Path(__file__).resolve().parent.parent.parent / "data" / "rq2_benchmark" / "annotations"
    ap.add_argument("--annotations-dir", default=str(default))
    args = ap.parse_args()

    d = Path(args.annotations_dir)
    votes = load(d, keep_unsure=False)
    if not votes:
        raise SystemExit(f"no annotator CSVs under {d}")
    n_raters = max(len(v) for v in votes.values())
    items = [v for v in votes.values() if len(v) == n_raters]

    print(f"annotators : {n_raters}")
    print(f"images     : {len(items)} judged by all {n_raters} (of {len(votes)} total)\n")

    print("REPORTED IN THE PAPER")
    report("3-class, UNSURE mapped to template-free", items, CLASSES)

    print("\nALTERNATIVE CODINGS, for transparency")
    report("binary: templated vs rest",
           [["T" if x == "TEMPLATE" else "R" for x in v] for v in items], ["T", "R"])
    report("binary: meme vs non-meme",
           [["M" if x != "NON_MEME" else "N" for x in v] for v in items], ["M", "N"])

    kept = load(d, keep_unsure=True)
    kept_items = [v for v in kept.values() if len(v) == n_raters]
    n_unsure = sum(v.count("UNSURE") for v in kept_items)
    report(f"4-class, UNSURE kept as its own class ({n_unsure} votes)",
           kept_items, CLASSES + ["UNSURE"])


if __name__ == "__main__":
    main()
