#!/usr/bin/env python3
"""Turn the per-annotator CSVs into the RQ2 ground truth the scorer reads.

    python annotations/build_ground_truth.py            # -> annotations/ground_truth.csv

Rules, in the order they are applied:

  1. An annotator's UNSURE ("can't tell") is treated as template-free: an image whose
     template a judge cannot identify is, by construction, not a recognizable template.
  2. The class of each image is the majority vote over annotators. An even split
     (e.g. 2-2-1 with no leader, or 2-2 among four) has no majority and the image is
     excluded from scoring; the count is reported.
  3. For images the majority calls templated, the template *name* is the majority among
     the names the templated-voting annotators gave, compared case/hyphen/underscore-
     insensitively, and written in its canonical surface form.
  4. Ties are resolved from tie_breaks.csv (sample_id, kind, label), a hand adjudication
     recorded separately so the raw annotator files stay untouched. A listed image that is
     not actually tied is reported and left to the majority.

Output columns:
    sample_id  the image file name, joins to every open_set_predictions.csv
    label      a template name, "Template-Free", or "Non-Meme"  (what the scorer uses)
    kind       TEMPLATE / TEMPLATELESS / NON_MEME               (the coarse class)
    n_votes    annotators who judged this image
    n_agree    how many voted for the winning class
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

KIND_TO_LABEL = {"TEMPLATE": "Templated-Meme", "TEMPLATELESS": "Template-Free", "NON_MEME": "Non-Meme"}


def norm(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations-dir", default=str(Path(__file__).parent / "annotations"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "ground_truth.csv"))
    ap.add_argument("--keep-unsure", action="store_true",
                    help="leave UNSURE as-is instead of mapping it to template-free")
    ap.add_argument("--tie-breaks", default=str(Path(__file__).parent / "tie_breaks.csv"),
                    help="hand adjudications for tied images; ignored if the file is absent")
    args = ap.parse_args()

    breaks: dict[str, tuple[str, str]] = {}
    tie_path = Path(args.tie_breaks)
    if tie_path.exists():
        with tie_path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                kind = (row.get("kind") or "").strip()
                if kind not in KIND_TO_LABEL:
                    raise SystemExit(f"{tie_path}: bad kind {kind!r} for {row.get('sample_id')}")
                breaks[(row.get("sample_id") or "").strip()] = (kind, (row.get("label") or "").strip())

    files = sorted(p for p in Path(args.annotations_dir).glob("*.csv") if p.name != "sessions.csv")
    if not files:
        raise SystemExit(f"no annotator CSVs under {args.annotations_dir}")

    votes: dict[str, list[str]] = {}
    names: dict[str, list[str]] = {}
    annotators = []
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            rows_in = list(csv.DictReader(fh))
        if not rows_in:
            continue
        annotators.append(rows_in[0].get("annotator", path.stem))
        for row in rows_in:
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                continue
            k = (row.get("true_label_kind") or "").strip()
            if k == "UNSURE" and not args.keep_unsure:
                k = "TEMPLATELESS"
            votes.setdefault(image_id, []).append(k)
            label = (row.get("true_label") or "").strip()
            if k == "TEMPLATE" and label:
                names.setdefault(image_id, []).append(label)

    rows, ties, adjudicated, not_tied = [], 0, 0, []
    for image_id in sorted(votes):
        cast = [v for v in votes[image_id] if v in KIND_TO_LABEL]
        ranked = Counter(cast).most_common() if cast else []
        is_tie = not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1])
        if image_id in breaks and not is_tie:
            not_tied.append(image_id)          # majority exists; the hand call is not needed
        if is_tie:
            if image_id not in breaks:
                ties += 1
                continue
            kind, forced = breaks[image_id]
            support = Counter(cast)[kind]
            adjudicated += 1
            label = forced if (kind == "TEMPLATE" and forced) else KIND_TO_LABEL[kind]
            rows.append({"sample_id": image_id, "label": label, "kind": kind,
                         "n_votes": len(votes[image_id]), "n_agree": support})
            continue
        kind, support = ranked[0]
        label = KIND_TO_LABEL[kind]
        if kind == "TEMPLATE" and names.get(image_id):
            by_key = Counter(norm(n) for n in names[image_id]).most_common()
            winner = by_key[0][0]
            label = next(n for n in names[image_id] if norm(n) == winner)
        rows.append({"sample_id": image_id, "label": label, "kind": kind,
                     "n_votes": len(votes[image_id]), "n_agree": support})

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample_id", "label", "kind", "n_votes", "n_agree"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    kinds = Counter(r["kind"] for r in rows)
    templated = [r for r in rows if r["kind"] == "TEMPLATE"]
    print(f"annotators : {', '.join(annotators)}")
    print(f"images     : {len(votes)} judged, {len(rows)} labelled "
          f"({adjudicated} ties adjudicated from {tie_path.name}, {ties} unresolved)")
    if not_tied:
        print(f"note       : {len(not_tied)} tie-break entr{'y' if len(not_tied)==1 else 'ies'} not needed "
              f"(majority exists): {', '.join(not_tied)}")
    print(f"unsure     : {'kept as UNSURE' if args.keep_unsure else 'mapped to template-free'}")
    for k in ("TEMPLATE", "TEMPLATELESS", "NON_MEME"):
        print(f"  {KIND_TO_LABEL[k]:16s} {kinds[k]:4d}  ({kinds[k] / len(rows) * 100:.1f}%)")
    print(f"templated  : {len(templated)} images across {len({r['label'] for r in templated})} distinct template names")
    print(f"unanimous  : {sum(1 for r in rows if r['n_agree'] == r['n_votes'])}/{len(rows)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
