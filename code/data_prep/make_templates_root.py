#!/usr/bin/env python3
"""Build the ``--templates-root`` directory the annotation front end expects.

The annotation tool shows example images of any template a model predicted. It reads
them from a directory holding one entry per template name, each entry being a folder of
images of that template. In our working copy those entries were symlinks into the
ImgFlip source tree; symlinks do not survive redistribution, so this script recreates
them from whatever path you unpacked the ImgFlip images to.

    python make_templates_root.py \
        --imgflip-root /path/to/imgflip-canonical/images/_filtered_images \
        --templates-csv ../../data/manifests/imgflip_templates.csv \
        --out ./templates_root

By default it creates symlinks, which costs no disk. Pass ``--copy`` if your filesystem
or archive format cannot carry them, and ``--max-images N`` to copy only the first N
images per template, which is ample for the three examples the tool displays.

Only the templates listed in ``imgflip_templates.csv`` are linked, so the result matches
the 1,704-template label space used in the paper rather than the full source tree.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imgflip-root", required=True,
                    help="directory containing one folder per template (…/images/_filtered_images)")
    default_csv = Path(__file__).resolve().parent.parent.parent / "data" / "manifests" / "imgflip_templates.csv"
    ap.add_argument("--templates-csv", default=str(default_csv),
                    help="manifest listing the templates in the label space "
                         "(defaults to the copy shipped with this package, regardless of cwd)")
    ap.add_argument("--out", default="./templates_root", help="directory to create")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    ap.add_argument("--max-images", type=int, default=None,
                    help="with --copy, copy at most this many images per template")
    args = ap.parse_args()

    source = Path(args.imgflip_root).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"not a directory: {source}")

    with open(args.templates_csv, newline="", encoding="utf-8-sig") as fh:
        wanted = [row["template"].strip() for row in csv.DictReader(fh) if row.get("template", "").strip()]
    if not wanted:
        raise SystemExit(f"no templates listed in {args.templates_csv}")

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    made, missing = 0, []
    for name in wanted:
        src = source / name
        if not src.is_dir():
            missing.append(name)
            continue
        dst = out / name
        if dst.exists() or dst.is_symlink():
            continue
        if args.copy:
            dst.mkdir(parents=True)
            files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            for p in files[: args.max_images] if args.max_images else files:
                shutil.copy2(p, dst / p.name)
        else:
            os.symlink(src, dst)
        made += 1

    print(f"templates requested : {len(wanted)}")
    print(f"created             : {made} ({'copies' if args.copy else 'symlinks'}) in {out}")
    if missing:
        print(f"missing from source : {len(missing)}")
        for name in missing[:10]:
            print(f"    {name}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
        print("A non-empty missing list usually means --imgflip-root points at the wrong level;")
        print("it should be the directory that directly contains the per-template folders.")


if __name__ == "__main__":
    main()
