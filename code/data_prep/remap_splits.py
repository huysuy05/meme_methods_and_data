#!/usr/bin/env python3
"""Write copies of the split parquets whose image paths point at a real ImgFlip tree.

The parquets store every ImgFlip image under the placeholder root /path/to/imgflip-canonical.
Eight evaluation scripts can swap that prefix themselves (--path-prefix-from/--path-prefix-to),
but the three clustering scripts read the paths as they are. This writes remapped copies that
any script can be given:

    python remap_splits.py --out /tmp/splits      # -> /tmp/splits/imgflip_80_20_seed{40,41,42}/

By default it reads the splits from the dataset release and points them at the ImgFlip images
shipped in it, $DATA_ROOT/imgflip, and it reports any remapped path that does not exist.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

PLACEHOLDER = "/path/to/imgflip-canonical"
# The dataset release: $DATA_ROOT if set, otherwise data/ in this package.
DATA_ROOT = Path(os.environ.get("DATA_ROOT") or Path(__file__).resolve().parent.parent.parent / "data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default=str(DATA_ROOT / "splits"), help="folder holding imgflip_80_20_seed*/")
    ap.add_argument("--imgflip-root", default=str(DATA_ROOT / "imgflip"),
                    help="replaces the placeholder root; must contain images/_filtered_images/")
    ap.add_argument("--out", required=True, help="where to write the remapped copies")
    ap.add_argument("--seeds", default="40,41,42")
    args = ap.parse_args()

    root = str(Path(args.imgflip_root).expanduser().resolve())
    for seed in [s.strip() for s in args.seeds.split(",") if s.strip()]:
        dest = Path(args.out) / f"imgflip_80_20_seed{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        for part in ("train", "test"):
            df = pd.read_parquet(Path(args.splits) / f"imgflip_80_20_seed{seed}" / f"{part}.parquet")
            df["image_path"] = df["image_path"].str.replace(PLACEHOLDER, root, n=1, regex=False)
            missing = int((~df["image_path"].map(os.path.exists)).sum())
            df.to_parquet(dest / f"{part}.parquet", index=False)
            print(f"seed {seed} {part:5s}: {len(df):,} rows -> {dest / f'{part}.parquet'}"
                  + (f"  ({missing:,} paths missing under {root})" if missing else ""))


if __name__ == "__main__":
    main()
