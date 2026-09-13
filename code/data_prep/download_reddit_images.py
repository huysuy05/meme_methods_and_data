#!/usr/bin/env python3
"""
Download Reddit meme images referenced in the Reddit2024_nolabel parquet.

Source of links:
    ./Reddit2024_nolabel/subreddits23/meme_submissions.zst.parquet
      - column `id`  : already the filename base, e.g. "meme_submissions_32897"
      - column `url` : the post's link (mostly i.redd.it direct images)
      - (image_url column is i.redditmedia.com -> DEAD (HTTP 500), so we ignore it)

Images are written to:
    ./Reddit2024_nolabel/images/<id>.jpg
with the SAME naming convention as the 154k images already there (all ".jpg").

Deleted / removed images are NOT saved. We treat a link as "deleted" when:
  - it returns a non-200 status (i.redd.it deleted -> HTTP 404), or
  - the final response is not an image/* content-type (imgur removed -> html page), or
  - it is the imgur "removed.png" placeholder, or
  - the bytes don't decode as a valid image.

Already-present images are skipped, so the script is resumable: just re-run it.

Usage:
    python download_reddit_images.py                 # full run
    python download_reddit_images.py --sample 500    # test on first 500 missing
    python download_reddit_images.py --workers 48    # tune concurrency
"""
import argparse
import io
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image

# Defaults are relative to the working directory; override with --parquet / --images-dir.
# Both are overridable on the CLI so a future move only needs a flag, not a code edit.
DEFAULT_PARQUET = "./Reddit2024_nolabel/subreddits23/meme_submissions.zst.parquet"
DEFAULT_IMAGES_DIR = "./Reddit2024_nolabel/images"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")
# Domains that never point at a single downloadable image -> don't waste a request.
SKIP_DOMAINS = {
    "youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com",
    "v.redd.it", "www.reddit.com", "reddit.com", "old.reddit.com",
    "gfycat.com", "redgifs.com", "www.redgifs.com",
    "streamable.com", "twitter.com", "x.com",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; meme-archiver/1.0)"}

# Set from CLI args in main(). Kept as module globals because fetch_one runs in worker
# threads and reads them directly.
IMAGES_DIR = DEFAULT_IMAGES_DIR
PARQUET = DEFAULT_PARQUET
RESIZE_MAX = 0          # 0 = keep original; >0 = cap the longest side to this many px
JPEG_QUALITY = 90


def resolve_url(url: str):
    """Return a best-guess DIRECT image URL, or None to skip this row."""
    if not url:
        return None
    u = url.strip()
    parsed = urlparse(u)
    host = parsed.netloc.lower()
    path = parsed.path.lower()

    if host in SKIP_DOMAINS:
        return None

    # imgur albums / galleries are multi-image -> skip
    if "imgur.com" in host and (path.startswith("/a/") or path.startswith("/gallery/")):
        return None

    # already a direct image link
    if path.endswith(IMAGE_EXTS):
        return u

    # i.redd.it links normally already carry an extension; keep as-is
    if host == "i.redd.it":
        return u

    # imgur single-image page (e.g. imgur.com/GGslZex) -> direct .jpg
    if host in ("imgur.com", "www.imgur.com", "m.imgur.com", "i.imgur.com"):
        slug = parsed.path.strip("/")
        if slug and "/" not in slug:
            return f"https://i.imgur.com/{slug}.jpg"
        return None

    # anything else: attempt it; the content-type check will reject non-images
    return u


def is_valid_image(content: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(content)) as im:
            im.verify()
        return True
    except Exception:
        return False


class Counters:
    def __init__(self):
        self.lock = threading.Lock()
        self.downloaded = 0
        self.deleted = 0        # 404 / removed / non-image
        self.skipped = 0        # unresolvable (album, video, etc.)
        self.errors = 0         # network/timeout failures

    def bump(self, field):
        with self.lock:
            setattr(self, field, getattr(self, field) + 1)


def fetch_one(session, row, counters, timeout):
    fid, url = row
    dest = os.path.join(IMAGES_DIR, f"{fid}.jpg")
    if os.path.exists(dest):
        return "exists"

    target = resolve_url(url)
    if target is None:
        counters.bump("skipped")
        return "skipped"

    try:
        resp = session.get(target, headers=HEADERS, timeout=timeout,
                           stream=True, allow_redirects=True)
    except requests.RequestException:
        counters.bump("errors")
        return "error"

    try:
        if resp.status_code != 200:
            counters.bump("deleted")
            return "deleted"

        ctype = resp.headers.get("Content-Type", "").lower()
        if not ctype.startswith("image/"):
            # imgur removed -> redirects to an html page
            counters.bump("deleted")
            return "deleted"

        # cap read to avoid pathological huge bodies (25 MB)
        content = resp.raw.read(25 * 1024 * 1024, decode_content=True)
    finally:
        resp.close()

    # imgur "removed.png" placeholder is a tiny 503-byte png
    final_url = resp.url.lower()
    if "removed.png" in final_url or len(content) < 512:
        counters.bump("deleted")
        return "deleted"

    if not is_valid_image(content):
        counters.bump("deleted")
        return "deleted"

    tmp = dest + ".part"
    try:
        if RESIZE_MAX > 0:
            # Downscale the longest side to RESIZE_MAX and re-encode as JPEG. Only ever
            # shrinks (thumbnail is a no-op when the image is already smaller), so no
            # upscaling artefacts. All downstream methods downscale far below this anyway.
            with Image.open(io.BytesIO(content)) as im:
                im = im.convert("RGB")
                im.thumbnail((RESIZE_MAX, RESIZE_MAX), Image.Resampling.LANCZOS)
                im.save(tmp, format="JPEG", quality=JPEG_QUALITY)
        else:
            with open(tmp, "wb") as f:
                f.write(content)
        os.replace(tmp, dest)
    except (OSError, ValueError):  # PIL raises OSError/ValueError on odd files
        if os.path.exists(tmp):
            os.remove(tmp)
        counters.bump("errors")
        return "error"

    counters.bump("downloaded")
    return "downloaded"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--sample", type=int, default=0,
                    help="only process the first N missing rows (0 = all)")
    ap.add_argument("--max-downloads", type=int, default=0,
                    help="stop after saving this many new images (0 = no cap)")
    ap.add_argument("--min-free-gb", type=float, default=5.0,
                    help="stop launching new fetches once free disk drops below this")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET,
                    help="path to the submissions parquet (id, url columns)")
    ap.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR,
                    help="output folder; existing <id>.jpg files here are skipped (resumable)")
    ap.add_argument("--resize", type=int, default=0,
                    help="cap the longest image side to this many px and re-encode as JPEG "
                         "(0 = keep original bytes). Shrinks only, never upscales.")
    ap.add_argument("--jpeg-quality", type=int, default=90,
                    help="JPEG quality used when --resize re-encodes (ignored otherwise)")
    args = ap.parse_args()

    global IMAGES_DIR, PARQUET, RESIZE_MAX, JPEG_QUALITY
    IMAGES_DIR = args.images_dir
    PARQUET = args.parquet
    RESIZE_MAX = max(0, args.resize)
    JPEG_QUALITY = args.jpeg_quality

    if not os.path.exists(PARQUET):
        raise SystemExit(f"Parquet not found: {PARQUET}")
    os.makedirs(IMAGES_DIR, exist_ok=True)
    print(f"parquet     = {PARQUET}")
    print(f"images_dir  = {IMAGES_DIR}")
    print(f"resize      = {'off' if RESIZE_MAX == 0 else str(RESIZE_MAX) + 'px @ q' + str(JPEG_QUALITY)}")
    print(f"free space  = {shutil.disk_usage(IMAGES_DIR).free / 1e9:.1f} GB\n", flush=True)

    print("Loading parquet ...", flush=True)
    df = pd.read_parquet(PARQUET, columns=["id", "url"])
    df = df.dropna(subset=["id", "url"]).drop_duplicates(subset=["id"])
    print(f"  {len(df):,} unique rows with a url", flush=True)

    existing = {n[:-4] for n in os.listdir(IMAGES_DIR) if n.endswith(".jpg")}
    print(f"  {len(existing):,} images already present", flush=True)

    todo = [(fid, url) for fid, url in zip(df["id"], df["url"]) if fid not in existing]
    if args.sample:
        todo = todo[:args.sample]
    total = len(todo)
    print(f"  {total:,} images to attempt\n", flush=True)
    if not total:
        print("Nothing to do.")
        return

    counters = Counters()
    start = time.time()
    session = requests.Session()
    stop_flag = threading.Event()

    def free_gb():
        return shutil.disk_usage(IMAGES_DIR).free / 1e9

    def guarded_fetch(row):
        # cheap pre-checks so we stop launching work when limits are hit
        if stop_flag.is_set():
            return "stopped"
        if args.max_downloads and counters.downloaded >= args.max_downloads:
            stop_flag.set()
            return "stopped"
        if free_gb() < args.min_free_gb:
            stop_flag.set()
            return "stopped"
        return fetch_one(session, row, counters, args.timeout)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(guarded_fetch, row) for row in todo]
        done = 0
        for _ in as_completed(futures):
            done += 1
            if done % 500 == 0 or done == total:
                el = time.time() - start
                rate = done / el if el else 0
                eta = (total - done) / rate / 60 if rate else 0
                with counters.lock:
                    print(f"[{done:,}/{total:,}] "
                          f"ok={counters.downloaded:,} deleted={counters.deleted:,} "
                          f"skip={counters.skipped:,} err={counters.errors:,} "
                          f"| {rate:.0f}/s ETA {eta:.0f}m", flush=True)

    if stop_flag.is_set():
        reason = "max-downloads reached" if (
            args.max_downloads and counters.downloaded >= args.max_downloads
        ) else f"free disk < {args.min_free_gb} GB"
        print(f"\n[STOPPED EARLY: {reason}]")

    final_total = sum(1 for n in os.listdir(IMAGES_DIR) if n.endswith(".jpg"))
    print("\n===== DONE =====")
    print(f"Newly downloaded : {counters.downloaded:,}")
    print(f"Deleted/removed  : {counters.deleted:,}")
    print(f"Skipped (album/video/non-image link): {counters.skipped:,}")
    print(f"Network errors   : {counters.errors:,}")
    print(f"Elapsed          : {(time.time()-start)/60:.1f} min")
    print(f"TOTAL images now in folder: {final_total:,}")


if __name__ == "__main__":
    main()
