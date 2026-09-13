#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import json
import random
import struct
import time
import zipfile
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

# Sentinel label/source for unlabeled discovery images (social-media corpus).
# majority_vote_mapping() filters on source == "imgflip_train", so these never vote.
UNLABELED_TEMPLATE = "UNLABELED"
SOCIAL_MEDIA_SOURCE = "social_media"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_int_list(text: str) -> tuple[int, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer seed.")
    return tuple(int(value) for value in values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rgb_image(path: str | Path) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            try:
                img = ImageOps.exif_transpose(img)
            except (SyntaxError, ValueError, OSError, struct.error):
                # Scraped social-media images often carry a malformed EXIF blob;
                # Pillow surfaces that as SyntaxError from TiffImagePlugin. Only the
                # orientation hint is unreadable, so keep the pixels and skip the rotate.
                pass
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, struct.error):
        return None


def collect_labeled_rows(
    dataset_root: Path,
    max_templates: int | None = None,
    max_images_per_template: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    template_dirs = sorted([path for path in dataset_root.iterdir() if path.is_dir()])
    if max_templates is not None:
        template_dirs = template_dirs[:max_templates]

    for template_dir in template_dirs:
        image_paths = sorted(
            [path for path in template_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS]
        )
        if max_images_per_template is not None:
            image_paths = image_paths[:max_images_per_template]
        for image_path in image_paths:
            rows.append(
                {
                    "image_path": str(image_path),
                    "template": template_dir.name,
                    "source": "imgflip",
                }
            )

    if not rows:
        raise ValueError(f"No labeled images found under {dataset_root}")
    return pd.DataFrame(rows)


def _read_parquet_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            parquet_names = [name for name in zf.namelist() if name.lower().endswith(".parquet")]
            if not parquet_names:
                raise ValueError(f"No parquet file found inside {path}")
            if len(parquet_names) > 1:
                preferred = [name for name in parquet_names if "meme_entries" in Path(name).name]
                if preferred:
                    parquet_name = preferred[0]
                else:
                    raise ValueError(
                        f"Multiple parquet files found inside {path}. "
                        "Please unzip it first or keep only the target parquet."
                    )
            else:
                parquet_name = parquet_names[0]
            with zf.open(parquet_name) as infile:
                return pd.read_parquet(io.BytesIO(infile.read()))
    return pd.read_parquet(path)


def _normalize_split_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if {"image_path", "template"}.issubset(df.columns):
        rows = df.copy()
    elif {"path", "template_name"}.issubset(df.columns):
        rows = df.loc[:, ["path", "template_name"]].rename(
            columns={"path": "image_path", "template_name": "template"}
        ).copy()
    else:
        raise ValueError(
            "Parquet must contain either ['image_path', 'template'] or ['path', 'template_name'] columns."
        )

    rows["image_path"] = rows["image_path"].astype(str)
    rows["template"] = rows["template"].astype(str)
    if "source" not in rows.columns:
        rows["source"] = "imgflip"
    else:
        rows["source"] = rows["source"].astype(str)
    return rows.reset_index(drop=True)


def collect_rows_from_parquet(
    parquet_path: Path,
    image_root: Path | None = None,
    path_prefix_from: str | None = None,
    path_prefix_to: str | None = None,
    max_templates: int | None = None,
    max_images_per_template: int | None = None,
) -> pd.DataFrame:
    df = _read_parquet_file(parquet_path)
    rows = _normalize_split_dataframe(df)

    if path_prefix_from is not None and path_prefix_to is not None:
        rows["image_path"] = rows["image_path"].str.replace(path_prefix_from, path_prefix_to, regex=False)

    if image_root is not None:
        image_root = image_root.expanduser().resolve()
        rows["image_path"] = rows["image_path"].map(
            lambda value: str(image_root / Path(value).parent.name / Path(value).name)
            if not Path(value).exists()
            else value
        )

    if max_templates is not None:
        keep_templates = sorted(rows["template"].unique())[:max_templates]
        rows = rows[rows["template"].isin(keep_templates)].copy()

    if max_images_per_template is not None:
        rows = (
            rows.groupby("template", group_keys=False)
            .head(max_images_per_template)
            .reset_index(drop=True)
        )
    else:
        rows = rows.reset_index(drop=True)

    if rows.empty:
        raise ValueError(f"No rows collected from {parquet_path}")
    return rows


def load_dataset_rows(
    dataset_root: str | None,
    parquet_path: str | None,
    image_root: str | None = None,
    path_prefix_from: str | None = None,
    path_prefix_to: str | None = None,
    max_templates: int | None = None,
    max_images_per_template: int | None = None,
) -> pd.DataFrame:
    if dataset_root:
        return collect_labeled_rows(
            Path(dataset_root).expanduser().resolve(),
            max_templates=max_templates,
            max_images_per_template=max_images_per_template,
        )
    if parquet_path:
        return collect_rows_from_parquet(
            Path(parquet_path).expanduser().resolve(),
            image_root=None if image_root is None else Path(image_root),
            path_prefix_from=path_prefix_from,
            path_prefix_to=path_prefix_to,
            max_templates=max_templates,
            max_images_per_template=max_images_per_template,
        )
    raise ValueError("Either dataset_root or parquet_path must be provided.")


def load_split_rows(
    train_parquet: str,
    test_parquet: str,
    image_root: str | None = None,
    path_prefix_from: str | None = None,
    path_prefix_to: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = collect_rows_from_parquet(
        Path(train_parquet).expanduser().resolve(),
        image_root=None if image_root is None else Path(image_root),
        path_prefix_from=path_prefix_from,
        path_prefix_to=path_prefix_to,
    )
    test_df = collect_rows_from_parquet(
        Path(test_parquet).expanduser().resolve(),
        image_root=None if image_root is None else Path(image_root),
        path_prefix_from=path_prefix_from,
        path_prefix_to=path_prefix_to,
    )
    train_df["source"] = "imgflip_train"
    test_df["source"] = "imgflip_test"
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def filter_valid_image_rows(df: pd.DataFrame) -> pd.DataFrame:
    keep_rows: list[bool] = []
    for image_path in tqdm(df["image_path"].tolist(), desc="verify images"):
        keep_rows.append(load_rgb_image(image_path) is not None)
    filtered = df[keep_rows].reset_index(drop=True)
    dropped = len(df) - len(filtered)
    if dropped:
        print(f"Dropped {dropped} unreadable images.")
    return filtered


def drop_small_classes(
    df: pd.DataFrame,
    min_images_per_class: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = df["template"].value_counts()
    keep_templates = counts[counts >= min_images_per_class].index
    filtered = df[df["template"].isin(keep_templates)].copy().reset_index(drop=True)

    removed_templates = counts[counts < min_images_per_class]
    summary = {
        "min_images_required_per_class": int(min_images_per_class),
        "templates_before": int(counts.shape[0]),
        "templates_after": int(filtered["template"].nunique()),
        "templates_removed": int(removed_templates.shape[0]),
        "images_before": int(len(df)),
        "images_after": int(len(filtered)),
        "images_removed": int(len(df) - len(filtered)),
        "removed_template_examples": removed_templates.head(20).to_dict(),
    }
    return filtered, summary


def stratified_split(
    df: pd.DataFrame,
    train_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df["template"],
        random_state=random_seed,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df["source"] = "imgflip_train"
    test_df["source"] = "imgflip_test"
    return train_df, test_df


def align_dataframe(df: pd.DataFrame, kept_paths: list[str]) -> pd.DataFrame:
    path_to_index = {path: idx for idx, path in enumerate(df["image_path"].tolist())}
    return df.iloc[[path_to_index[path] for path in kept_paths]].reset_index(drop=True)


def parse_path_list(text: str) -> tuple[Path, ...]:
    """Parse a comma-separated list of directories into resolved Paths."""
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one directory path.")
    return tuple(Path(value).expanduser().resolve() for value in values)


def exclude_paths_by_content(paths: list[Path], exclude_dir: Path, label: str = "images") -> list[Path]:
    """Drop every path whose bytes match a file in ``exclude_dir``.

    Matching is by SHA-256, never by name: the held-out sample is a folder of *copies*
    under new filenames, and the corpora contain byte-identical duplicates, so a name or
    path check would silently leave the originals in. File size is a cheap pre-filter, so
    only genuine candidates are hashed.
    """
    exclude_dir = Path(exclude_dir).expanduser().resolve()
    held_out = _hashes_of_folder(exclude_dir)
    sizes = {p.stat().st_size for p in exclude_dir.iterdir() if p.is_file()}
    candidates = [p for p in paths if p.stat().st_size in sizes]
    with ThreadPoolExecutor(max_workers=32) as pool:
        drop = {p for p, h in zip(candidates, pool.map(_sha256_file, candidates)) if h in held_out}
    kept = [p for p in paths if p not in drop]
    print(f"excluded_held_out_{label}={len(drop)} (by sha256, {len(candidates)} candidates hashed, {len(kept)} kept)")
    return kept


def collect_unlabeled_corpus_rows(
    roots: tuple[Path, ...] | list[Path],
    max_images: int | None = None,
    random_seed: int = 42,
    exclude_dir: Path | None = None,
) -> pd.DataFrame:
    """Enumerate images under one or more roots as an unlabeled discovery corpus.

    Rows carry ``template=UNLABELED_TEMPLATE`` and ``source=SOCIAL_MEDIA_SOURCE`` so that
    ``majority_vote_mapping`` (which only counts ``source == "imgflip_train"``) never treats
    them as label evidence. Directories are globbed at call time, so images added later are
    picked up on the next run without any code change.

    Images are NOT decoded here; unreadable files are skipped later during encoding.
    """
    paths: list[str] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Discovery corpus root does not exist: {root}")
        found = [
            str(path)
            for path in sorted(root.rglob("*"))
            if path.suffix.lower() in VALID_EXTS and path.is_file()
        ]
        print(f"corpus_root={root} images={len(found)}")
        paths.extend(found)

    paths = sorted(set(paths))  # deterministic order, no double-counting overlapping roots
    if exclude_dir is not None:
        # The open-set images were sampled *from* these corpora; without this their originals
        # would sit in the discovery set and the model would cluster the very images it is
        # then asked to predict.
        paths = [str(p) for p in exclude_paths_by_content([Path(p) for p in paths], exclude_dir, "corpus_images")]
    if max_images is not None and 0 < max_images < len(paths):
        rng = np.random.default_rng(random_seed)
        keep = rng.choice(len(paths), size=int(max_images), replace=False)
        paths = [paths[int(i)] for i in sorted(keep)]

    return pd.DataFrame(
        {
            "image_path": paths,
            "template": UNLABELED_TEMPLATE,
            "source": SOCIAL_MEDIA_SOURCE,
        }
    )


def restrict_to_labeled_clusters(
    vectors: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_to_template: dict[int, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only clusters that majority vote actually named.

    Test images are then assigned among named clusters only, so every prediction is a real
    template label and ``UNKNOWN_CLUSTER`` can never occur. Needed once an unlabeled corpus
    joins discovery, because many clusters then contain no ImgFlip images at all.
    """
    keep = [index for index, cluster_id in enumerate(cluster_ids) if int(cluster_id) in cluster_to_template]
    if not keep:
        raise ValueError("No cluster received a template label from the ImgFlip training split.")
    return vectors[keep], cluster_ids[keep]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def compute_metrics_with_rejection(y_true: np.ndarray, y_pred: np.ndarray, reject_token: str) -> dict[str, float]:
    metrics = compute_metrics(y_true, y_pred)
    covered_mask = y_pred != reject_token
    coverage = float(covered_mask.mean()) if y_pred.size else 0.0
    metrics["coverage"] = coverage
    if covered_mask.any():
        covered_true = y_true[covered_mask]
        covered_pred = y_pred[covered_mask]
        metrics["covered_accuracy"] = float(accuracy_score(covered_true, covered_pred))
        metrics["covered_count"] = int(covered_mask.sum())
    else:
        metrics["covered_accuracy"] = 0.0
        metrics["covered_count"] = 0
    metrics["rejected_count"] = int((~covered_mask).sum())
    return metrics


def finalize_run_timing(
    metadata: dict[str, Any],
    summary_row: dict[str, Any],
    run_started_at: str,
    start_perf: float,
) -> dict[str, Any]:
    duration_seconds = round(time.perf_counter() - start_perf, 3)
    duration_minutes = round(duration_seconds / 60.0, 3)
    timing = {
        "started_at": run_started_at,
        "finished_at": now_iso(),
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_minutes,
    }
    metadata["timing"] = timing
    summary_row["duration_seconds"] = duration_seconds
    summary_row["duration_minutes"] = duration_minutes
    return timing


def summarize_batch_timings(batch_rows: list[dict[str, Any]]) -> dict[str, float | int]:
    durations = [float(row["duration_seconds"]) for row in batch_rows if "duration_seconds" in row]
    if not durations:
        return {
            "run_count": int(len(batch_rows)),
            "average_duration_seconds": 0.0,
            "average_duration_minutes": 0.0,
            "total_duration_seconds": 0.0,
            "total_duration_minutes": 0.0,
        }
    total_duration_seconds = round(float(sum(durations)), 3)
    average_duration_seconds = round(total_duration_seconds / len(durations), 3)
    return {
        "run_count": int(len(durations)),
        "average_duration_seconds": average_duration_seconds,
        "average_duration_minutes": round(average_duration_seconds / 60.0, 3),
        "total_duration_seconds": total_duration_seconds,
        "total_duration_minutes": round(total_duration_seconds / 60.0, 3),
    }


SEED_PLACEHOLDER = "{seed}"


def resolve_seed_paths(config_fields: dict[str, Any], seed: int) -> dict[str, Any]:
    """Substitute ``{seed}`` in the split parquet paths for a per-seed run.

    Lets one command evaluate several seeds against *different* splits, e.g.
    ``--train-parquet splits/imgflip_80_20_seed{seed}/train.parquet``, so seed
    variance reflects the data partition and not just model initialisation.
    Paths without the placeholder pass through untouched, which keeps the
    existing single-fixed-split behaviour working.
    """
    resolved = dict(config_fields)
    for key in ("train_parquet", "test_parquet", "cnn_checkpoint"):
        value = resolved.get(key)
        if value is None:
            continue
        text = str(value)
        if SEED_PLACEHOLDER in text:
            resolved[key] = Path(text.replace(SEED_PLACEHOLDER, str(int(seed))))
    return resolved


# --------------------------------------------------------------------------------------
# Open-set prediction (RQ2): score an unlabeled folder with a fitted model
# --------------------------------------------------------------------------------------

OPEN_SET_TEMPLATE_FREE = "Template-Free"
OPEN_SET_NON_MEME = "Non-Meme"
OPEN_SET_TEMPLATED = "Templated-Meme"
OPEN_SET_LABELS = (OPEN_SET_TEMPLATE_FREE, OPEN_SET_NON_MEME)
OPEN_SET_SOURCE = "open_set"
OPEN_SET_PREDICTIONS_FILE = "open_set_predictions.csv"


def add_open_set_args(parser: argparse.ArgumentParser, default_reject_label: str = OPEN_SET_TEMPLATE_FREE) -> None:
    """Shared CLI for scoring a folder of unlabeled images after the normal ImgFlip run.

    The folder (e.g. ``Social_Media/random_sampled``) has no labels, so nothing is
    trained on it; the model fitted on the ImgFlip train split just predicts every
    image in it. Predictions land in ``open_set_predictions.csv`` next to the usual
    ``test_predictions.csv``.
    """
    parser.add_argument(
        "--open-set-dir",
        default=None,
        help="Folder of unlabeled images to predict after training/clustering. Each image gets either "
             "an ImgFlip template name or the reject label below.",
    )
    parser.add_argument(
        "--open-set-labels",
        default=None,
        help="Ground truth for the --open-set-dir images: a sample_id,label CSV or the annotation "
             "tool's merged.csv. When given, predictions are scored (MCC/kappa/F1 + templated-vs-rest "
             "precision/recall) into open_set_metrics.json. Can also be applied later with "
             "score_open_set.py without rerunning the model.",
    )
    parser.add_argument(
        "--open-set-reject-label",
        choices=list(OPEN_SET_LABELS),
        default=default_reject_label,
        help="Open-set class written when the model rejects an image as matching no known template. "
             f"Default for this script: {default_reject_label}.",
    )


def collect_open_set_rows(folder: Path) -> pd.DataFrame:
    """Every readable image directly inside ``folder``, unlabeled, in sorted filename order."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Open-set folder does not exist: {folder}")
    paths = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS)
    if not paths:
        raise ValueError(f"No images found in open-set folder: {folder}")
    df = pd.DataFrame(
        {
            "image_path": [str(path) for path in paths],
            "sample_id": [path.name for path in paths],
            "template": UNLABELED_TEMPLATE,
            "source": OPEN_SET_SOURCE,
        }
    )
    df = filter_valid_image_rows(df)
    print(f"open_set_dir={folder} images={len(df)}")
    return df


def open_set_labels(pred_templates: np.ndarray, reject_tokens: tuple[str, ...], reject_label: str) -> np.ndarray:
    """Map a model's internal reject token(s) onto the open-set class; template names pass through."""
    tokens = set(reject_tokens)
    return np.array([reject_label if str(pred) in tokens else str(pred) for pred in pred_templates], dtype=object)


def write_open_set_predictions(
    run_dir: Path,
    open_df: pd.DataFrame,
    pred_labels: np.ndarray,
    extra_columns: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write ``open_set_predictions.csv`` and return (path, summary for run_config.json).

    ``pred_label`` is the fine label: an ImgFlip template name, or one of the open-set
    classes. ``pred_class`` collapses that to the three RQ2 classes, so the binary
    "templated vs anything else" metrics and the template-level metrics both read off
    the same file.
    """
    pred_labels = np.asarray(pred_labels, dtype=object)
    out = open_df.loc[:, ["sample_id", "image_path"]].copy()
    out["pred_label"] = pred_labels
    out["pred_class"] = np.where(np.isin(pred_labels, OPEN_SET_LABELS), pred_labels, OPEN_SET_TEMPLATED)
    for name, values in (extra_columns or {}).items():
        out[name] = values

    path = run_dir / OPEN_SET_PREDICTIONS_FILE
    out.to_csv(path, index=False)

    class_counts = {str(k): int(v) for k, v in out["pred_class"].value_counts().items()}
    templated = out.loc[out["pred_class"] == OPEN_SET_TEMPLATED, "pred_label"]
    summary = {
        "images": int(len(out)),
        "predictions_file": str(path),
        "class_counts": class_counts,
        "distinct_templates_predicted": int(templated.nunique()),
    }
    print(f"open_set_images={len(out)} class_counts={class_counts}")
    print(f"open_set_predictions_saved={path}")
    return path, summary


# --------------------------------------------------------------------------------------
# Open-set scoring (RQ2): compare open_set_predictions.csv against annotated ground truth
# --------------------------------------------------------------------------------------

OPEN_SET_METRICS_FILE = "open_set_metrics.json"

# Spellings of the two special classes accepted in a ground-truth file. Matching is done
# on a normalised key (lower-case, runs of non-alphanumerics -> "_"), so "Non-Meme",
# "NON_MEME" and "non meme" all land on the same class. Mirrors annotations/annotate.py.
_NON_MEME_KEYS = {"non_meme", "nonmeme", "not_meme", "not_a_meme", "no_meme"}
_TEMPLATE_FREE_KEYS = {
    "templateless", "template_less", "template_free", "templatefree", "no_template",
    "non_templated", "not_templated", "untemplated",
}


def _normalise_label_key(label: Any) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", str(label if label is not None else "").strip().lower()).strip("_")


def canonical_open_set_label(label: Any, template_vocab: dict[str, str] | None = None) -> str:
    """Map a free-text ground-truth label onto the label space the models emit.

    Special classes fold to ``Non-Meme`` / ``Template-Free``; anything else is matched
    case/hyphen/underscore-insensitively against ``template_vocab`` (normalised key ->
    canonical template name) and returned canonicalised, or verbatim when unknown.
    """
    key = _normalise_label_key(label)
    if key in _NON_MEME_KEYS:
        return OPEN_SET_NON_MEME
    if key in _TEMPLATE_FREE_KEYS:
        return OPEN_SET_TEMPLATE_FREE
    if template_vocab and key in template_vocab:
        return template_vocab[key]
    return str(label).strip()


def load_open_set_labels(path: Path, template_vocab: dict[str, str] | None = None) -> pd.DataFrame:
    """Read ground truth for the open-set images -> DataFrame(sample_id, label).

    Accepts either a plain ``sample_id,label`` file or the annotation tool's ``merged.csv``
    (``image_id`` + ``consensus_label``, one row per annotator; the first row per image is
    used and blank consensus -- a tie -- drops the image). Column names are matched in
    order of preference so both shapes work unchanged.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    id_col = next((c for c in ("sample_id", "image_id") if c in df.columns), None)
    label_col = next((c for c in ("label", "consensus_label", "true_label", "template") if c in df.columns), None)
    if id_col is None or label_col is None:
        raise ValueError(
            f"{path}: need an id column (sample_id|image_id) and a label column "
            f"(label|consensus_label|true_label|template); found {list(df.columns)}"
        )
    out = df[[id_col, label_col]].rename(columns={id_col: "sample_id", label_col: "label"})
    out["sample_id"] = out["sample_id"].str.strip()
    out["label"] = out["label"].str.strip()
    out = out[out["sample_id"] != ""].drop_duplicates("sample_id", keep="first")
    unresolved = int((out["label"] == "").sum())
    out = out[out["label"] != ""].reset_index(drop=True)
    out["label"] = [canonical_open_set_label(v, template_vocab) for v in out["label"]]
    print(f"open_set_labels={path} images={len(out)} unresolved_dropped={unresolved}")
    return out


def compute_open_set_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Three views of the same predictions.

    ``fine``   -- full label space (every template name + the two special classes), the
                 exact ``compute_metrics`` the ImgFlip run reports.
    ``coarse`` -- three classes: Templated-Meme / Template-Free / Non-Meme.
    ``binary`` -- templated vs anything else; precision/recall/F1 of the templated class,
                 i.e. how well the model separates known templates from the rest.
    """
    from sklearn.metrics import precision_recall_fscore_support

    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    special = set(OPEN_SET_LABELS)

    def coarse(labels: np.ndarray) -> np.ndarray:
        return np.array([v if v in special else OPEN_SET_TEMPLATED for v in labels], dtype=object)

    true_c, pred_c = coarse(y_true), coarse(y_pred)
    true_b = true_c == OPEN_SET_TEMPLATED
    pred_b = pred_c == OPEN_SET_TEMPLATED
    p, r, f, _ = precision_recall_fscore_support(true_b, pred_b, average="binary", zero_division=0)

    # Among images that really are templated, how often is the *right* template named?
    templated_mask = true_b
    template_hit = float((y_true[templated_mask] == y_pred[templated_mask]).mean()) if templated_mask.any() else 0.0

    # Two conditional views the open-set table reports separately, because they answer
    # different questions. "predicted templated" asks: when this model claims a template,
    # how good is the claim? "true templated" asks: on the images that really are
    # templated, how well does it do -- including the ones it wrongly rejected.
    def _subset(mask: np.ndarray) -> dict[str, Any]:
        if not mask.any():
            return {"n": 0}
        return {"n": int(mask.sum()), **compute_metrics(y_true[mask], y_pred[mask])}

    on_predicted_templated = _subset(pred_b)
    on_true_templated = _subset(true_b)

    return {
        "n_images": int(len(y_true)),
        "true_class_counts": {str(k): int(v) for k, v in zip(*np.unique(true_c, return_counts=True))},
        "pred_class_counts": {str(k): int(v) for k, v in zip(*np.unique(pred_c, return_counts=True))},
        "fine": compute_metrics(y_true, y_pred),
        "coarse": compute_metrics(true_c, pred_c),
        "binary_templated_vs_rest": {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "accuracy": float((true_b == pred_b).mean()),
            "n_true_templated": int(true_b.sum()),
            "n_pred_templated": int(pred_b.sum()),
        },
        "on_predicted_templated": on_predicted_templated,
        "on_true_templated": on_true_templated,
        "template_accuracy_on_templated": template_hit,
    }


def score_open_set_predictions(
    run_dir: Path,
    labels_path: Path,
    template_vocab: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Join ``run_dir/open_set_predictions.csv`` with ground truth, score, write metrics."""
    run_dir = Path(run_dir)
    preds = pd.read_csv(run_dir / OPEN_SET_PREDICTIONS_FILE, dtype={"sample_id": str, "pred_label": str})
    if template_vocab is None:
        # Whatever the model could emit is, by construction, the canonical spelling.
        template_vocab = {
            _normalise_label_key(v): v
            for v in preds["pred_label"].unique()
            if v not in OPEN_SET_LABELS
        }
    labels = load_open_set_labels(Path(labels_path), template_vocab)

    joined = preds.merge(labels, on="sample_id", how="inner")
    missing_pred = len(labels) - len(joined)
    if joined.empty:
        raise ValueError("No overlap between predictions and labels on sample_id.")
    metrics = compute_open_set_metrics(joined["label"].to_numpy(), joined["pred_label"].to_numpy())
    metrics["labels_file"] = str(labels_path)
    metrics["labeled_images_without_prediction"] = int(missing_pred)
    metrics["predicted_images_without_label"] = int(len(preds) - len(joined))
    unknown = sorted({v for v in joined["label"] if v not in OPEN_SET_LABELS and v not in set(preds["pred_label"])})
    if unknown:
        metrics["ground_truth_templates_never_predicted"] = len(unknown)

    (run_dir / OPEN_SET_METRICS_FILE).write_text(json.dumps(metrics, indent=2))
    b = metrics["binary_templated_vs_rest"]
    print(
        f"open_set_scored={len(joined)} "
        f"fine_mcc={metrics['fine']['mcc']:.4f} fine_f1={metrics['fine']['f1']:.4f} "
        f"coarse_mcc={metrics['coarse']['mcc']:.4f} "
        f"binary_precision={b['precision']:.4f} binary_recall={b['recall']:.4f}"
    )
    print(f"open_set_metrics_saved={run_dir / OPEN_SET_METRICS_FILE}")
    return metrics


# --------------------------------------------------------------------------------------
# Non-Meme as a class the methods themselves learn (RQ2)
# --------------------------------------------------------------------------------------

NON_MEME_SOURCE = "non_meme"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashes_of_folder(folder: Path, num_workers: int = 32) -> set[str]:
    paths = [p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        return set(pool.map(_sha256_file, paths))


def collect_non_meme_rows(
    root: Path,
    exclude_dir: Path | None = None,
    max_images: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Images under ``root`` as an ordinary labelled class named ``Non-Meme``.

    This is what lets a method answer "meme or not" with its *own* machinery -- an extra
    class in the classifier head, extra reference vectors for the neighbour methods, extra
    labelled voters for the clustering methods -- rather than deferring to a second model.

    ``exclude_dir`` (the open-set sample) is removed by SHA-256, not by filename, because
    the sampled images are copies and the corpora contain byte-identical duplicates. File
    size is used as a cheap pre-filter so only genuine candidates are hashed.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--non-meme-root does not exist: {root}")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS)
    if not paths:
        raise ValueError(f"No images found under --non-meme-root: {root}")
    print(f"non_meme_root={root} images={len(paths)}")

    if exclude_dir is not None:
        paths = exclude_paths_by_content(paths, exclude_dir, "non_meme_images")

    if max_images is not None and 0 < max_images < len(paths):
        rng = np.random.default_rng(random_seed)
        keep = rng.choice(len(paths), size=int(max_images), replace=False)
        paths = [paths[int(i)] for i in sorted(keep)]
        print(f"non_meme_subsampled_to={len(paths)}")

    return pd.DataFrame(
        {
            "image_path": [str(p) for p in paths],
            "template": OPEN_SET_NON_MEME,
            "source": NON_MEME_SOURCE,
        }
    )


def add_non_meme_class(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    non_meme_df: pd.DataFrame,
    train_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    """Split the Non-Meme images the same way as the templates and fold them in.

    Returns the widened frames plus a summary for ``run_config.json``. When ``test_df`` is
    ``None`` every Non-Meme image goes to train, which is what the clustering methods want:
    they need the class present among the labelled discovery rows, and are scored on the
    unchanged ImgFlip test split.
    """
    if test_df is None:
        widened_train = pd.concat([train_df, non_meme_df], ignore_index=True)
        summary = {
            "non_meme_images": int(len(non_meme_df)),
            "non_meme_train": int(len(non_meme_df)),
            "non_meme_test": 0,
        }
        print(f"non_meme_class_added train={len(non_meme_df)} test=0")
        return widened_train, None, summary

    nm_train, nm_test = train_test_split(
        non_meme_df, train_size=train_size, random_state=random_seed, shuffle=True
    )
    widened_train = pd.concat([train_df, nm_train.reset_index(drop=True)], ignore_index=True)
    widened_test = pd.concat([test_df, nm_test.reset_index(drop=True)], ignore_index=True)
    summary = {
        "non_meme_images": int(len(non_meme_df)),
        "non_meme_train": int(len(nm_train)),
        "non_meme_test": int(len(nm_test)),
    }
    print(f"non_meme_class_added train={len(nm_train)} test={len(nm_test)} "
          f"(classes now {widened_train['template'].nunique()})")
    return widened_train, widened_test, summary


def add_non_meme_args(parser: argparse.ArgumentParser) -> None:
    """CLI for training the ``Non-Meme`` class into the method itself."""
    parser.add_argument(
        "--non-meme-root",
        default=None,
        help="Folder of non-meme images (e.g. ../non_memes). They join the method's own label "
             "space as a class called 'Non-Meme', so the model predicts meme-vs-non-meme with the "
             "same mechanism it uses for templates -- an extra class in the head, extra reference "
             "vectors, or extra labelled cluster voters. Images also present in --open-set-dir are "
             "removed by sha256 first.",
    )
    parser.add_argument(
        "--max-non-meme-images",
        type=int,
        default=None,
        help="Subsample the non-meme class to this many images. Useful because the corpus is large "
             "relative to any single template (~72 images), so leaving it uncapped makes Non-Meme "
             "by far the biggest class. Default: use all of them.",
    )


# --------------------------------------------------------------------------------------
# Cluster evidence for annotation (RQ2)
# --------------------------------------------------------------------------------------

CLUSTER_EXAMPLES_FILE = "cluster_examples.csv"


def write_cluster_examples(
    run_dir: Path,
    discovery_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_to_template: dict[int, str],
    member_scores: np.ndarray | None = None,
    n_examples: int = 6,
) -> tuple[Path, dict[str, Any]]:
    """For every cluster, a few representative member images.

    An open-set image landing in an *unnamed* cluster tells an annotator nothing on its
    own -- but the other images in that cluster do. Seeing them answers the question the
    model cannot: is this a recurring visual pattern (a template ImgFlip lacks) or an
    unrelated photo? ``n_members`` matters as much as the pictures: a cluster of 400 is a
    real recurring format, a cluster of 11 is probably incidental.

    ``member_scores`` ranks members within a cluster, higher first (cosine to the cluster
    centroid, or negative Hamming distance to its representative). Without it, members are
    taken in corpus order.
    """
    labels = np.asarray(cluster_labels)
    paths = discovery_df["image_path"].to_numpy()
    sources = discovery_df["source"].to_numpy() if "source" in discovery_df.columns else np.full(len(labels), "")
    scores = None if member_scores is None else np.asarray(member_scores, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.searchsorted(sorted_labels, np.unique(sorted_labels), side="left")
    ends = np.searchsorted(sorted_labels, np.unique(sorted_labels), side="right")

    for cluster_id, start, end in zip(np.unique(sorted_labels), starts, ends):
        if int(cluster_id) < 0:          # HDBSCAN/DBSCAN noise, not a cluster
            continue
        members = order[start:end]
        if scores is not None:
            members = members[np.argsort(-scores[members], kind="stable")]
        pick = members[:n_examples]
        member_sources = sources[members]
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "template": cluster_to_template.get(int(cluster_id), ""),
                "n_members": int(len(members)),
                "n_labeled_members": int((member_sources == "imgflip_train").sum()),
                "example_paths": "|".join(str(p) for p in paths[pick]),
            }
        )

    df = pd.DataFrame(rows)
    path = run_dir / CLUSTER_EXAMPLES_FILE
    df.to_csv(path, index=False)
    named = int((df["template"] != "").sum())
    summary = {
        "clusters": int(len(df)),
        "named_clusters": named,
        "unnamed_clusters": int(len(df) - named),
        "examples_per_cluster": int(n_examples),
        "file": str(path),
    }
    print(f"cluster_examples={len(df)} named={named} unnamed={len(df) - named}")
    print(f"cluster_examples_saved={path}")
    return path, summary


def cosine_to_own_centroid(
    x: np.ndarray, labels: np.ndarray, centroids: np.ndarray, centroid_ids: np.ndarray
) -> np.ndarray:
    """Similarity of every row to the centroid of its own cluster; -inf for noise points."""
    lookup = {int(cid): i for i, cid in enumerate(np.asarray(centroid_ids))}
    out = np.full(len(labels), -np.inf, dtype=np.float64)
    for position, label in enumerate(np.asarray(labels)):
        index = lookup.get(int(label))
        if index is not None:
            out[position] = float(np.dot(x[position], centroids[index]))
    return out


# --------------------------------------------------------------------------------------
# Template-Free by confidence threshold (RQ2)
# --------------------------------------------------------------------------------------


def add_template_confidence_args(parser: argparse.ArgumentParser) -> None:
    """CLI for turning low confidence into an explicit ``Template-Free`` answer.

    A closed-set classifier always names *some* template, so on open-set data it cannot
    say "this matches nothing I know". Its own confidence carries that information: on
    ImgFlip validation, where every image genuinely is templated, the score is high, and
    an open-set image scoring far below that range is one the model would not stand
    behind. The cut-off is calibrated on validation rather than guessed.
    """
    parser.add_argument(
        "--template-confidence-percentile",
        type=float,
        default=None,
        help="Calibrate a Template-Free cut-off at this percentile of the model's confidence on "
             "*templated* validation images, then apply it to --open-set-dir: a template prediction "
             "scoring below it becomes Template-Free. 5 keeps 95%% of genuinely templated validation "
             "images as template predictions. Omit to disable (the model then only ever answers with "
             "a template or, where trained, Non-Meme).",
    )
    parser.add_argument(
        "--template-confidence-threshold",
        type=float,
        default=None,
        help="Use this fixed cut-off instead of calibrating one. Overrides the percentile.",
    )


def calibrate_template_confidence(
    confidences: np.ndarray,
    true_labels: np.ndarray,
    percentile: float | None,
    fixed_threshold: float | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Cut-off below which a template prediction is not trusted, plus a report.

    Only genuinely templated validation rows inform the threshold; ``Non-Meme`` rows are
    excluded, since the question is "how confident is this model when the answer really is
    a template", not "how confident is it in general".
    """
    if fixed_threshold is not None:
        return float(fixed_threshold), {"threshold": float(fixed_threshold), "source": "fixed"}
    if percentile is None:
        return None, {}

    templated = np.asarray(true_labels) != OPEN_SET_NON_MEME
    scores = np.asarray(confidences, dtype=np.float64)[templated]
    if scores.size == 0:
        raise ValueError("No templated validation rows available to calibrate a confidence threshold.")
    threshold = float(np.percentile(scores, percentile))
    report = {
        "threshold": threshold,
        "source": "calibrated_on_validation",
        "percentile": float(percentile),
        "validation_templated_images": int(scores.size),
        "validation_confidence": {
            "min": float(scores.min()),
            "p05": float(np.percentile(scores, 5)),
            "median": float(np.median(scores)),
            "max": float(scores.max()),
        },
    }
    print(f"template_confidence_threshold={threshold:.6f} "
          f"(p{percentile:g} of {scores.size} templated val images; "
          f"median {np.median(scores):.4f})")
    return threshold, report


def apply_template_confidence(
    labels: np.ndarray,
    confidences: np.ndarray,
    threshold: float | None,
    reject_label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Demote under-confident *template* predictions to ``reject_label``.

    ``Non-Meme`` answers are left alone: that is a class the model was trained on, so a
    low score there means "an unconvincing non-meme", not "none of the above".
    """
    labels = np.asarray(labels, dtype=object)
    if threshold is None:
        return labels, {}
    scores = np.asarray(confidences, dtype=np.float64)
    demote = (scores < threshold) & (labels != OPEN_SET_NON_MEME)
    out = np.where(demote, reject_label, labels)
    summary = {
        "threshold": float(threshold),
        "demoted_to_template_free": int(demote.sum()),
        "kept_as_template": int((~demote & (labels != OPEN_SET_NON_MEME)).sum()),
    }
    print(f"template_confidence_applied threshold={threshold:.4f} "
          f"demoted_to_{reject_label}={int(demote.sum())}")
    return out, summary
