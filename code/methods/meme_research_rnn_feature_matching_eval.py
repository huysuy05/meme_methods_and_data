#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from skimage.feature import ORB

# OpenCV parallelises internally by default, which oversubscribes the CPU once the
# ThreadPoolExecutor below is also running. Let the executor own the parallelism.
cv2.setNumThreads(1)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from meme_research_eval_utils import (
    add_non_meme_args,
    add_non_meme_class,
    collect_non_meme_rows,
    OPEN_SET_TEMPLATE_FREE,
    add_open_set_args,
    collect_open_set_rows,
    open_set_labels,
    score_open_set_predictions,
    write_open_set_predictions,
    compute_metrics_with_rejection,
    drop_small_classes,
    filter_valid_image_rows,
    finalize_run_timing,
    load_dataset_rows,
    load_rgb_image,
    load_split_rows,
    now_iso,
    parse_int_list,
    resolve_seed_paths,
    set_seed,
    stratified_split,
    summarize_batch_timings,
)


REJECT_TOKEN = "__OUTLIER__"

# cv2 does not export the LSH index constant, and LSH is the only FLANN index that
# accepts the packed binary descriptors ORB produces.
FLANN_INDEX_LSH = 6


@dataclass
class RunConfig:
    dataset_root: Path | None
    parquet_path: Path | None
    train_parquet: Path | None
    test_parquet: Path | None
    image_root: Path | None
    output_dir: Path
    train_size: float = 0.80
    val_size: float = 0.10
    min_images_per_class: int = 7
    random_seed: int = 42
    max_refs_per_template: int = 20
    max_test_images: int | None = None
    target_size: int = 256
    n_keypoints: int = 512
    fast_threshold: float = 0.08
    feature_backend: str = "opencv"
    # d: two keypoints count as shared when their descriptors are within this Hamming distance.
    max_match_distance: float = 27.0
    # m: two images are a match when they share at least this many keypoints within d.
    min_match_values: tuple[int, ...] = (20,)
    flann_table_number: int = 6
    flann_key_size: int = 12
    flann_multi_probe_level: int = 1
    flann_checks: int = 50
    radius_values: tuple[float, ...] = (0.90, 0.93, 0.95, 0.97, 0.99)
    weights: tuple[str, ...] = ("uniform", "distance")
    num_workers: int = 4
    max_templates: int | None = None
    max_images_per_template: int | None = None
    path_prefix_from: str | None = None
    path_prefix_to: str | None = None
    seeds: tuple[int, ...] | None = None
    open_set_dir: Path | None = None
    open_set_reject_label: str = OPEN_SET_TEMPLATE_FREE
    open_set_labels: Path | None = None
    non_meme_root: Path | None = None
    max_non_meme_images: int | None = None


def parse_float_list(text: str) -> tuple[float, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float radius.")
    return tuple(float(value) for value in values)


def parse_str_list(text: str) -> tuple[str, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one weight string.")
    return tuple(values)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Closed-set ImgFlip evaluation for a standalone feature-matching rNN baseline (rNN-FM). "
            "Images are represented by ORB descriptors and matched with FLANN: two images are a match "
            "when they share at least m keypoints within Hamming distance d. The resulting match counts "
            "form a distance matrix that is classified by radius-neighbor voting."
        )
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--parquet-path", default=None)
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--test-parquet", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="SEED/runs/meme_research_rnn_feature_matching_eval")
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--min-images-per-class", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-refs-per-template", type=int, default=20)
    parser.add_argument("--max-test-images", type=int, default=None)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--n-keypoints", type=int, default=512)
    parser.add_argument("--fast-threshold", type=float, default=0.08)
    parser.add_argument(
        "--feature-backend",
        choices=["opencv", "skimage"],
        default="opencv",
        help="ORB implementation. opencv is far faster; skimage reproduces earlier runs. "
             "Results from the two backends are NOT comparable.",
    )
    parser.add_argument(
        "--max-match-distance",
        type=float,
        default=27.0,
        help="d: maximum Hamming distance (0-256) for two ORB descriptors to count as a shared keypoint.",
    )
    parser.add_argument(
        "--min-match-values",
        type=parse_int_list,
        default=(20,),
        help="m: minimum shared keypoints for two images to be a match. Pass several to tune m on the val split.",
    )
    parser.add_argument("--flann-table-number", type=int, default=6)
    parser.add_argument("--flann-key-size", type=int, default=12)
    parser.add_argument("--flann-multi-probe-level", type=int, default=1)
    parser.add_argument("--flann-checks", type=int, default=50)
    parser.add_argument("--radius-values", type=parse_float_list, default=(0.90, 0.93, 0.95, 0.97, 0.99))
    parser.add_argument("--weights", type=parse_str_list, default=("uniform", "distance"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--max-images-per-template", type=int, default=None)
    parser.add_argument("--path-prefix-from", default=None)
    parser.add_argument("--path-prefix-to", default=None)
    parser.add_argument("--seeds", type=parse_int_list, default=None)
    add_non_meme_args(parser)
    add_open_set_args(parser)
    args = parser.parse_args()
    if any(radius >= 1.0 for radius in args.radius_values):
        # Non-matching pairs sit at exactly 1.0, so a radius of 1.0 would make every
        # reference a neighbour and silently discard the m threshold.
        raise SystemExit("--radius-values must all be < 1.0; 1.0 admits non-matching pairs as neighbours.")
    return RunConfig(
        dataset_root=None if args.dataset_root is None else Path(args.dataset_root).expanduser().resolve(),
        parquet_path=None if args.parquet_path is None else Path(args.parquet_path).expanduser().resolve(),
        train_parquet=None if args.train_parquet is None else Path(args.train_parquet).expanduser().resolve(),
        test_parquet=None if args.test_parquet is None else Path(args.test_parquet).expanduser().resolve(),
        image_root=None if args.image_root is None else Path(args.image_root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        train_size=args.train_size,
        val_size=args.val_size,
        min_images_per_class=args.min_images_per_class,
        random_seed=args.random_seed,
        max_refs_per_template=args.max_refs_per_template,
        max_test_images=args.max_test_images,
        target_size=args.target_size,
        n_keypoints=args.n_keypoints,
        fast_threshold=args.fast_threshold,
        feature_backend=args.feature_backend,
        max_match_distance=args.max_match_distance,
        min_match_values=tuple(int(value) for value in args.min_match_values),
        flann_table_number=args.flann_table_number,
        flann_key_size=args.flann_key_size,
        flann_multi_probe_level=args.flann_multi_probe_level,
        flann_checks=args.flann_checks,
        radius_values=args.radius_values,
        weights=args.weights,
        num_workers=args.num_workers,
        max_templates=args.max_templates,
        max_images_per_template=args.max_images_per_template,
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
        seeds=args.seeds,
        open_set_dir=None if args.open_set_dir is None else Path(args.open_set_dir).expanduser().resolve(),
        open_set_reject_label=args.open_set_reject_label,
        open_set_labels=None if args.open_set_labels is None else Path(args.open_set_labels).expanduser().resolve(),
        non_meme_root=None if args.non_meme_root is None else Path(args.non_meme_root).expanduser().resolve(),
        max_non_meme_images=args.max_non_meme_images,
    )


def extract_descriptors(
    path: str,
    target_size: int,
    n_keypoints: int,
    fast_threshold: float,
    backend: str = "opencv",
) -> np.ndarray | None:
    """ORB descriptors for one image, as packed uint8 (n, 32).

    Both backends return 256-bit descriptors, but scikit-image hands them back as
    unpacked booleans (n, 256); we pack those so FLANN's LSH index accepts them and so
    the Hamming threshold d is on the same 0-256 scale either way. Keypoint selection
    still differs between the backends, so their numbers are not comparable and a run
    records which one produced it.
    """
    image = load_rgb_image(path)
    if image is None:
        raise ValueError(f"Unreadable image slipped through filtering: {path}")
    gray = image.convert("L").resize((target_size, target_size), resample=Image.Resampling.LANCZOS)

    if backend == "opencv":
        arr = np.asarray(gray, dtype=np.uint8)
        # skimage's fast_threshold is on a [0, 1] image; OpenCV's is on the 0-255 scale.
        orb = cv2.ORB_create(
            nfeatures=n_keypoints,
            fastThreshold=max(1, int(round(fast_threshold * 255.0))),
        )
        _, descriptors = orb.detectAndCompute(arr, None)
        if descriptors is None or descriptors.size == 0:
            return None
        return np.ascontiguousarray(descriptors, dtype=np.uint8)

    arr = np.asarray(gray, dtype=np.float32) / 255.0
    orb = ORB(n_keypoints=n_keypoints, fast_threshold=fast_threshold)
    try:
        orb.detect_and_extract(arr)
        descriptors = orb.descriptors
    except RuntimeError:
        descriptors = None
    if descriptors is None or descriptors.size == 0:
        return None
    return np.ascontiguousarray(np.packbits(descriptors.astype(bool), axis=1), dtype=np.uint8)


def build_flann_matcher(cfg: RunConfig) -> cv2.FlannBasedMatcher:
    """A FLANN matcher over binary ORB descriptors.

    Replaces the brute-force matcher of Courtois and Frissen (2023): LSH gives approximate
    nearest neighbours, so a pair is matched in time sublinear in the descriptor count at
    the cost of occasionally missing a true neighbour.
    """
    index_params = {
        "algorithm": FLANN_INDEX_LSH,
        "table_number": cfg.flann_table_number,
        "key_size": cfg.flann_key_size,
        "multi_probe_level": cfg.flann_multi_probe_level,
    }
    return cv2.FlannBasedMatcher(index_params, {"checks": cfg.flann_checks})


def count_shared_keypoints(
    matcher: cv2.FlannBasedMatcher,
    query_desc: np.ndarray | None,
    max_match_distance: float,
) -> int:
    """How many query keypoints have a neighbour in the trained image within distance d."""
    if query_desc is None or len(query_desc) == 0:
        return 0
    try:
        knn = matcher.knnMatch(query_desc, k=1)
    except cv2.error:
        # LSH refuses degenerate indices (e.g. too few descriptors to fill a hash table).
        return 0
    return sum(
        1
        for candidates in knn
        if candidates and candidates[0].distance <= max_match_distance
    )


def extract_many_descriptors(paths: list[str], cfg: RunConfig, desc: str) -> list[np.ndarray | None]:
    """Extract descriptors for many images, in parallel but preserving input order."""
    workers = max(1, cfg.num_workers)
    if workers == 1:
        return [
            extract_descriptors(path, cfg.target_size, cfg.n_keypoints, cfg.fast_threshold, cfg.feature_backend)
            for path in tqdm(paths, desc=desc)
        ]

    def work(path: str) -> np.ndarray | None:
        return extract_descriptors(
            path, cfg.target_size, cfg.n_keypoints, cfg.fast_threshold, cfg.feature_backend
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(work, paths), total=len(paths), desc=desc))


def compute_match_counts(
    query_descs: list[np.ndarray | None],
    ref_descs: list[np.ndarray | None],
    cfg: RunConfig,
    desc: str,
) -> np.ndarray:
    """Shared-keypoint counts for every (query, reference) pair.

    Iterates over references rather than queries so each reference's LSH index is built
    once and reused for every query, which is where FLANN's speed advantage comes from.
    Counts are stored rather than distances so the match threshold m can be tuned on the
    val split without re-running the matching.
    """
    counts = np.zeros((len(query_descs), len(ref_descs)), dtype=np.uint16)

    def process_ref(ref_idx: int) -> tuple[int, np.ndarray]:
        column = np.zeros(len(query_descs), dtype=np.uint16)
        ref_desc = ref_descs[ref_idx]
        if ref_desc is None or len(ref_desc) == 0:
            return ref_idx, column
        matcher = build_flann_matcher(cfg)
        matcher.add([ref_desc])
        try:
            matcher.train()
        except cv2.error:
            return ref_idx, column
        for query_idx, query_desc in enumerate(query_descs):
            column[query_idx] = count_shared_keypoints(matcher, query_desc, cfg.max_match_distance)
        return ref_idx, column

    workers = max(1, cfg.num_workers)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for ref_idx, column in tqdm(
                executor.map(process_ref, range(len(ref_descs))),
                total=len(ref_descs),
                desc=desc,
            ):
                counts[:, ref_idx] = column
    else:
        for ref_idx in tqdm(range(len(ref_descs)), desc=desc):
            _, column = process_ref(ref_idx)
            counts[:, ref_idx] = column
    return counts


def distances_from_match_counts(counts: np.ndarray, min_matches: int, n_keypoints: int) -> np.ndarray:
    """Turn shared-keypoint counts into the distance matrix the rNN step consumes.

    Pairs below m keypoints are not matches and sit at the maximum distance of 1.0. Matched
    pairs are graded by how many keypoints they share, so rNN's radius can be stricter than
    the bare m threshold; at the widest radius the matrix behaves as the plain binary
    match/no-match graph.
    """
    scale = float(max(n_keypoints, 1))
    graded = 1.0 - np.minimum(counts.astype(np.float32) / scale, 1.0)
    return np.where(counts >= min_matches, graded, 1.0).astype(np.float32)


def predict_from_distance_matrix(
    distance_matrix: np.ndarray,
    train_labels: np.ndarray,
    radius: float,
    weights: str,
) -> np.ndarray:
    preds = np.full(distance_matrix.shape[0], -1, dtype=np.int64)
    num_classes = int(train_labels.max()) + 1
    for row_idx, row in enumerate(distance_matrix):
        neighbor_idx = np.where(row <= radius)[0]
        if neighbor_idx.size == 0:
            continue
        labels = train_labels[neighbor_idx]
        if weights == "uniform":
            scores = np.bincount(labels, minlength=num_classes).astype(np.float32)
        elif weights == "distance":
            inv = 1.0 / np.clip(row[neighbor_idx], 1e-6, None)
            scores = np.bincount(labels, weights=inv, minlength=num_classes).astype(np.float32)
        else:
            raise ValueError(f"Unsupported weights={weights}")
        preds[row_idx] = int(scores.argmax())
    return preds


def decode_predictions(preds: np.ndarray, idx_to_label: dict[int, str]) -> np.ndarray:
    return np.array(
        [idx_to_label[int(pred)] if int(pred) >= 0 else REJECT_TOKEN for pred in preds],
        dtype=object,
    )


def run_once(cfg: RunConfig, run_dir: Path) -> dict[str, float | int | str]:
    set_seed(cfg.random_seed)
    run_started_at = now_iso()
    start_perf = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir={run_dir}")

    if cfg.train_parquet is not None and cfg.test_parquet is not None:
        train_df, test_df = load_split_rows(
            str(cfg.train_parquet),
            str(cfg.test_parquet),
            image_root=None if cfg.image_root is None else str(cfg.image_root),
            path_prefix_from=cfg.path_prefix_from,
            path_prefix_to=cfg.path_prefix_to,
        )
        df = pd.concat([train_df, test_df], ignore_index=True)
        filtering_summary = {"precomputed_split": True}
    else:
        df = load_dataset_rows(
            dataset_root=None if cfg.dataset_root is None else str(cfg.dataset_root),
            parquet_path=None if cfg.parquet_path is None else str(cfg.parquet_path),
            image_root=None if cfg.image_root is None else str(cfg.image_root),
            path_prefix_from=cfg.path_prefix_from,
            path_prefix_to=cfg.path_prefix_to,
            max_templates=cfg.max_templates,
            max_images_per_template=cfg.max_images_per_template,
        )
        df = filter_valid_image_rows(df)
        df, filtering_summary = drop_small_classes(df, cfg.min_images_per_class)
        train_df, test_df = stratified_split(df, train_size=cfg.train_size, random_seed=cfg.random_seed)
    non_meme_summary = None
    if cfg.non_meme_root is not None:
        # Non-Meme becomes an ordinary class of this method, learnt with the same
        # mechanism as the templates -- no separate detector is consulted.
        non_meme_df = collect_non_meme_rows(
            cfg.non_meme_root, exclude_dir=cfg.open_set_dir,
            max_images=cfg.max_non_meme_images, random_seed=cfg.random_seed,
        )
        train_df, test_df, non_meme_summary = add_non_meme_class(
            train_df, test_df, non_meme_df, cfg.train_size, cfg.random_seed
        )
    tune_train_df, tune_val_df = train_test_split(
        train_df,
        test_size=cfg.val_size,
        stratify=train_df["template"],
        random_state=cfg.random_seed,
    )
    tune_train_df = (
        tune_train_df.groupby("template", group_keys=False)
        .head(cfg.max_refs_per_template)
        .reset_index(drop=True)
    )
    train_df = (
        train_df.groupby("template", group_keys=False)
        .head(cfg.max_refs_per_template)
        .reset_index(drop=True)
    )
    tune_val_df = tune_val_df.reset_index(drop=True)
    if cfg.max_test_images is not None:
        test_df = test_df.iloc[:cfg.max_test_images].reset_index(drop=True)

    labels = sorted(train_df["template"].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    tune_train_descs = extract_many_descriptors(tune_train_df["image_path"].tolist(), cfg, "extract tune-train descriptors")
    tune_val_descs = extract_many_descriptors(tune_val_df["image_path"].tolist(), cfg, "extract tune-val descriptors")
    train_descs = extract_many_descriptors(train_df["image_path"].tolist(), cfg, "extract train descriptors")
    test_descs = extract_many_descriptors(test_df["image_path"].tolist(), cfg, "extract test descriptors")

    tune_val_counts = compute_match_counts(tune_val_descs, tune_train_descs, cfg, "match tune-val to tune-train")
    test_counts = compute_match_counts(test_descs, train_descs, cfg, "match test to train")

    tune_train_y = tune_train_df["template"].map(label_to_idx).to_numpy(dtype=np.int64)
    train_y = train_df["template"].map(label_to_idx).to_numpy(dtype=np.int64)

    search_rows: list[dict[str, float | str]] = []
    best_score = float("-inf")
    best_params: tuple[int, float, str] | None = None

    for min_matches in cfg.min_match_values:
        tune_val_dist = distances_from_match_counts(tune_val_counts, min_matches, cfg.n_keypoints)
        for radius in cfg.radius_values:
            for weights in cfg.weights:
                val_preds = predict_from_distance_matrix(tune_val_dist, tune_train_y, radius, weights)
                val_pred_labels = decode_predictions(val_preds, idx_to_label)
                metrics = compute_metrics_with_rejection(
                    tune_val_df["template"].to_numpy(),
                    val_pred_labels,
                    REJECT_TOKEN,
                )
                row = {
                    "min_matches": int(min_matches),
                    "max_match_distance": float(cfg.max_match_distance),
                    "radius": float(radius),
                    "weights": str(weights),
                    "val_f1": float(metrics["f1"]),
                    "val_accuracy": float(metrics["accuracy"]),
                    "val_coverage": float(metrics["coverage"]),
                    "val_covered_accuracy": float(metrics["covered_accuracy"]),
                }
                search_rows.append(row)
                print(row)
                if float(metrics["f1"]) > best_score:
                    best_score = float(metrics["f1"])
                    best_params = (int(min_matches), float(radius), str(weights))

    if best_params is None:
        raise RuntimeError("Hyperparameter search failed.")

    best_min_matches, best_radius, best_weights = best_params
    test_dist = distances_from_match_counts(test_counts, best_min_matches, cfg.n_keypoints)
    final_preds = predict_from_distance_matrix(test_dist, train_y, best_radius, best_weights)
    test_pred_labels = decode_predictions(final_preds, idx_to_label)
    metrics = compute_metrics_with_rejection(
        test_df["template"].to_numpy(),
        test_pred_labels,
        REJECT_TOKEN,
    )

    predictions_df = test_df.copy()
    predictions_df["pred_template"] = test_pred_labels
    predictions_df["is_rejected"] = predictions_df["pred_template"].eq(REJECT_TOKEN)
    predictions_df["correct"] = predictions_df["pred_template"].to_numpy() == predictions_df["template"].to_numpy()
    predictions_df = predictions_df.loc[:, ["image_path", "template", "pred_template", "is_rejected", "correct", "source"]]
    predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)

    open_set_summary = None
    if cfg.open_set_dir is not None:
        open_df = collect_open_set_rows(cfg.open_set_dir)
        open_descs = extract_many_descriptors(open_df["image_path"].tolist(), cfg, "extract open-set descriptors")
        open_counts = compute_match_counts(open_descs, train_descs, cfg, "match open-set to train")
        open_dist = distances_from_match_counts(open_counts, best_min_matches, cfg.n_keypoints)
        open_preds = predict_from_distance_matrix(open_dist, train_y, best_radius, best_weights)
        # No reference within the chosen radius is the model's "not a known template".
        open_labels = open_set_labels(decode_predictions(open_preds, idx_to_label), (REJECT_TOKEN,), cfg.open_set_reject_label)
        _, open_set_summary = write_open_set_predictions(
            run_dir, open_df, open_labels
        )
    pd.DataFrame(search_rows).to_csv(run_dir / "val_search.csv", index=False)

    metadata = {
        "config": {
            **asdict(cfg),
            "dataset_root": None if cfg.dataset_root is None else str(cfg.dataset_root),
            "parquet_path": None if cfg.parquet_path is None else str(cfg.parquet_path),
            "train_parquet": None if cfg.train_parquet is None else str(cfg.train_parquet),
            "test_parquet": None if cfg.test_parquet is None else str(cfg.test_parquet),
            "image_root": None if cfg.image_root is None else str(cfg.image_root),
            "output_dir": str(cfg.output_dir),
            "open_set_dir": None if cfg.open_set_dir is None else str(cfg.open_set_dir),
            "open_set_labels": None if cfg.open_set_labels is None else str(cfg.open_set_labels),
            "non_meme_root": None if cfg.non_meme_root is None else str(cfg.non_meme_root),
        },
        "non_meme_class": non_meme_summary,
        "dataset": {
            "images_total_after_filtering": int(len(df)),
            "classes_after_filtering": int(df["template"].nunique()),
            "train_images": int(len(train_df)),
            "val_images": int(len(tune_val_df)),
            "test_images": int(len(test_df)),
            "small_class_filtering": filtering_summary,
        },
        "selection": {
            "best_min_matches": int(best_min_matches),
            "max_match_distance": float(cfg.max_match_distance),
            "best_radius": float(best_radius),
            "best_weights": best_weights,
            "best_val_f1": float(best_score),
        },
        "test_metrics": metrics,
    }
    summary_row = {
        "run_dir": str(run_dir),
        "seed": int(cfg.random_seed),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "mcc": float(metrics["mcc"]),
        "cohen_kappa": float(metrics["cohen_kappa"]),
        "coverage": float(metrics["coverage"]),
        "covered_accuracy": float(metrics["covered_accuracy"]),
    }
    if open_set_summary is not None and cfg.open_set_labels is not None:
        open_set_summary["metrics"] = score_open_set_predictions(run_dir, cfg.open_set_labels)
    metadata["open_set"] = open_set_summary
    timing = finalize_run_timing(metadata, summary_row, run_started_at, start_perf)
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))
    (run_dir / "test_metrics.json").write_text(json.dumps({"test_metrics": metrics, "timing": timing}, indent=2))
    pd.DataFrame([summary_row]).to_csv(run_dir / "metrics_summary.csv", index=False)

    print(
        f"d={cfg.max_match_distance} m={best_min_matches} "
        f"best_radius={best_radius} best_weights={best_weights}"
    )
    print(f"test_accuracy={metrics['accuracy']:.4f}")
    print(f"test_f1={metrics['f1']:.4f}")
    print(f"test_coverage={metrics['coverage']:.4f}")
    print(f"duration_seconds={timing['duration_seconds']:.3f}")
    print(f"predictions_saved={run_dir / 'test_predictions.csv'}")
    return summary_row


def main() -> None:
    cfg = parse_args()
    seed_values = cfg.seeds if cfg.seeds is not None else (cfg.random_seed,)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if len(seed_values) == 1:
        single_cfg = RunConfig(**resolve_seed_paths({**asdict(cfg), "random_seed": int(seed_values[0]), "seeds": None}, seed_values[0]))
        run_once(single_cfg, cfg.output_dir / timestamp)
        return

    batch_dir = cfg.output_dir / f"{timestamp}_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_rows: list[dict[str, float | int | str]] = []
    for seed in seed_values:
        print(f"\n=== Running seed {seed} ===")
        seed_cfg = RunConfig(**resolve_seed_paths({**asdict(cfg), "random_seed": int(seed), "seeds": None}, seed))
        batch_rows.append(run_once(seed_cfg, batch_dir / f"seed_{int(seed)}"))
    batch_timing = summarize_batch_timings(batch_rows)
    pd.DataFrame(batch_rows).to_csv(batch_dir / "batch_metrics_summary.csv", index=False)
    (batch_dir / "batch_config.json").write_text(
        json.dumps(
            {
                "output_dir": str(cfg.output_dir),
                "batch_dir": str(batch_dir),
                "seeds": [int(seed) for seed in seed_values],
                "timing": batch_timing,
            },
            indent=2,
        )
    )
    print(f"average_duration_seconds={batch_timing['average_duration_seconds']:.3f}")
    print(f"\nbatch_summary_saved={batch_dir / 'batch_metrics_summary.csv'}")


if __name__ == "__main__":
    main()
