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
from sklearn.linear_model import LassoLars
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from meme_research_eval_utils import (
    add_non_meme_args,
    add_non_meme_class,
    collect_non_meme_rows,
    OPEN_SET_TEMPLATE_FREE,
    add_open_set_args,
    collect_open_set_rows,
    score_open_set_predictions,
    write_open_set_predictions,
    compute_metrics,
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


TRAIN_MATRIX: np.ndarray | None = None
TRAIN_LABELS: np.ndarray | None = None
SCALER: StandardScaler | None = None
ALPHA: float = 0.35
MAX_ITER: int = 1000
IDX_TO_LABEL: dict[int, str] = {}
TARGET_SIZE: tuple[int, int] = (64, 64)


@dataclass
class RunConfig:
    dataset_root: Path | None
    parquet_path: Path | None
    train_parquet: Path | None
    test_parquet: Path | None
    image_root: Path | None
    output_dir: Path
    train_size: float = 0.80
    min_images_per_class: int = 7
    random_seed: int = 42
    max_refs_per_template: int = 30
    max_test_images: int | None = None
    target_size: int = 64
    alpha: float = 0.35
    max_iter: int = 1000
    num_workers: int = 1
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


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Closed-set adaptation of the meme-research sparse matching baseline. "
            "This rebuilds the sampled reference bank from the ImgFlip train split and predicts the held-out test split."
        )
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--parquet-path", default=None)
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--test-parquet", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="SEED/runs/meme_research_sparse_matching_eval")
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--min-images-per-class", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-refs-per-template", type=int, default=30)
    parser.add_argument("--max-test-images", type=int, default=None)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--max-images-per-template", type=int, default=None)
    parser.add_argument("--path-prefix-from", default=None)
    parser.add_argument("--path-prefix-to", default=None)
    parser.add_argument("--seeds", type=parse_int_list, default=None)
    add_non_meme_args(parser)
    add_open_set_args(parser)
    args = parser.parse_args()
    return RunConfig(
        dataset_root=None if args.dataset_root is None else Path(args.dataset_root).expanduser().resolve(),
        parquet_path=None if args.parquet_path is None else Path(args.parquet_path).expanduser().resolve(),
        train_parquet=None if args.train_parquet is None else Path(args.train_parquet).expanduser().resolve(),
        test_parquet=None if args.test_parquet is None else Path(args.test_parquet).expanduser().resolve(),
        image_root=None if args.image_root is None else Path(args.image_root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        train_size=args.train_size,
        min_images_per_class=args.min_images_per_class,
        random_seed=args.random_seed,
        max_refs_per_template=args.max_refs_per_template,
        max_test_images=args.max_test_images,
        target_size=args.target_size,
        alpha=args.alpha,
        max_iter=args.max_iter,
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


def read_gray_image(path: str, target_size: tuple[int, int]) -> np.ndarray:
    image = load_rgb_image(path)
    if image is None:
        raise ValueError(f"Unreadable image slipped through filtering: {path}")
    gray = image.convert("L").resize(target_size, resample=Image.Resampling.LANCZOS)
    return np.asarray(gray, dtype=np.uint8)


def stack_images(images: list[np.ndarray]) -> np.ndarray:
    flattened = [img.reshape(-1, 1) for img in images]
    return np.concatenate(flattened, axis=1)


def delta_vectorized(coeffs: np.ndarray, class_nums: np.ndarray) -> np.ndarray:
    n, m = len(coeffs), len(class_nums)
    if n != m:
        raise ValueError(f"Coefficient count {n} does not match label count {m}")
    k = int(np.max(class_nums)) + 1
    mask = np.subtract(np.multiply(np.ones((n, k)), np.arange(k)), class_nums[:, np.newaxis])
    mask = np.where(mask == 0, 1, 0)
    return mask * coeffs[:, np.newaxis]


def residual_vectorized(y: np.ndarray, dictionary: np.ndarray, coeffs: np.ndarray, class_nums: np.ndarray) -> np.ndarray:
    delta_matrix = delta_vectorized(coeffs, class_nums)
    predictions = np.dot(dictionary, delta_matrix)
    errors = predictions - y.reshape(-1, 1)
    return np.linalg.norm(errors, axis=0)


def setup_globals(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    scaler: StandardScaler,
    idx_to_label: dict[int, str],
    alpha: float,
    max_iter: int,
    target_size: tuple[int, int],
) -> None:
    global TRAIN_MATRIX, TRAIN_LABELS, SCALER, IDX_TO_LABEL, ALPHA, MAX_ITER, TARGET_SIZE
    TRAIN_MATRIX = train_matrix
    TRAIN_LABELS = train_labels
    SCALER = scaler
    IDX_TO_LABEL = idx_to_label
    ALPHA = alpha
    MAX_ITER = max_iter
    TARGET_SIZE = target_size


def predict_one(path: str) -> tuple[str, str]:
    if TRAIN_MATRIX is None or TRAIN_LABELS is None or SCALER is None:
        raise RuntimeError("Sparse matching globals were not initialized.")
    image = read_gray_image(path, TARGET_SIZE)
    stacked = image.reshape(-1, 1)
    scaled = SCALER.transform(stacked.T).T
    clf = LassoLars(alpha=ALPHA, max_iter=MAX_ITER)
    clf.fit(TRAIN_MATRIX, scaled)
    coeffs = clf.coef_
    pred_idx = int(np.argmin(residual_vectorized(scaled, TRAIN_MATRIX, coeffs, TRAIN_LABELS)))
    return path, IDX_TO_LABEL[pred_idx]


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
    reference_df = (
        train_df.groupby("template", group_keys=False)
        .head(cfg.max_refs_per_template)
        .reset_index(drop=True)
    )
    if cfg.max_test_images is not None:
        test_df = (
            test_df.groupby("template", group_keys=False)
            .head(max(1, cfg.max_test_images // max(test_df["template"].nunique(), 1)))
            .reset_index(drop=True)
        )
        if len(test_df) > cfg.max_test_images:
            test_df = test_df.iloc[:cfg.max_test_images].reset_index(drop=True)

    labels = sorted(reference_df["template"].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    reference_images = [
        read_gray_image(path, (cfg.target_size, cfg.target_size))
        for path in tqdm(reference_df["image_path"].tolist(), desc="load reference images")
    ]
    reference_matrix = stack_images(reference_images)
    scaler = StandardScaler()
    scaler.fit(reference_matrix.T)
    train_matrix = scaler.transform(reference_matrix.T).T.astype(np.float32)
    train_labels = reference_df["template"].map(label_to_idx).to_numpy(dtype=np.int64)

    setup_globals(
        train_matrix=train_matrix,
        train_labels=train_labels,
        scaler=scaler,
        idx_to_label=idx_to_label,
        alpha=cfg.alpha,
        max_iter=cfg.max_iter,
        target_size=(cfg.target_size, cfg.target_size),
    )

    test_paths = test_df["image_path"].tolist()
    results: list[tuple[str, str]] = []
    if cfg.num_workers > 1:
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
            for item in tqdm(executor.map(predict_one, test_paths), total=len(test_paths), desc="predict sparse matching"):
                results.append(item)
    else:
        for path in tqdm(test_paths, desc="predict sparse matching"):
            results.append(predict_one(path))

    path_to_pred = dict(results)
    predictions_df = test_df.copy()
    predictions_df["pred_template"] = predictions_df["image_path"].map(path_to_pred)
    predictions_df["correct"] = predictions_df["pred_template"].to_numpy() == predictions_df["template"].to_numpy()
    metrics = compute_metrics(
        predictions_df["template"].to_numpy(),
        predictions_df["pred_template"].to_numpy(),
    )

    predictions_df = predictions_df.loc[:, ["image_path", "template", "pred_template", "correct", "source"]]
    predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)

    open_set_summary = None
    if cfg.open_set_dir is not None:
        open_df = collect_open_set_rows(cfg.open_set_dir)
        open_paths = open_df["image_path"].tolist()
        open_results: list[tuple[str, str]] = []
        if cfg.num_workers > 1:
            with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
                for item in tqdm(executor.map(predict_one, open_paths), total=len(open_paths), desc="predict open-set"):
                    open_results.append(item)
        else:
            for path in tqdm(open_paths, desc="predict open-set"):
                open_results.append(predict_one(path))
        # Sparse reconstruction always picks the class with the smallest residual: no reject path.
        open_pred = dict(open_results)
        open_labels = np.array([open_pred[path] for path in open_paths], dtype=object)
        _, open_set_summary = write_open_set_predictions(
            run_dir, open_df, open_labels
        )

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
            "test_images": int(len(test_df)),
            "reference_images": int(len(reference_df)),
            "small_class_filtering": filtering_summary,
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
    }
    if open_set_summary is not None and cfg.open_set_labels is not None:
        open_set_summary["metrics"] = score_open_set_predictions(run_dir, cfg.open_set_labels)
    metadata["open_set"] = open_set_summary
    timing = finalize_run_timing(metadata, summary_row, run_started_at, start_perf)
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))
    (run_dir / "test_metrics.json").write_text(json.dumps({"test_metrics": metrics, "timing": timing}, indent=2))
    pd.DataFrame([summary_row]).to_csv(run_dir / "metrics_summary.csv", index=False)

    print(f"reference_images={len(reference_df)}")
    print(f"test_accuracy={metrics['accuracy']:.4f}")
    print(f"test_f1={metrics['f1']:.4f}")
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
