#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier
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
    compute_metrics,
    compute_metrics_with_rejection,
    drop_small_classes,
    filter_valid_image_rows,
    finalize_run_timing,
    load_dataset_rows,
    load_rgb_image,
    load_split_rows,
    now_iso,
    resolve_seed_paths,
    set_seed,
    stratified_split,
    summarize_batch_timings,
)


REJECT_TOKEN = "__OUTLIER__"


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
    radius_values: tuple[int, ...] = (6, 8, 10, 12, 14, 16, 20, 24, 32)
    weights: tuple[str, ...] = ("uniform", "distance")
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


def parse_int_list(text: str) -> tuple[int, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer radius.")
    return tuple(int(value) for value in values)


def parse_seed_list(text: str) -> tuple[int, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer seed.")
    return tuple(int(value) for value in values)


def parse_str_list(text: str) -> tuple[str, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one weight string.")
    return tuple(values)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Closed-set adaptation of the meme-research RNN+pHash notebooks. "
            "The original notebooks use RadiusNeighborsClassifier over pHash vectors; "
            "this script ports that logic to a remote-friendly CLI."
        )
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--parquet-path", default=None)
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--test-parquet", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="SEED/runs/meme_research_rnn_phash_eval")
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--min-images-per-class", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--radius-values", type=parse_int_list, default=(6, 8, 10, 12, 14, 16, 20, 24, 32))
    parser.add_argument("--weights", type=parse_str_list, default=("uniform", "distance"))
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--max-images-per-template", type=int, default=None)
    parser.add_argument("--path-prefix-from", default=None)
    parser.add_argument("--path-prefix-to", default=None)
    parser.add_argument("--seeds", type=parse_seed_list, default=None)
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
        val_size=args.val_size,
        min_images_per_class=args.min_images_per_class,
        random_seed=args.random_seed,
        radius_values=args.radius_values,
        weights=args.weights,
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


def compute_phash_bits(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    img_size = hash_size * highfreq_factor
    gray = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    dct_rows = dct(pixels, axis=0, norm="ortho")
    dct_2d = dct(dct_rows, axis=1, norm="ortho")
    low_freq = dct_2d[:hash_size, :hash_size]
    flat = low_freq.reshape(-1)
    median = np.median(flat[1:]) if flat.size > 1 else np.median(flat)
    return (low_freq > median).astype(np.uint8).reshape(-1)


def hex_to_hash(hexstr: str, hash_size: int = 8) -> np.ndarray:
    count = hash_size * (hash_size // 4)
    if len(hexstr) != count:
        raise ValueError(f"Expected hex string size of {count}, got {len(hexstr)}")
    rows = []
    for idx in range(count // 2):
        byte = int(hexstr[idx * 2: idx * 2 + 2], 16)
        rows.extend([1 if byte & 2 ** bit else 0 for bit in range(8)])
    return np.asarray(rows, dtype=np.uint8)


def encode_phashes(df: pd.DataFrame) -> np.ndarray:
    if "phash" in df.columns and df["phash"].notna().all():
        vectors = [hex_to_hash(str(value)) for value in df["phash"].tolist()]
        return np.vstack(vectors).astype(np.float32)

    vectors: list[np.ndarray] = []
    for path in tqdm(df["image_path"].tolist(), desc="compute phash"):
        image = load_rgb_image(path)
        if image is None:
            raise ValueError(f"Unreadable image slipped through filtering: {path}")
        vectors.append(compute_phash_bits(image))
    return np.vstack(vectors).astype(np.float32)


def fit_and_predict(
    train_vectors: np.ndarray,
    train_labels: np.ndarray,
    test_vectors: np.ndarray,
    radius: int,
    weights: str,
) -> np.ndarray:
    clf = RadiusNeighborsClassifier(
        radius=radius,
        weights=weights,
        metric="manhattan",
        outlier_label=-1,
    )
    clf.fit(train_vectors, train_labels)
    return clf.predict(test_vectors)


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
    tune_train_df = tune_train_df.reset_index(drop=True)
    tune_val_df = tune_val_df.reset_index(drop=True)

    labels = sorted(train_df["template"].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    tune_train_vectors = encode_phashes(tune_train_df)
    tune_val_vectors = encode_phashes(tune_val_df)
    train_vectors = encode_phashes(train_df)
    test_vectors = encode_phashes(test_df)

    tune_train_y = tune_train_df["template"].map(label_to_idx).to_numpy(dtype=np.int64)
    tune_val_y = tune_val_df["template"].map(label_to_idx).to_numpy(dtype=np.int64)
    train_y = train_df["template"].map(label_to_idx).to_numpy(dtype=np.int64)

    search_rows: list[dict[str, float | int | str]] = []
    best_score = float("-inf")
    best_params: tuple[int, str] | None = None

    for radius in cfg.radius_values:
        for weights in cfg.weights:
            val_preds = fit_and_predict(
                tune_train_vectors,
                tune_train_y,
                tune_val_vectors,
                radius=radius,
                weights=weights,
            )
            val_pred_labels = decode_predictions(val_preds, idx_to_label)
            val_true_labels = tune_val_df["template"].to_numpy()
            metrics = compute_metrics_with_rejection(val_true_labels, val_pred_labels, REJECT_TOKEN)
            row = {
                "radius": int(radius),
                "weights": str(weights),
                "val_accuracy": float(metrics["accuracy"]),
                "val_f1": float(metrics["f1"]),
                "val_coverage": float(metrics["coverage"]),
                "val_covered_accuracy": float(metrics["covered_accuracy"]),
            }
            search_rows.append(row)
            score = float(metrics["f1"])
            print(row)
            if score > best_score:
                best_score = score
                best_params = (radius, weights)

    if best_params is None:
        raise RuntimeError("Hyperparameter search failed to produce a candidate.")

    best_radius, best_weights = best_params
    final_preds = fit_and_predict(
        train_vectors,
        train_y,
        test_vectors,
        radius=best_radius,
        weights=best_weights,
    )
    test_pred_labels = decode_predictions(final_preds, idx_to_label)
    test_true_labels = test_df["template"].to_numpy()
    metrics = compute_metrics_with_rejection(test_true_labels, test_pred_labels, REJECT_TOKEN)

    predictions_df = test_df.copy()
    predictions_df["pred_template"] = test_pred_labels
    predictions_df["is_rejected"] = predictions_df["pred_template"].eq(REJECT_TOKEN)
    predictions_df["correct"] = predictions_df["pred_template"].to_numpy() == predictions_df["template"].to_numpy()
    predictions_df = predictions_df.loc[:, ["image_path", "template", "pred_template", "is_rejected", "correct", "source"]]
    predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)

    open_set_summary = None
    if cfg.open_set_dir is not None:
        open_df = collect_open_set_rows(cfg.open_set_dir)
        open_vectors = encode_phashes(open_df)
        open_preds = fit_and_predict(train_vectors, train_y, open_vectors, radius=best_radius, weights=best_weights)
        # No training neighbour within the chosen radius is the model's "not a known template".
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
            "best_radius": int(best_radius),
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

    print(f"best_radius={best_radius} best_weights={best_weights}")
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
