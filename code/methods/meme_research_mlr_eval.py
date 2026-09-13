#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import feature
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from meme_research_eval_utils import (
    add_non_meme_args,
    add_template_confidence_args,
    apply_template_confidence,
    calibrate_template_confidence,
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
    max_templates: int | None = None
    max_images_per_template: int | None = None
    path_prefix_from: str | None = None
    path_prefix_to: str | None = None
    max_iter: int = 1000
    seeds: tuple[int, ...] | None = None
    open_set_dir: Path | None = None
    open_set_reject_label: str = OPEN_SET_TEMPLATE_FREE
    open_set_labels: Path | None = None
    non_meme_root: Path | None = None
    template_confidence_percentile: float | None = None
    template_confidence_threshold: float | None = None
    max_non_meme_images: int | None = None


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Closed-set adaptation of the meme-research multinomial logistic regression baseline. "
            "Features follow the notebook: HSV histogram + grayscale histogram + LBP."
        )
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--parquet-path", default=None)
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--test-parquet", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="SEED/runs/meme_research_mlr_eval")
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--min-images-per-class", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--max-images-per-template", type=int, default=None)
    parser.add_argument("--path-prefix-from", default=None)
    parser.add_argument("--path-prefix-to", default=None)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seeds", type=parse_int_list, default=None)
    add_non_meme_args(parser)
    add_template_confidence_args(parser)
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
        max_templates=args.max_templates,
        max_images_per_template=args.max_images_per_template,
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
        max_iter=args.max_iter,
        seeds=args.seeds,
        open_set_dir=None if args.open_set_dir is None else Path(args.open_set_dir).expanduser().resolve(),
        open_set_reject_label=args.open_set_reject_label,
        open_set_labels=None if args.open_set_labels is None else Path(args.open_set_labels).expanduser().resolve(),
        non_meme_root=None if args.non_meme_root is None else Path(args.non_meme_root).expanduser().resolve(),
        template_confidence_percentile=args.template_confidence_percentile,
        template_confidence_threshold=args.template_confidence_threshold,
        max_non_meme_images=args.max_non_meme_images,
    )


def extract_features(image_path: str) -> np.ndarray:
    image = load_rgb_image(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at path {image_path}")

    rgb = np.array(image)
    hsv_img = np.array(image.convert("HSV"))
    hsv_features: list[np.ndarray] = []
    for channel in range(3):
        hist, _ = np.histogram(hsv_img[:, :, channel], bins=256, range=(0, 256))
        hsv_features.append(hist.astype(np.float32))

    gray = np.array(image.convert("L"))
    grayscale_hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    grayscale_hist = grayscale_hist.astype(np.float32)

    lbp = feature.local_binary_pattern(gray, P=24, R=8, method="uniform")
    n_points = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_features = lbp_hist.astype(np.float32)
    lbp_features /= (lbp_features.sum() + 1e-7)
    if len(lbp_features) == 27:
        lbp_features = np.append(lbp_features, 0.0)

    return np.hstack([*hsv_features, grayscale_hist, lbp_features]).astype(np.float32)


def encode_features(df: pd.DataFrame) -> np.ndarray:
    return np.vstack([extract_features(path) for path in tqdm(df["image_path"].tolist(), desc="extract features")])


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
    labels = sorted(train_df["template"].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    train_x = encode_features(train_df)
    test_x = encode_features(test_df)
    train_y = train_df["template"].map(label_to_idx).to_numpy(dtype=np.int64)
    test_y = test_df["template"].map(label_to_idx).to_numpy(dtype=np.int64)

    clf = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        random_state=cfg.random_seed,
        n_jobs=-1,
        max_iter=cfg.max_iter,
    )
    clf.fit(train_x, train_y)
    template_confidence_threshold = None
    template_confidence_report: dict = {}
    if cfg.template_confidence_percentile is not None or cfg.template_confidence_threshold is not None:
        # No dedicated validation split here, so calibrate on the training rows the model
        # was fitted on. Confidence is optimistic in-sample, making this a conservative
        # (high) cut-off; --template-confidence-threshold overrides it if that is unwanted.
        template_confidence_threshold, template_confidence_report = calibrate_template_confidence(
            clf.predict_proba(train_x).max(axis=1),
            train_df["template"].to_numpy(),
            cfg.template_confidence_percentile,
            cfg.template_confidence_threshold,
        )

    pred_idx = clf.predict(test_x)
    pred_labels = np.array([idx_to_label[int(idx)] for idx in pred_idx])
    true_labels = np.array([idx_to_label[int(idx)] for idx in test_y])
    metrics = compute_metrics(true_labels, pred_labels)

    probabilities = clf.predict_proba(test_x)
    confidences = probabilities.max(axis=1)

    predictions_df = test_df.copy()
    predictions_df["pred_template"] = pred_labels
    predictions_df["confidence"] = confidences
    predictions_df["correct"] = predictions_df["pred_template"].to_numpy() == predictions_df["template"].to_numpy()
    predictions_df = predictions_df.loc[:, ["image_path", "template", "pred_template", "confidence", "correct", "source"]]
    predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)

    open_set_summary = None
    if cfg.open_set_dir is not None:
        open_df = collect_open_set_rows(cfg.open_set_dir)
        open_x = encode_features(open_df)
        # Logistic regression has no reject path: every image gets its argmax template.
        # The probability is written alongside so a cut-off can be applied post hoc.
        open_labels = np.array([idx_to_label[int(idx)] for idx in clf.predict(open_x)], dtype=object)
        open_conf = clf.predict_proba(open_x).max(axis=1)
        open_labels, threshold_summary = apply_template_confidence(
            open_labels, open_conf, template_confidence_threshold, cfg.open_set_reject_label
        )
        template_confidence_report.update(threshold_summary)
        _, open_set_summary = write_open_set_predictions(
            run_dir, open_df, open_labels, extra_columns={"confidence": open_conf},
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
        "template_confidence": template_confidence_report,
        "dataset": {
            "images_total_after_filtering": int(len(df)),
            "classes_after_filtering": int(df["template"].nunique()),
            "train_images": int(len(train_df)),
            "test_images": int(len(test_df)),
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

    print(f"test_accuracy={metrics['accuracy']:.4f}")
    print(f"test_f1={metrics['f1']:.4f}")
    print(f"test_mcc={metrics['mcc']:.4f}")
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
