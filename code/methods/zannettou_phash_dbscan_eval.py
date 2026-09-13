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
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from seed_unsupervised_eval import (
    align_dataframe,
    collect_labeled_rows,
    compute_metrics,
    drop_small_classes,
    filter_valid_image_rows,
    load_rgb_image,
    majority_vote_mapping,
    parse_int_list,
    set_seed,
)
from meme_research_eval_utils import (
    add_non_meme_args,
    cosine_to_own_centroid,
    write_cluster_examples,
    add_non_meme_class,
    collect_non_meme_rows,
    OPEN_SET_TEMPLATE_FREE,
    add_open_set_args,
    collect_open_set_rows,
    score_open_set_predictions,
    write_open_set_predictions,
    collect_unlabeled_corpus_rows,
    finalize_run_timing,
    load_split_rows,
    now_iso,
    parse_path_list,
    pick_device,
    resolve_seed_paths,
    restrict_to_labeled_clusters,
    summarize_batch_timings,
)
from gpu_hash_utils import encode_phashes_gpu, radius_neighbor_graph_hamming_gpu


@dataclass
class RunConfig:
    imgflip_root: Path
    output_dir: Path
    train_parquet: Path | None = None
    test_parquet: Path | None = None
    train_size: float = 0.80
    min_images_per_class: int = 7
    random_seed: int = 42
    phash_distance_threshold: int = 8
    phash_hash_size: int = 8
    dbscan_min_samples: int = 4
    max_templates: int | None = None
    max_images_per_template: int | None = None
    seeds: tuple[int, ...] | None = None
    open_set_dir: Path | None = None
    open_set_reject_label: str = OPEN_SET_TEMPLATE_FREE
    open_set_labels: Path | None = None
    non_meme_root: Path | None = None
    max_non_meme_images: int | None = None
    discovery_extra_roots: tuple[Path, ...] | None = None
    max_corpus_images: int | None = None
    phash_batch_size: int = 512
    phash_workers: int = 8
    pairwise_tile_size: int = 8192
    max_neighbors: int | None = None



def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "pHash + DBSCAN baseline adapted from Zannettou et al. (2018), using the local dataset "
            "split and majority-vote cluster naming instead of KYM-based annotation."
        )
    )
    parser.add_argument("--imgflip-root", default=None)
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--test-parquet", default=None)
    parser.add_argument("--output-dir", default="SEED/uns_runs/zannettou_phash_dbscan_eval")
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--min-images-per-class", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--phash-distance-threshold", type=int, default=8)
    parser.add_argument("--phash-hash-size", type=int, default=8)
    parser.add_argument("--dbscan-min-samples", type=int, default=5)
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--max-images-per-template", type=int, default=None)
    parser.add_argument("--seeds", type=parse_int_list, default=None)
    parser.add_argument(
        "--discovery-extra-roots",
        type=parse_path_list,
        default=None,
        help="Comma-separated image folders (e.g. social-media corpora) added to the clustering "
             "corpus as unlabeled data. They never contribute to majority-vote naming.",
    )
    parser.add_argument(
        "--max-corpus-images",
        type=int,
        default=None,
        help="Optionally subsample the unlabeled corpus to this many images (for smoke tests).",
    )
    parser.add_argument("--phash-batch-size", type=int, default=512)
    parser.add_argument("--phash-workers", type=int, default=8)
    parser.add_argument(
        "--pairwise-tile-size",
        type=int,
        default=8192,
        help="Tile edge for the GPU pairwise Hamming comparison. Lower it if VRAM is tight.",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=None,
        help="Cap neighbours kept per image. Bounds memory on corpora with huge near-duplicate "
             "groups; leave unset for exact DBSCAN behaviour.",
    )
    add_non_meme_args(parser)
    add_open_set_args(parser)
    args = parser.parse_args()
    return RunConfig(
        imgflip_root=None if args.imgflip_root is None else Path(args.imgflip_root).expanduser().resolve(),
        train_parquet=None if args.train_parquet is None else Path(args.train_parquet).expanduser().resolve(),
        test_parquet=None if args.test_parquet is None else Path(args.test_parquet).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        train_size=args.train_size,
        min_images_per_class=args.min_images_per_class,
        random_seed=args.random_seed,
        phash_distance_threshold=args.phash_distance_threshold,
        phash_hash_size=args.phash_hash_size,
        dbscan_min_samples=args.dbscan_min_samples,
        max_templates=args.max_templates,
        max_images_per_template=args.max_images_per_template,
        seeds=args.seeds,
        open_set_dir=None if args.open_set_dir is None else Path(args.open_set_dir).expanduser().resolve(),
        open_set_reject_label=args.open_set_reject_label,
        open_set_labels=None if args.open_set_labels is None else Path(args.open_set_labels).expanduser().resolve(),
        non_meme_root=None if args.non_meme_root is None else Path(args.non_meme_root).expanduser().resolve(),
        max_non_meme_images=args.max_non_meme_images,
        discovery_extra_roots=args.discovery_extra_roots,
        max_corpus_images=args.max_corpus_images,
        phash_batch_size=args.phash_batch_size,
        phash_workers=args.phash_workers,
        pairwise_tile_size=args.pairwise_tile_size,
        max_neighbors=args.max_neighbors,
    )


def compute_phash_bits(image: Image.Image, hash_size: int) -> np.ndarray:
    img_size = hash_size * 4
    gray = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    dct_rows = dct(pixels, axis=0, norm="ortho")
    dct_2d = dct(dct_rows, axis=1, norm="ortho")
    low_freq = dct_2d[:hash_size, :hash_size]
    flat = low_freq.reshape(-1)
    median = np.median(flat[1:]) if flat.size > 1 else np.median(flat)
    return (low_freq > median).astype(np.uint8).reshape(-1)


def encode_phashes(paths: list[str], hash_size: int, desc: str) -> tuple[np.ndarray, list[str]]:
    hashes: list[np.ndarray] = []
    kept_paths: list[str] = []
    for path in tqdm(paths, desc=desc):
        image = load_rgb_image(path)
        if image is None:
            continue
        hashes.append(compute_phash_bits(image, hash_size))
        kept_paths.append(path)
    if not hashes:
        raise ValueError(f"No valid images encoded for {desc}")
    return np.vstack(hashes), kept_paths


def compute_cluster_representatives(hash_bits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cluster_ids = np.array(sorted([cluster for cluster in np.unique(labels) if cluster >= 0]), dtype=np.int64)
    representatives: list[np.ndarray] = []
    for cluster_id in cluster_ids:
        members = hash_bits[labels == cluster_id]
        majority = (members.mean(axis=0) >= 0.5).astype(np.uint8)
        distances = cdist(members.astype(np.float32), majority[None, :].astype(np.float32), metric="hamming").reshape(-1)
        representatives.append(members[int(distances.argmin())].astype(np.uint8))
    if not representatives:
        raise ValueError("No non-noise clusters were found.")
    return np.vstack(representatives), cluster_ids


def assign_to_representatives(test_hashes: np.ndarray, representatives: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    distances = cdist(test_hashes.astype(np.float32), representatives.astype(np.float32), metric="hamming")
    best = distances.argmin(axis=1)
    return cluster_ids[best]


def run_once(cfg: RunConfig, run_dir: Path) -> dict[str, float | int | str]:
    set_seed(cfg.random_seed)
    run_started_at = now_iso()
    start_perf = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir={run_dir}")

    if cfg.train_parquet is not None and cfg.test_parquet is not None:
        train_df, test_df = load_split_rows(str(cfg.train_parquet), str(cfg.test_parquet))
        imgflip_df = pd.concat([train_df, test_df], ignore_index=True)
        filtering_summary = {"precomputed_split": True}
    else:
        if cfg.imgflip_root is None:
            raise ValueError("Either --imgflip-root or both --train-parquet and --test-parquet must be provided.")
        imgflip_df = collect_labeled_rows(
            cfg.imgflip_root,
            max_templates=cfg.max_templates,
            max_images_per_template=cfg.max_images_per_template,
        )
        imgflip_df = filter_valid_image_rows(imgflip_df)
        imgflip_df, filtering_summary = drop_small_classes(imgflip_df, cfg.min_images_per_class)

        train_df, test_df = train_test_split(
            imgflip_df,
            train_size=cfg.train_size,
            stratify=imgflip_df["template"],
            random_state=cfg.random_seed,
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_df["source"] = "imgflip_train"
        test_df["source"] = "imgflip_test"

    # Discovery corpus = labeled ImgFlip train split + any unlabeled corpora (social media).
    # Only the ImgFlip rows carry source == "imgflip_train", so majority_vote_mapping keeps
    # naming the clusters from labeled data alone.
    non_meme_summary = None
    if cfg.non_meme_root is not None:
        # Labelled like the ImgFlip rows, so majority_vote_mapping can name a cluster
        # "Non-Meme". The method still decides entirely by clustering.
        non_meme_df = collect_non_meme_rows(
            cfg.non_meme_root, exclude_dir=cfg.open_set_dir,
            max_images=cfg.max_non_meme_images, random_seed=cfg.random_seed,
        )
        non_meme_df["source"] = "imgflip_train"   # counted by the majority vote
        train_df, _, non_meme_summary = add_non_meme_class(
            train_df, None, non_meme_df, cfg.train_size, cfg.random_seed
        )
    discovery_parts = [train_df]
    corpus_count = 0
    if cfg.discovery_extra_roots:
        corpus_df = collect_unlabeled_corpus_rows(
            cfg.discovery_extra_roots,
            max_images=cfg.max_corpus_images,
            random_seed=cfg.random_seed,
            exclude_dir=cfg.open_set_dir,
        )
        corpus_count = len(corpus_df)
        if corpus_count:
            discovery_parts.append(corpus_df)
    discovery_df = pd.concat(discovery_parts, ignore_index=True)
    print(f"discovery_images={len(discovery_df)} (imgflip_train={len(train_df)} unlabeled={corpus_count})")

    device = pick_device()
    print(f"device={device}")

    discovery_hashes, kept_discovery_paths = encode_phashes_gpu(
        discovery_df["image_path"].tolist(),
        hash_size=cfg.phash_hash_size,
        device=device,
        batch_size=cfg.phash_batch_size,
        num_workers=cfg.phash_workers,
        desc="encode discovery phash",
    )
    test_hashes, kept_test_paths = encode_phashes_gpu(
        test_df["image_path"].tolist(),
        hash_size=cfg.phash_hash_size,
        device=device,
        batch_size=cfg.phash_batch_size,
        num_workers=cfg.phash_workers,
        desc="encode test phash",
    )
    discovery_df = align_dataframe(discovery_df, kept_discovery_paths)
    test_df = align_dataframe(test_df, kept_test_paths)

    # Pairwise Hamming comparison of every pHash (Zannettou et al., Steps 2-3). A dense
    # matrix is impossible at corpus scale, so the GPU emits only pairs within the threshold
    # and DBSCAN consumes that sparse graph directly.
    neighbor_graph = radius_neighbor_graph_hamming_gpu(
        discovery_hashes,
        max_bits=cfg.phash_distance_threshold,
        device=device,
        tile_size=cfg.pairwise_tile_size,
        max_neighbors=cfg.max_neighbors,
    )
    clusterer = DBSCAN(
        eps=float(cfg.phash_distance_threshold),
        min_samples=cfg.dbscan_min_samples,
        metric="precomputed",
    )
    cluster_labels = clusterer.fit_predict(neighbor_graph.tocsr())

    representatives, cluster_ids = compute_cluster_representatives(discovery_hashes, cluster_labels)
    cluster_to_template = majority_vote_mapping(discovery_df, cluster_labels)
    # Clusters made purely of unlabeled corpus images have no template; dropping them keeps
    # every test prediction a real template name.
    all_representatives, all_cluster_ids = representatives, cluster_ids
    # Hamming distance to the cluster's own representative, negated so higher = more typical.
    _rep_lookup = {int(c): i for i, c in enumerate(all_cluster_ids)}
    _member_scores = np.full(len(cluster_labels), -np.inf)
    for _pos, _lab in enumerate(cluster_labels):
        _idx = _rep_lookup.get(int(_lab))
        if _idx is not None:
            _member_scores[_pos] = -float(np.count_nonzero(discovery_hashes[_pos] != all_representatives[_idx]))
    _, cluster_examples_summary = write_cluster_examples(
        run_dir, discovery_df, cluster_labels, cluster_to_template, member_scores=_member_scores,
    )
    representatives, cluster_ids = restrict_to_labeled_clusters(representatives, cluster_ids, cluster_to_template)
    test_cluster_ids = assign_to_representatives(test_hashes, representatives, cluster_ids)
    test_pred_templates = np.array([cluster_to_template.get(int(cid), "UNKNOWN_CLUSTER") for cid in test_cluster_ids])
    y_true = test_df["template"].to_numpy()
    metrics = compute_metrics(y_true, test_pred_templates)

    predictions_df = test_df.copy()
    predictions_df["pred_cluster"] = test_cluster_ids
    predictions_df["pred_template"] = test_pred_templates
    predictions_df["correct"] = predictions_df["pred_template"].to_numpy() == predictions_df["template"].to_numpy()
    predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)

    open_set_summary = None
    if cfg.open_set_dir is not None:
        open_df = collect_open_set_rows(cfg.open_set_dir)
        open_hashes, kept_open_paths = encode_phashes_gpu(
            open_df["image_path"].tolist(),
            hash_size=cfg.phash_hash_size,
            device=device,
            batch_size=cfg.phash_batch_size,
            num_workers=cfg.phash_workers,
            desc="encode open-set phash",
        )
        open_df = align_dataframe(open_df, kept_open_paths)
        # Open-set assignment ranks against *every* discovered cluster, including the ones only
        # social-media images populate. Landing in an unnamed cluster is the model's own signal
        # that the image matches no known template, so it becomes the reject label here instead
        # of being forced onto the nearest named centroid the way the closed-set test path does.
        open_dist = cdist(open_hashes.astype(np.float32), all_representatives.astype(np.float32), metric="hamming")
        open_best = open_dist.argmin(axis=1)
        open_cluster_ids = all_cluster_ids[open_best]
        open_bits = np.rint(open_dist[np.arange(len(open_best)), open_best] * open_hashes.shape[1]).astype(np.int64)
        open_labels = np.array(
            [cluster_to_template.get(int(cid), cfg.open_set_reject_label) for cid in open_cluster_ids], dtype=object
        )
        _, open_set_summary = write_open_set_predictions(
            run_dir,
            open_df,
            open_labels,
            extra_columns={"pred_cluster": open_cluster_ids, "nearest_representative_hamming_bits": open_bits},
        )

    labeled_discovery = int(discovery_df["source"].eq("imgflip_train").sum())
    metadata = {
        "config": {
            **asdict(cfg),
            "imgflip_root": None if cfg.imgflip_root is None else str(cfg.imgflip_root),
            "train_parquet": None if cfg.train_parquet is None else str(cfg.train_parquet),
            "test_parquet": None if cfg.test_parquet is None else str(cfg.test_parquet),
            "output_dir": str(cfg.output_dir),
            "open_set_dir": None if cfg.open_set_dir is None else str(cfg.open_set_dir),
            "open_set_labels": None if cfg.open_set_labels is None else str(cfg.open_set_labels),
            "non_meme_root": None if cfg.non_meme_root is None else str(cfg.non_meme_root),
            "discovery_extra_roots": None
            if cfg.discovery_extra_roots is None
            else [str(root) for root in cfg.discovery_extra_roots],
            "device": str(device),
        },
        "non_meme_class": non_meme_summary,
        "cluster_examples": cluster_examples_summary,
        "dataset": {
            "imgflip_images_total_after_filtering": int(len(imgflip_df)),
            "imgflip_classes_after_filtering": int(imgflip_df["template"].nunique()),
            "imgflip_train_images": labeled_discovery,
            "imgflip_test_images": int(len(test_df)),
            "unlabeled_corpus_images": int(len(discovery_df) - labeled_discovery),
            "discovery_images_total": int(len(discovery_df)),
            "small_class_filtering": filtering_summary,
        },
        "clustering": {
            "cluster_method": "dbscan_hamming_gpu_pairwise",
            "distance_threshold_bits": int(cfg.phash_distance_threshold),
            "hash_size": int(cfg.phash_hash_size),
            "neighbor_graph_edges": int(neighbor_graph.nnz),
            "discovered_cluster_count": int(len({int(c) for c in cluster_labels if c >= 0})),
            "labeled_cluster_count": int(len(cluster_ids)),
            "mapped_cluster_count": int(len(cluster_to_template)),
            "noise_points": int((cluster_labels < 0).sum()),
            "unknown_cluster_predictions": int((test_pred_templates == "UNKNOWN_CLUSTER").sum()),
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
        "discovered_cluster_count": int(len(cluster_ids)),
        "mapped_cluster_count": int(len(cluster_to_template)),
    }
    if open_set_summary is not None and cfg.open_set_labels is not None:
        open_set_summary["metrics"] = score_open_set_predictions(run_dir, cfg.open_set_labels)
    metadata["open_set"] = open_set_summary
    timing = finalize_run_timing(metadata, summary_row, run_started_at, start_perf)
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))
    (run_dir / "test_metrics.json").write_text(json.dumps({"test_metrics": metrics, "timing": timing}, indent=2))
    pd.DataFrame([summary_row]).to_csv(run_dir / "metrics_summary.csv", index=False)

    print(f"discovery_images={len(discovery_df)} imgflip_train_images={labeled_discovery} imgflip_test_images={len(test_df)}")
    print(f"discovered_cluster_count={len(cluster_ids)} mapped_cluster_count={len(cluster_to_template)}")
    print(f"test_accuracy={metrics['accuracy']:.4f}")
    print(f"test_f1={metrics['f1']:.4f}")
    print(f"test_precision={metrics['precision']:.4f}")
    print(f"test_recall={metrics['recall']:.4f}")
    print(f"test_mcc={metrics['mcc']:.4f}")
    print(f"test_cohen_kappa={metrics['cohen_kappa']:.4f}")
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
