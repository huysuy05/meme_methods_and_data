#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from scipy.fftpack import dct
from transformers import AutoImageProcessor, CLIPVisionModel

from seed_unsupervised_eval import (
    align_dataframe,
    build_reducer,
    collect_labeled_rows,
    compute_centroids,
    compute_metrics,
    drop_small_classes,
    filter_valid_image_rows,
    majority_vote_mapping,
    parse_int_list,
    pick_device,
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
    resolve_seed_paths,
    restrict_to_labeled_clusters,
    summarize_batch_timings,
)
from gpu_backend import (
    assign_to_centroids,
    assign_to_centroids_with_scores,
    configure_torch_backends,
    encode_single_backbone,
    fit_hdbscan,
    resolve_amp_dtype,
)
from gpu_hash_utils import encode_phashes_gpu


@dataclass
class RunConfig:
    imgflip_root: Path | None
    output_dir: Path
    train_parquet: Path | None = None
    test_parquet: Path | None = None
    train_size: float = 0.80
    min_images_per_class: int = 7
    random_seed: int = 42
    batch_size: int = 64
    embedding_method: str = "clip"
    clip_model_id: str = "openai/clip-vit-base-patch32"
    reducer_dim: int = 128
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int | None = None
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
    num_workers: int = 8
    io_workers: int = 16
    encode_dtype: str = "fp32"


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "BERTopic-style unsupervised template discovery baseline adapted to the local dataset: "
            "choose either CLIP or pHash embeddings, reduce them with UMAP, cluster with HDBSCAN, "
            "and assign template names by majority vote on the discovery split."
        )
    )
    parser.add_argument("--imgflip-root", default=None)
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--test-parquet", default=None)
    parser.add_argument("--output-dir", default="SEED/uns_runs/bertopic_clip_phash_hdbscan_eval")
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--min-images-per-class", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-method", choices=["clip", "phash"], default="clip")
    parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--reducer-dim", type=int, default=128)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=10)
    parser.add_argument("--hdbscan-min-samples", type=int, default=5)
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers for image decode/preprocess. 0 starves the GPU.",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=16,
        help="Threads used for the up-front image readability scan.",
    )
    parser.add_argument(
        "--encode-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="fp32",
        help=(
            "Autocast dtype for CLIP embedding. Defaults to fp32: bf16 is roughly 2x "
            "faster but perturbs the embeddings that clustering consumes."
        ),
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
        batch_size=args.batch_size,
        embedding_method=args.embedding_method,
        clip_model_id=args.clip_model_id,
        reducer_dim=args.reducer_dim,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
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
        num_workers=args.num_workers,
        io_workers=args.io_workers,
        encode_dtype=args.encode_dtype,
    )


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


class ClipPhashEmbedder:
    """CLIP or pHash embeddings, both computed on the GPU.

    CLIP runs through a DataLoader so images are decoded in worker processes instead of
    the main loop; pHash defers to ``gpu_hash_utils.encode_phashes_gpu``, which turns the
    2-D DCT into a pair of batched matmuls.
    """

    PHASH_HASH_SIZE = 8
    PHASH_HIGHFREQ_FACTOR = 4

    def __init__(
        self,
        embedding_method: str,
        model_id: str,
        device: torch.device,
        batch_size: int,
        num_workers: int = 8,
        amp_dtype: torch.dtype | None = None,
    ):
        self.embedding_method = embedding_method
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.amp_dtype = amp_dtype
        self.processor = None
        self.model = None
        if self.embedding_method == "clip":
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = CLIPVisionModel.from_pretrained(model_id, use_safetensors=True).to(device).eval()

    def encode_paths(self, paths: list[str], desc: str) -> tuple[np.ndarray, list[str]]:
        if self.embedding_method == "clip":
            assert self.processor is not None and self.model is not None
            return encode_single_backbone(
                self.model,
                self.processor,
                paths,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                device=self.device,
                desc=desc,
                amp_dtype=self.amp_dtype,
            )
        if self.embedding_method == "phash":
            bits, kept_paths = encode_phashes_gpu(
                paths,
                hash_size=self.PHASH_HASH_SIZE,
                device=self.device,
                batch_size=max(self.batch_size, 512),
                num_workers=max(self.num_workers, 8),
                highfreq_factor=self.PHASH_HIGHFREQ_FACTOR,
                desc=desc,
            )
            return bits.astype(np.float32), kept_paths
        raise ValueError(f"Unsupported embedding_method: {self.embedding_method}")


def compute_phash_bits(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """CPU reference implementation. ``encode_phashes_gpu`` reproduces this bit for bit."""
    img_size = hash_size * highfreq_factor
    gray = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    dct_rows = dct(pixels, axis=0, norm="ortho")
    dct_2d = dct(dct_rows, axis=1, norm="ortho")
    low_freq = dct_2d[:hash_size, :hash_size]
    flat = low_freq.reshape(-1)
    median = np.median(flat[1:]) if flat.size > 1 else np.median(flat)
    return (low_freq > median).astype(np.uint8).reshape(-1)

def select_features(embedding_method: str, features: np.ndarray) -> np.ndarray:
    if embedding_method == "clip":
        return l2_normalize(features.astype(np.float32))
    if embedding_method == "phash":
        phash_centered = features.astype(np.float32) * 2.0 - 1.0
        return l2_normalize(phash_centered)
    raise ValueError(f"Unsupported embedding_method: {embedding_method}")


# fit_hdbscan and assign_to_centroids now come from gpu_backend (cuML HDBSCAN and a
# chunked GPU matmul respectively).


def run_once(cfg: RunConfig, run_dir: Path) -> dict[str, float | int | str]:
    set_seed(cfg.random_seed)
    run_started_at = now_iso()
    start_perf = time.perf_counter()
    device = pick_device()
    configure_torch_backends()
    encode_dtype = resolve_amp_dtype(cfg.encode_dtype)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device} ({torch.cuda.get_device_name(0)})")
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
        imgflip_df = filter_valid_image_rows(imgflip_df, io_workers=cfg.io_workers)
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

    embedder = ClipPhashEmbedder(
        embedding_method=cfg.embedding_method,
        model_id=cfg.clip_model_id,
        device=device,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        amp_dtype=encode_dtype,
    )
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

    train_paths = discovery_df["image_path"].tolist()
    test_paths = test_df["image_path"].tolist()

    train_encoded, kept_train_paths = embedder.encode_paths(
        train_paths,
        desc=f"encode discovery {cfg.embedding_method}",
    )
    test_encoded, kept_test_paths = embedder.encode_paths(
        test_paths,
        desc=f"encode test {cfg.embedding_method}",
    )
    discovery_df = align_dataframe(discovery_df, kept_train_paths)
    test_df = align_dataframe(test_df, kept_test_paths)

    train_features = select_features(cfg.embedding_method, train_encoded)
    test_features = select_features(cfg.embedding_method, test_encoded)

    reducer = build_reducer("umap", cfg.reducer_dim, cfg.random_seed)
    train_reduced = reducer.fit_transform(train_features)
    test_reduced = reducer.transform(test_features)
    train_reduced = l2_normalize(np.asarray(train_reduced, dtype=np.float32))
    test_reduced = l2_normalize(np.asarray(test_reduced, dtype=np.float32))

    _, cluster_labels = fit_hdbscan(
        train_reduced,
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
    )
    centroids, centroid_ids = compute_centroids(train_reduced, cluster_labels, device)
    cluster_to_template = majority_vote_mapping(discovery_df, cluster_labels)
    # Clusters made purely of unlabeled corpus images have no template; dropping them keeps
    # every test prediction a real template name.
    all_centroids, all_centroid_ids = centroids, centroid_ids
    _, cluster_examples_summary = write_cluster_examples(
        run_dir, discovery_df, cluster_labels, cluster_to_template,
        member_scores=cosine_to_own_centroid(train_reduced, cluster_labels, all_centroids, all_centroid_ids),
    )
    centroids, centroid_ids = restrict_to_labeled_clusters(centroids, centroid_ids, cluster_to_template)
    test_cluster_ids = assign_to_centroids(test_reduced, centroids, centroid_ids, device)
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
        open_encoded, kept_open_paths = embedder.encode_paths(
            open_df["image_path"].tolist(), desc=f"encode open-set {cfg.embedding_method}"
        )
        open_df = align_dataframe(open_df, kept_open_paths)
        open_reduced = reducer.transform(select_features(cfg.embedding_method, open_encoded))
        open_reduced = l2_normalize(np.asarray(open_reduced, dtype=np.float32))
        # Open-set assignment ranks against *every* discovered cluster, including the ones only
        # social-media images populate. Landing in an unnamed cluster is the model's own signal
        # that the image matches no known template, so it becomes the reject label here instead
        # of being forced onto the nearest named centroid the way the closed-set test path does.
        open_cluster_ids, open_cosine = assign_to_centroids_with_scores(open_reduced, all_centroids, all_centroid_ids, device)
        open_labels = np.array(
            [cluster_to_template.get(int(cid), cfg.open_set_reject_label) for cid in open_cluster_ids], dtype=object
        )
        _, open_set_summary = write_open_set_predictions(
            run_dir, open_df, open_labels, extra_columns={"pred_cluster": open_cluster_ids, "centroid_cosine": open_cosine},
        )

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
            "device": str(device),
            "discovery_extra_roots": None
            if cfg.discovery_extra_roots is None
            else [str(root) for root in cfg.discovery_extra_roots],
        },
        "non_meme_class": non_meme_summary,
        "cluster_examples": cluster_examples_summary,
        "dataset": {
            "imgflip_images_total_after_filtering": int(len(imgflip_df)),
            "imgflip_classes_after_filtering": int(imgflip_df["template"].nunique()),
            "imgflip_train_images": int(discovery_df["source"].eq("imgflip_train").sum()),
            "imgflip_test_images": int(len(test_df)),
            "unlabeled_corpus_images": int(discovery_df["source"].ne("imgflip_train").sum()),
            "discovery_images_total": int(len(discovery_df)),
            "small_class_filtering": filtering_summary,
        },
        "clustering": {
            "cluster_method": "hdbscan",
            "discovered_cluster_count": int(len({int(c) for c in cluster_labels if c >= 0})),
            "labeled_cluster_count": int(len(centroid_ids)),
            "mapped_cluster_count": int(len(cluster_to_template)),
            "unknown_cluster_predictions": int((test_pred_templates == "UNKNOWN_CLUSTER").sum()),
        },
        "test_metrics": metrics,
    }
    summary_row = {
        "run_dir": str(run_dir),
        "seed": int(cfg.random_seed),
        "embedding_method": cfg.embedding_method,
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "mcc": float(metrics["mcc"]),
        "cohen_kappa": float(metrics["cohen_kappa"]),
        "discovered_cluster_count": int(len(centroid_ids)),
        "mapped_cluster_count": int(len(cluster_to_template)),
    }
    if open_set_summary is not None and cfg.open_set_labels is not None:
        open_set_summary["metrics"] = score_open_set_predictions(run_dir, cfg.open_set_labels)
    metadata["open_set"] = open_set_summary
    timing = finalize_run_timing(metadata, summary_row, run_started_at, start_perf)
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))
    (run_dir / "test_metrics.json").write_text(json.dumps({"test_metrics": metrics, "timing": timing}, indent=2))
    pd.DataFrame([summary_row]).to_csv(run_dir / "metrics_summary.csv", index=False)

    print(f"discovery_images={len(discovery_df)} imgflip_test_images={len(test_df)}")
    print(f"discovered_cluster_count={len(centroid_ids)} mapped_cluster_count={len(cluster_to_template)}")
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
