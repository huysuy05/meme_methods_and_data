#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import struct
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, Dinov2Model, SiglipVisionModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

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
    IdentityReducer,
    PathDataset,
    assign_to_centroids,
    assign_to_centroids_with_scores,
    build_reducer,
    compute_centroids,
    configure_torch_backends,
    encode_dual_backbones,
    encode_single_backbone,
    filter_valid_image_paths,
    fit_clusterer,
    fit_kmeans,
    move_batch_to_device,
    pooled_output,
    refine_clusters,
    require_cuda,
    resolve_amp_dtype,
)

# Re-exported so the sibling eval scripts keep importing these from here.
__all__ = [
    "IdentityReducer",
    "align_dataframe",
    "assign_to_centroids",
    "build_reducer",
    "collect_labeled_rows",
    "compute_centroids",
    "compute_metrics",
    "drop_small_classes",
    "filter_valid_image_rows",
    "fit_clusterer",
    "load_rgb_image",
    "majority_vote_mapping",
    "parse_int_list",
    "pick_device",
    "refine_clusters",
    "set_seed",
]


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


@dataclass
class RunConfig:
    imgflip_root: Path | None
    train_parquet: Path | None
    test_parquet: Path | None
    output_dir: Path
    train_size: float = 0.80
    min_images_per_class: int = 7
    random_seed: int = 42
    batch_size: int = 16
    num_workers: int = 0
    siglip_model_id: str = "google/siglip2-base-patch16-224"
    dino_model_id: str = "facebook/dinov2-base"
    alpha: float = 0.65
    reducer: str = "pca"
    reducer_dim: int = 128
    cluster_method: str = "kmeans"
    kmeans_clusters: int | None = None
    refinement_steps: int = 1
    max_templates: int | None = None
    max_images_per_template: int | None = None
    max_unlabeled_images: int | None = None
    discovery_extra_roots: tuple[Path, ...] | None = None
    max_corpus_images: int | None = None
    use_ssft: bool = False
    ssft_checkpoint_dir: Path | None = None
    ssft_epochs: int = 1
    ssft_lr: float = 1e-5
    ssft_weight_decay: float = 1e-4
    ssft_temperature: float = 0.1
    ssft_projection_dim: int = 256
    ssft_objective: str = "ntxent"
    ssft_deepcluster_clusters: int | None = None
    ssft_early_stopping_patience: int = 2
    ssft_early_stopping_min_delta: float = 1e-4
    ssft_val_size: float = 0.10
    amp_dtype: str = "bf16"
    encode_dtype: str = "fp32"
    io_workers: int = 16
    open_set_dir: Path | None = None
    open_set_reject_label: str = OPEN_SET_TEMPLATE_FREE
    open_set_labels: Path | None = None
    non_meme_root: Path | None = None
    max_non_meme_images: int | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    """CUDA or nothing. This pipeline is GPU-only; see gpu_backend.require_cuda."""
    return require_cuda()


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


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
    root: Path,
    max_templates: int | None = None,
    max_images_per_template: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    template_dirs = sorted([path for path in root.iterdir() if path.is_dir()])
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
        raise ValueError(f"No labeled images found under {root}")
    return pd.DataFrame(rows)


def split_labeled_and_unlabeled_from_train(
    train_df: pd.DataFrame,
    max_unlabeled_images: int | None,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_unlabeled_images is None or max_unlabeled_images <= 0:
        return train_df.reset_index(drop=True), pd.DataFrame(columns=train_df.columns)

    sample_n = min(int(max_unlabeled_images), len(train_df))
    unlabeled_idx = (
        train_df.sample(n=sample_n, random_state=random_seed, replace=False).index
        if sample_n > 0
        else []
    )
    labeled_df = train_df.reset_index(drop=True).copy()
    unlabeled_df = train_df.loc[unlabeled_idx].reset_index(drop=True).copy()
    if not unlabeled_df.empty:
        unlabeled_df["template"] = "UNLABELED"
        unlabeled_df["source"] = "imgflip_unlabeled"
    return labeled_df, unlabeled_df


def filter_valid_image_rows(df: pd.DataFrame, io_workers: int = 16) -> pd.DataFrame:
    keep_rows = filter_valid_image_paths(df["image_path"].tolist(), num_workers=io_workers)
    filtered = df[keep_rows].reset_index(drop=True)
    dropped = len(df) - len(filtered)
    if dropped:
        print(f"Dropped {dropped} unreadable images.")
    return filtered


def drop_small_classes(df: pd.DataFrame, min_images_per_class: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = df["template"].value_counts()
    keep_templates = counts[counts >= min_images_per_class].index
    filtered = df[df["template"].isin(keep_templates)].copy().reset_index(drop=True)
    removed = counts[counts < min_images_per_class]
    summary = {
        "min_images_required_per_class": int(min_images_per_class),
        "templates_before": int(counts.shape[0]),
        "templates_after": int(filtered["template"].nunique()),
        "templates_removed": int(removed.shape[0]),
        "images_before": int(len(df)),
        "images_after": int(len(filtered)),
        "images_removed": int(len(df) - len(filtered)),
        "removed_template_examples": removed.head(20).to_dict(),
    }
    return filtered, summary


class FrozenImageEmbedder:
    def __init__(
        self,
        model_id: str,
        kind: str,
        device: torch.device,
        batch_size: int,
        num_workers: int = 8,
        amp_dtype: torch.dtype | None = None,
    ):
        self.model_id = model_id
        self.kind = kind
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.amp_dtype = amp_dtype
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        if kind == "siglip":
            self.model = SiglipVisionModel.from_pretrained(model_id).to(device)
        elif kind == "dino":
            self.model = Dinov2Model.from_pretrained(model_id).to(device)
        else:
            raise ValueError(f"Unsupported kind={kind}")
        self.model.eval()

    def encode_paths(self, paths: list[str], desc: str) -> tuple[np.ndarray, list[str]]:
        """Single-backbone encode. Prefer ``encode_dual_backbones`` when two backbones
        cover the same path list, so the images are only decoded once."""
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


class LabeledPathDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int]):
        self.paths = paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.paths[index], int(self.labels[index])


class SSFTCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: list[str]) -> dict[str, Any] | None:
        view1: list[Image.Image] = []
        view2: list[Image.Image] = []
        kept_paths: list[str] = []

        for image_path in batch:
            image = load_rgb_image(image_path)
            if image is None:
                continue
            view1.append(make_ssft_view(image))
            view2.append(make_ssft_view(image))
            kept_paths.append(image_path)

        if not kept_paths:
            return None

        inputs1 = self.processor(images=view1, return_tensors="pt")
        inputs2 = self.processor(images=view2, return_tensors="pt")
        return {"view1": dict(inputs1), "view2": dict(inputs2), "paths": kept_paths}


class SingleViewCollator:
    def __init__(self, processor, augment: bool):
        self.processor = processor
        self.augment = augment

    def __call__(self, batch: list[tuple[str, int]]) -> dict[str, Any] | None:
        images: list[Image.Image] = []
        labels: list[int] = []
        kept_paths: list[str] = []

        for image_path, label in batch:
            image = load_rgb_image(image_path)
            if image is None:
                continue
            images.append(make_ssft_view(image) if self.augment else image)
            labels.append(int(label))
            kept_paths.append(image_path)

        if not kept_paths:
            return None

        inputs = self.processor(images=images, return_tensors="pt")
        return {
            "inputs": dict(inputs),
            "labels": torch.tensor(labels, dtype=torch.long),
            "paths": kept_paths,
        }


class ProjectionHead(nn.Module):
    def __init__(self, hidden_size: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), dim=-1)


class ClusterHead(nn.Module):
    def __init__(self, hidden_size: int, num_clusters: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_clusters)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def make_ssft_view(image: Image.Image) -> Image.Image:
    width, height = image.size
    crop_scale = random.uniform(0.90, 1.00)
    crop_w = max(1, int(width * crop_scale))
    crop_h = max(1, int(height * crop_scale))
    if crop_w < width:
        left = random.randint(0, width - crop_w)
    else:
        left = 0
    if crop_h < height:
        top = random.randint(0, height - crop_h)
    else:
        top = 0
    cropped = image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.Resampling.BICUBIC)

    brightness = random.uniform(0.90, 1.10)
    contrast = random.uniform(0.90, 1.10)
    saturated = ImageEnhance.Brightness(cropped).enhance(brightness)
    saturated = ImageEnhance.Contrast(saturated).enhance(contrast)

    if random.random() < 0.20:
        saturated = saturated.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))

    return saturated


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    representations = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
    logits = representations @ representations.T
    logits = logits / temperature
    batch_size = z1.shape[0]
    logits = logits.masked_fill(
        torch.eye(2 * batch_size, device=logits.device, dtype=torch.bool),
        float("-inf"),
    )
    targets = torch.arange(2 * batch_size, device=logits.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, targets)


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    repr_loss = F.mse_loss(z1, z2)

    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)

    std_z1 = torch.sqrt(z1_centered.var(dim=0, unbiased=False) + eps)
    std_z2 = torch.sqrt(z2_centered.var(dim=0, unbiased=False) + eps)
    std_loss = (F.relu(1.0 - std_z1).mean() + F.relu(1.0 - std_z2).mean())

    n = z1.shape[0]
    cov_z1 = (z1_centered.T @ z1_centered) / max(n - 1, 1)
    cov_z2 = (z2_centered.T @ z2_centered) / max(n - 1, 1)
    off_diag_mask = ~torch.eye(cov_z1.shape[0], device=z1.device, dtype=torch.bool)
    cov_loss = cov_z1[off_diag_mask].pow(2).mean() + cov_z2[off_diag_mask].pow(2).mean()

    return sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss


def compute_ssft_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    objective: str,
    temperature: float,
) -> torch.Tensor:
    objective = objective.lower()
    if objective == "ntxent":
        return nt_xent_loss(z1, z2, temperature=temperature)
    if objective == "vicreg":
        return vicreg_loss(z1, z2)
    raise ValueError(f"Unsupported SSFT objective={objective}")


def infer_deepcluster_cluster_count(num_paths: int, configured_clusters: int | None) -> int:
    if configured_clusters is not None:
        return max(2, int(configured_clusters))
    heuristic = max(32, min(1024, num_paths // 100))
    return max(2, heuristic)


def encode_backbone_features(
    model: nn.Module,
    processor,
    paths: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str,
    amp_dtype: torch.dtype | None = None,
) -> tuple[np.ndarray, list[str]]:
    return encode_single_backbone(
        model,
        processor,
        paths,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        desc=desc,
        amp_dtype=amp_dtype,
    )


def build_deepcluster_assignments(
    model: nn.Module,
    processor,
    train_paths: list[str],
    val_paths: list[str],
    cfg: RunConfig,
    device: torch.device,
    epoch: int,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    train_features, kept_train_paths = encode_backbone_features(
        model,
        processor,
        train_paths,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        desc=f"deepcluster encode train epoch {epoch}",
    )
    num_clusters = infer_deepcluster_cluster_count(len(kept_train_paths), cfg.ssft_deepcluster_clusters)
    num_clusters = min(num_clusters, len(kept_train_paths))
    clusterer, train_labels = fit_kmeans(train_features, num_clusters, cfg.random_seed)

    kept_val_paths: list[str] = []
    val_labels = np.array([], dtype=np.int64)
    if val_paths:
        val_features, kept_val_paths = encode_backbone_features(
            model,
            processor,
            val_paths,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            device=device,
            desc=f"deepcluster encode val epoch {epoch}",
        )
        centroids = l2_normalize(np.asarray(clusterer.cluster_centers_, dtype=np.float32))
        val_scores = l2_normalize(val_features.astype(np.float32)) @ centroids.T
        val_labels = val_scores.argmax(axis=1).astype(np.int64)

    return kept_train_paths, train_labels, kept_val_paths, val_labels


def split_paths_for_ssft(
    paths: list[str],
    val_size: float,
    random_seed: int,
) -> tuple[list[str], list[str]]:
    if len(paths) < 2:
        return paths, []
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"ssft_val_size must be in (0, 1), got {val_size}")
    rng = random.Random(random_seed)
    shuffled = paths.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_size)))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


@torch.inference_mode()
def evaluate_ssft(
    model: nn.Module,
    projector: ProjectionHead,
    data_loader: DataLoader,
    device: torch.device,
    objective: str,
    temperature: float,
) -> float:
    model.eval()
    projector.eval()
    losses: list[float] = []
    for batch in data_loader:
        if batch is None:
            continue
        view1 = move_batch_to_device(batch["view1"], device)
        view2 = move_batch_to_device(batch["view2"], device)
        features1 = pooled_output(model(**view1)).float()
        features2 = pooled_output(model(**view2)).float()
        z1 = projector(features1)
        z2 = projector(features2)
        losses.append(float(compute_ssft_loss(z1, z2, objective=objective, temperature=temperature).item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.inference_mode()
def evaluate_deepcluster(
    model: nn.Module,
    cluster_head: ClusterHead,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    cluster_head.eval()
    losses: list[float] = []
    for batch in data_loader:
        if batch is None:
            continue
        inputs = move_batch_to_device(batch["inputs"], device)
        labels = batch["labels"].to(device)
        features = pooled_output(model(**inputs)).float()
        logits = cluster_head(features)
        losses.append(float(F.cross_entropy(logits, labels).item()))
    return float(np.mean(losses)) if losses else float("nan")


def build_deepcluster_loader(
    processor,
    paths: list[str],
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    augment: bool,
) -> DataLoader:
    return DataLoader(
        LabeledPathDataset(paths, labels.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=SingleViewCollator(processor, augment=augment),
        pin_memory=torch.cuda.is_available(),
    )


def run_deepcluster_ssft(
    embedder: FrozenImageEmbedder,
    discovery_paths: list[str],
    cfg: RunConfig,
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    train_paths = discovery_paths
    val_paths: list[str] = []
    if cfg.ssft_early_stopping_patience > 0:
        train_paths, val_paths = split_paths_for_ssft(
            discovery_paths,
            val_size=cfg.ssft_val_size,
            random_seed=cfg.random_seed,
        )
        print(f"[ssft-{embedder.kind}] train={len(train_paths)} val={len(val_paths)}")

    num_clusters = infer_deepcluster_cluster_count(len(train_paths), cfg.ssft_deepcluster_clusters)
    hidden_size = int(embedder.model.config.hidden_size)
    cluster_head = ClusterHead(hidden_size=hidden_size, num_clusters=num_clusters).to(device)
    optimizer = AdamW(
        list(embedder.model.parameters()) + list(cluster_head.parameters()),
        lr=cfg.ssft_lr,
        weight_decay=cfg.ssft_weight_decay,
    )
    amp_dtype = resolve_amp_dtype(cfg.amp_dtype)
    # bf16 carries fp32's exponent range, so loss scaling is only needed for fp16.
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)
    history: list[dict[str, float]] = []
    checkpoint_path = run_dir / f"ssft_{embedder.kind}.pt"
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, cfg.ssft_epochs + 1):
        kept_train_paths, train_labels, kept_val_paths, val_labels = build_deepcluster_assignments(
            embedder.model,
            embedder.processor,
            train_paths,
            val_paths,
            cfg,
            device,
            epoch,
        )
        train_loader = build_deepcluster_loader(
            embedder.processor,
            kept_train_paths,
            train_labels,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
            augment=True,
        )
        val_loader = None
        if len(kept_val_paths) > 0:
            val_loader = build_deepcluster_loader(
                embedder.processor,
                kept_val_paths,
                val_labels,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                shuffle=False,
                augment=False,
            )

        embedder.model.train()
        cluster_head.train()
        epoch_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"ssft {embedder.kind} epoch {epoch}", leave=False):
            if batch is None:
                continue
            inputs = move_batch_to_device(batch["inputs"], device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_dtype is not None
                else nullcontext()
            )
            with amp_context:
                features = pooled_output(embedder.model(**inputs)).float()
                logits = cluster_head(features)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.item()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss = float("nan")
        if val_loader is not None:
            val_loss = evaluate_deepcluster(embedder.model, cluster_head, val_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_pseudo_ce_loss": mean_loss,
                "val_pseudo_ce_loss": val_loss,
                "num_clusters": float(num_clusters),
            }
        )
        print(
            f"[ssft-{embedder.kind}] epoch={epoch} "
            f"train_loss={mean_loss:.4f} val_loss={val_loss:.4f} "
            f"clusters={num_clusters}"
        )

        should_save = False
        if val_loader is not None:
            if val_loss < best_val_loss - cfg.ssft_early_stopping_min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                should_save = True
            else:
                epochs_without_improvement += 1
        else:
            if mean_loss < best_train_loss:
                best_train_loss = mean_loss
                best_epoch = epoch
                should_save = True

        if should_save:
            torch.save(
                {
                    "kind": embedder.kind,
                    "model_id": embedder.model_id,
                    "ssft_epochs": cfg.ssft_epochs,
                    "ssft_lr": cfg.ssft_lr,
                    "ssft_weight_decay": cfg.ssft_weight_decay,
                    "ssft_objective": cfg.ssft_objective,
                    "ssft_deepcluster_clusters": num_clusters,
                    "best_epoch": float(best_epoch),
                    "best_val_loss": float(best_val_loss) if val_loader is not None else float("nan"),
                    "history": history,
                    "model_state_dict": embedder.model.state_dict(),
                    "cluster_head_state_dict": cluster_head.state_dict(),
                },
                checkpoint_path,
            )

        if val_loader is not None and cfg.ssft_early_stopping_patience > 0:
            if epochs_without_improvement >= cfg.ssft_early_stopping_patience:
                print(
                    f"[ssft-{embedder.kind}] early stopping at epoch={epoch} "
                    f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})"
                )
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    embedder.model.load_state_dict(checkpoint["model_state_dict"])
    embedder.model.eval()
    return {
        "checkpoint_path": str(checkpoint_path),
        "history": history,
        "best_epoch": float(best_epoch) if best_epoch else float("nan"),
        "best_val_loss": float(best_val_loss) if len(val_paths) > 0 else float("nan"),
        "num_clusters": float(num_clusters),
    }


def run_ssft(
    embedder: FrozenImageEmbedder,
    discovery_paths: list[str],
    cfg: RunConfig,
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    if not discovery_paths:
        raise ValueError("SSFT requested, but discovery set is empty.")

    if cfg.ssft_objective == "deepcluster":
        return run_deepcluster_ssft(embedder, discovery_paths, cfg, device, run_dir)

    train_paths = discovery_paths
    val_paths: list[str] = []
    if cfg.ssft_early_stopping_patience > 0:
        train_paths, val_paths = split_paths_for_ssft(
            discovery_paths,
            val_size=cfg.ssft_val_size,
            random_seed=cfg.random_seed,
        )
        print(f"[ssft-{embedder.kind}] train={len(train_paths)} val={len(val_paths)}")

    data_loader = DataLoader(
        PathDataset(train_paths),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=SSFTCollator(embedder.processor),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_paths:
        val_loader = DataLoader(
            PathDataset(val_paths),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=SSFTCollator(embedder.processor),
            pin_memory=torch.cuda.is_available(),
        )
    hidden_size = int(embedder.model.config.hidden_size)
    projector = ProjectionHead(hidden_size=hidden_size, projection_dim=cfg.ssft_projection_dim).to(device)
    optimizer = AdamW(
        list(embedder.model.parameters()) + list(projector.parameters()),
        lr=cfg.ssft_lr,
        weight_decay=cfg.ssft_weight_decay,
    )
    amp_dtype = resolve_amp_dtype(cfg.amp_dtype)
    # bf16 carries fp32's exponent range, so loss scaling is only needed for fp16.
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)
    history: list[dict[str, float]] = []
    checkpoint_path = run_dir / f"ssft_{embedder.kind}.pt"
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    embedder.model.train()
    projector.train()

    for epoch in range(1, cfg.ssft_epochs + 1):
        epoch_losses: list[float] = []
        for batch in tqdm(data_loader, desc=f"ssft {embedder.kind} epoch {epoch}", leave=False):
            if batch is None:
                continue

            view1 = move_batch_to_device(batch["view1"], device)
            view2 = move_batch_to_device(batch["view2"], device)
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_dtype is not None
                else nullcontext()
            )
            with amp_context:
                features1 = pooled_output(embedder.model(**view1)).float()
                features2 = pooled_output(embedder.model(**view2)).float()
                z1 = projector(features1)
                z2 = projector(features2)
                loss = compute_ssft_loss(
                    z1,
                    z2,
                    objective=cfg.ssft_objective,
                    temperature=cfg.ssft_temperature,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.item()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss = float("nan")
        if val_loader is not None:
            val_loss = evaluate_ssft(
                embedder.model,
                projector,
                val_loader,
                device=device,
                objective=cfg.ssft_objective,
                temperature=cfg.ssft_temperature,
            )
            embedder.model.train()
            projector.train()
        history.append(
            {
                "epoch": float(epoch),
                "train_contrastive_loss": mean_loss,
                "val_contrastive_loss": val_loss,
            }
        )
        print(
            f"[ssft-{embedder.kind}] epoch={epoch} "
            f"train_loss={mean_loss:.4f} val_loss={val_loss:.4f}"
        )

        should_save = False
        if val_loader is not None:
            if val_loss < best_val_loss - cfg.ssft_early_stopping_min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                should_save = True
            else:
                epochs_without_improvement += 1
        else:
            if mean_loss < best_train_loss:
                best_train_loss = mean_loss
                best_epoch = epoch
                should_save = True

        if should_save:
            torch.save(
                {
                    "kind": embedder.kind,
                    "model_id": embedder.model_id,
                    "ssft_epochs": cfg.ssft_epochs,
                    "ssft_lr": cfg.ssft_lr,
                    "ssft_weight_decay": cfg.ssft_weight_decay,
                    "ssft_objective": cfg.ssft_objective,
                    "ssft_temperature": cfg.ssft_temperature,
                    "best_epoch": float(best_epoch),
                    "best_val_loss": float(best_val_loss) if val_loader is not None else float("nan"),
                    "history": history,
                    "model_state_dict": embedder.model.state_dict(),
                },
                checkpoint_path,
            )

        if val_loader is not None and cfg.ssft_early_stopping_patience > 0:
            if epochs_without_improvement >= cfg.ssft_early_stopping_patience:
                print(
                    f"[ssft-{embedder.kind}] early stopping at epoch={epoch} "
                    f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})"
                )
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    embedder.model.load_state_dict(checkpoint["model_state_dict"])
    embedder.model.eval()
    return {
        "checkpoint_path": str(checkpoint_path),
        "history": history,
        "best_epoch": float(best_epoch) if best_epoch else float("nan"),
        "best_val_loss": float(best_val_loss) if val_loader is not None else float("nan"),
    }


def load_existing_ssft_checkpoint(
    embedder: FrozenImageEmbedder,
    checkpoint_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_dir / f"ssft_{embedder.kind}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Expected SSFT checkpoint for {embedder.kind} at {checkpoint_path}, but it does not exist."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain `model_state_dict`.")

    embedder.model.load_state_dict(state_dict)
    embedder.model.eval()

    return {
        "checkpoint_path": str(checkpoint_path),
        "loaded_existing": True,
        "ssft_objective": checkpoint.get("ssft_objective"),
        "best_epoch": float(checkpoint.get("best_epoch", float("nan"))),
        "best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "history": checkpoint.get("history", []),
    }


def align_dataframe(df: pd.DataFrame, kept_paths: list[str]) -> pd.DataFrame:
    path_to_index = {path: idx for idx, path in enumerate(df["image_path"].tolist())}
    return df.iloc[[path_to_index[path] for path in kept_paths]].reset_index(drop=True)


def fused_embedding(siglip: np.ndarray, dino: np.ndarray, alpha: float) -> np.ndarray:
    left = np.sqrt(alpha) * siglip
    right = np.sqrt(1.0 - alpha) * dino
    return l2_normalize(np.concatenate([left, right], axis=1))


# build_reducer / fit_clusterer / compute_centroids / assign_to_centroids /
# refine_clusters now live in gpu_backend and run on the GPU. They are imported at the
# top of this module and re-exported via __all__ for the sibling eval scripts.


def majority_vote_mapping(discovery_df: pd.DataFrame, cluster_labels: np.ndarray) -> dict[int, str]:
    mapping: dict[int, str] = {}
    labeled_mask = discovery_df["source"].eq("imgflip_train").to_numpy()
    labeled_templates = discovery_df.loc[labeled_mask, "template"].to_numpy()
    labeled_clusters = cluster_labels[labeled_mask]

    for cluster_id in sorted(set(labeled_clusters.tolist())):
        if cluster_id < 0:
            continue
        cluster_templates = labeled_templates[labeled_clusters == cluster_id]
        if cluster_templates.size == 0:
            continue
        values, counts = np.unique(cluster_templates, return_counts=True)
        mapping[int(cluster_id)] = str(values[counts.argmax()])
    return mapping


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


def parse_int_list(text: str) -> tuple[int, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer seed.")
    return tuple(int(value) for value in values)


def build_run_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        imgflip_root=None if args.imgflip_root is None else Path(args.imgflip_root).expanduser().resolve(),
        train_parquet=None if args.train_parquet is None else Path(args.train_parquet).expanduser().resolve(),
        test_parquet=None if args.test_parquet is None else Path(args.test_parquet).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        train_size=args.train_size,
        min_images_per_class=args.min_images_per_class,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        siglip_model_id=args.siglip_model_id,
        dino_model_id=args.dino_model_id,
        alpha=args.alpha,
        reducer=args.reducer,
        reducer_dim=args.reducer_dim,
        cluster_method=args.cluster_method,
        kmeans_clusters=args.kmeans_clusters,
        refinement_steps=args.refinement_steps,
        max_templates=args.max_templates,
        max_images_per_template=args.max_images_per_template,
        max_unlabeled_images=args.max_unlabeled_images,
        discovery_extra_roots=args.discovery_extra_roots,
        max_corpus_images=args.max_corpus_images,
        use_ssft=args.use_ssft,
        ssft_checkpoint_dir=(
            None if args.ssft_checkpoint_dir is None else Path(args.ssft_checkpoint_dir).expanduser().resolve()
        ),
        ssft_epochs=args.ssft_epochs,
        ssft_lr=args.ssft_lr,
        ssft_weight_decay=args.ssft_weight_decay,
        ssft_temperature=args.ssft_temperature,
        ssft_projection_dim=args.ssft_projection_dim,
        ssft_objective=args.ssft_objective,
        ssft_deepcluster_clusters=args.ssft_deepcluster_clusters,
        ssft_early_stopping_patience=args.ssft_early_stopping_patience,
        ssft_early_stopping_min_delta=args.ssft_early_stopping_min_delta,
        ssft_val_size=args.ssft_val_size,
        amp_dtype=args.amp_dtype,
        encode_dtype=args.encode_dtype,
        io_workers=args.io_workers,
        open_set_dir=None if args.open_set_dir is None else Path(args.open_set_dir).expanduser().resolve(),
        open_set_reject_label=args.open_set_reject_label,
        open_set_labels=None if args.open_set_labels is None else Path(args.open_set_labels).expanduser().resolve(),
        non_meme_root=None if args.non_meme_root is None else Path(args.non_meme_root).expanduser().resolve(),
        max_non_meme_images=args.max_non_meme_images,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unsupervised evaluation for frozen SigLIP + DINO using cluster discovery on "
            "ImgFlip 80% plus optional unlabeled samples drawn from the ImgFlip training split, "
            "then majority-vote cluster naming."
        )
    )
    parser.add_argument("--imgflip-root", default=None, help="Labeled ImgFlip folder-of-folders root.")
    parser.add_argument("--train-parquet", default=None, help="Optional fixed train split parquet.")
    parser.add_argument("--test-parquet", default=None, help="Optional fixed test split parquet.")
    parser.add_argument("--output-dir", default="SEED/runs/seed_unsupervised_eval")
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--min-images-per-class", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=None,
        help="Optional comma-separated list of seeds to run in batch mode, e.g. 42,43,44.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
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
        "--amp-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Autocast dtype for SSFT training. bf16 needs no loss scaling.",
    )
    parser.add_argument(
        "--encode-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="fp32",
        help=(
            "Autocast dtype for frozen-backbone embedding. Defaults to fp32: bf16 is "
            "roughly 2x faster but perturbs the embeddings that clustering consumes, "
            "so opt in deliberately rather than mixing it into existing baselines."
        ),
    )
    parser.add_argument("--siglip-model-id", default="google/siglip2-base-patch16-224")
    parser.add_argument("--dino-model-id", default="facebook/dinov2-base")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--reducer", choices=["none", "pca", "umap"], default="pca")
    parser.add_argument("--reducer-dim", type=int, default=128)
    parser.add_argument("--cluster-method", choices=["kmeans", "hdbscan"], default="hdbscan")
    parser.add_argument("--kmeans-clusters", type=int, default=None)
    parser.add_argument("--refinement-steps", type=int, default=1)
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--max-images-per-template", type=int, default=None)
    parser.add_argument(
        "--max-unlabeled-images",
        type=int,
        default=None,
        help="Randomly sample this many training images from ImgFlip and treat them as unlabeled discovery data.",
    )
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
        "--use_SSFT",
        "--use-ssft",
        dest="use_ssft",
        action="store_true",
        help="Run self-supervised fine-tuning on the discovery split before clustering.",
    )
    parser.add_argument(
        "--ssft-checkpoint-dir",
        default=None,
        help=(
            "Optional directory containing existing SSFT checkpoints named "
            "`ssft_siglip.pt` and `ssft_dino.pt`. When provided, the script "
            "loads these adapted backbones instead of retraining them."
        ),
    )
    parser.add_argument("--ssft-epochs", type=int, default=2)
    parser.add_argument("--ssft-lr", type=float, default=1e-5)
    parser.add_argument("--ssft-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--ssft-objective",
        choices=["ntxent", "vicreg", "deepcluster"],
        default="ntxent",
        help="Self-supervised objective used during SSFT.",
    )
    parser.add_argument(
        "--ssft-deepcluster-clusters",
        type=int,
        default=None,
        help="Number of pseudo-clusters used when --ssft-objective=deepcluster. Defaults to a heuristic.",
    )
    parser.add_argument("--ssft-temperature", type=float, default=0.1)
    parser.add_argument("--ssft-projection-dim", type=int, default=256)
    parser.add_argument(
        "--ssft-early-stopping-patience",
        type=int,
        default=2,
        help="Stop SSFT when validation contrastive loss does not improve for this many epochs. Set 0 to disable.",
    )
    parser.add_argument(
        "--ssft-early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss improvement required to reset SSFT early stopping.",
    )
    parser.add_argument(
        "--ssft-val-size",
        type=float,
        default=0.10,
        help="Holdout fraction used for SSFT early stopping on the discovery split.",
    )
    add_non_meme_args(parser)
    add_open_set_args(parser)
    return parser


def run_once(cfg: RunConfig, run_dir: Path) -> dict[str, Any]:
    set_seed(cfg.random_seed)
    run_started_at = now_iso()
    start_perf = time.perf_counter()
    device = pick_device()
    configure_torch_backends()
    encode_dtype = resolve_amp_dtype(cfg.encode_dtype)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device} ({torch.cuda.get_device_name(0)})")
    print(f"encode_dtype={cfg.encode_dtype} ssft_amp_dtype={cfg.amp_dtype}")
    print(f"run_dir={run_dir}")

    train_parquet = getattr(cfg, "train_parquet", None)
    test_parquet = getattr(cfg, "test_parquet", None)
    if train_parquet is not None and test_parquet is not None:
        train_imgflip_df, test_imgflip_df = load_split_rows(str(train_parquet), str(test_parquet))
        imgflip_df = pd.concat([train_imgflip_df, test_imgflip_df], ignore_index=True)
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

        train_imgflip_df, test_imgflip_df = train_test_split(
            imgflip_df,
            train_size=cfg.train_size,
            stratify=imgflip_df["template"],
            random_state=cfg.random_seed,
        )
        train_imgflip_df = train_imgflip_df.reset_index(drop=True)
        test_imgflip_df = test_imgflip_df.reset_index(drop=True)
        train_imgflip_df["source"] = "imgflip_train"
        test_imgflip_df["source"] = "imgflip_test"

    train_imgflip_df, unlabeled_imgflip_df = split_labeled_and_unlabeled_from_train(
        train_imgflip_df,
        max_unlabeled_images=cfg.max_unlabeled_images,
        random_seed=cfg.random_seed,
    )
    non_meme_summary = None
    if cfg.non_meme_root is not None:
        # Labelled like the ImgFlip rows, so majority_vote_mapping can name a cluster
        # "Non-Meme". The method still decides entirely by clustering.
        non_meme_df = collect_non_meme_rows(
            cfg.non_meme_root, exclude_dir=cfg.open_set_dir,
            max_images=cfg.max_non_meme_images, random_seed=cfg.random_seed,
        )
        non_meme_df["source"] = "imgflip_train"   # counted by the majority vote
        train_imgflip_df, _, non_meme_summary = add_non_meme_class(
            train_imgflip_df, None, non_meme_df, cfg.train_size, cfg.random_seed
        )
    discovery_parts = [train_imgflip_df]
    unlabeled_count = len(unlabeled_imgflip_df)
    if unlabeled_count:
        discovery_parts.append(unlabeled_imgflip_df)

    # External unlabeled corpora (e.g. social media). Only ImgFlip rows carry
    # source == "imgflip_train", so majority_vote_mapping still names clusters from labels alone.
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
    print(
        f"discovery_images={len(discovery_df)} (imgflip_train={len(train_imgflip_df)} "
        f"imgflip_unlabeled={unlabeled_count} corpus={corpus_count})"
    )

    siglip = FrozenImageEmbedder(
        cfg.siglip_model_id, "siglip", device, cfg.batch_size, cfg.num_workers, encode_dtype
    )
    dino = FrozenImageEmbedder(
        cfg.dino_model_id, "dino", device, cfg.batch_size, cfg.num_workers, encode_dtype
    )

    discovery_paths = discovery_df["image_path"].tolist()
    test_paths = test_imgflip_df["image_path"].tolist()
    ssft_summary: dict[str, Any] | None = None

    if cfg.ssft_checkpoint_dir is not None:
        print(f"Loading existing SSFT checkpoints from {cfg.ssft_checkpoint_dir}")
        ssft_summary = {
            "enabled": True,
            "loaded_existing": True,
            "discovery_images": int(len(discovery_paths)),
            "checkpoint_dir": str(cfg.ssft_checkpoint_dir),
            "siglip": load_existing_ssft_checkpoint(siglip, cfg.ssft_checkpoint_dir, device),
            "dino": load_existing_ssft_checkpoint(dino, cfg.ssft_checkpoint_dir, device),
        }
    elif cfg.use_ssft:
        ssft_dir = run_dir / "ssft"
        ssft_dir.mkdir(parents=True, exist_ok=True)
        print(
            "Running SSFT on discovery images: "
            f"objective={cfg.ssft_objective} epochs={cfg.ssft_epochs} "
            f"lr={cfg.ssft_lr:g} temp={cfg.ssft_temperature:g}"
        )
        ssft_summary = {
            "enabled": True,
            "loaded_existing": False,
            "discovery_images": int(len(discovery_paths)),
            "siglip": run_ssft(siglip, discovery_paths, cfg, device, ssft_dir),
            "dino": run_ssft(dino, discovery_paths, cfg, device, ssft_dir),
        }
    else:
        ssft_summary = {"enabled": False, "loaded_existing": False}

    # One pass per split feeds both backbones, so each image is read and decoded once
    # instead of twice. The two feature matrices come back row-aligned, which also
    # removes the old cross-check between two separately-built kept-path lists.
    discovery_siglip, discovery_dino, kept_discovery_paths = encode_dual_backbones(
        siglip.model,
        siglip.processor,
        dino.model,
        dino.processor,
        discovery_paths,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        desc="encode discovery siglip+dino",
        amp_dtype=encode_dtype,
    )
    test_siglip, test_dino, kept_test_paths = encode_dual_backbones(
        siglip.model,
        siglip.processor,
        dino.model,
        dino.processor,
        test_paths,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        desc="encode test siglip+dino",
        amp_dtype=encode_dtype,
    )

    discovery_df = align_dataframe(discovery_df, kept_discovery_paths)
    test_imgflip_df = align_dataframe(test_imgflip_df, kept_test_paths)

    discovery_fused = fused_embedding(discovery_siglip, discovery_dino, cfg.alpha)
    test_fused = fused_embedding(test_siglip, test_dino, cfg.alpha)

    reducer = build_reducer(cfg.reducer, cfg.reducer_dim, cfg.random_seed)
    discovery_reduced = reducer.fit_transform(discovery_fused)
    test_reduced = reducer.transform(test_fused)
    discovery_reduced = l2_normalize(np.asarray(discovery_reduced, dtype=np.float32))
    test_reduced = l2_normalize(np.asarray(test_reduced, dtype=np.float32))

    num_clusters = cfg.kmeans_clusters or train_imgflip_df["template"].nunique()
    clusterer, initial_labels = fit_clusterer(
        method=cfg.cluster_method,
        x=discovery_reduced,
        num_clusters=num_clusters,
        random_seed=cfg.random_seed,
    )

    refined_labels, centroids, centroid_ids = refine_clusters(
        discovery_reduced,
        initial_labels,
        steps=cfg.refinement_steps,
        device=device,
    )
    cluster_to_template = majority_vote_mapping(discovery_df, refined_labels)
    # Clusters made purely of unlabeled images have no template; dropping them keeps every
    # test prediction a real template name.
    all_centroids, all_centroid_ids = centroids, centroid_ids
    # Representative members of every cluster, named or not: the evidence an annotator
    # needs to judge an image that landed in a cluster ImgFlip never reached.
    _, cluster_examples_summary = write_cluster_examples(
        run_dir, discovery_df, refined_labels, cluster_to_template,
        member_scores=cosine_to_own_centroid(discovery_reduced, refined_labels, all_centroids, all_centroid_ids),
    )
    centroids, centroid_ids = restrict_to_labeled_clusters(centroids, centroid_ids, cluster_to_template)
    test_cluster_ids = assign_to_centroids(test_reduced, centroids, centroid_ids, device)
    test_pred_templates = np.array([cluster_to_template.get(int(cluster_id), "UNKNOWN_CLUSTER") for cluster_id in test_cluster_ids])
    y_true = test_imgflip_df["template"].to_numpy()

    metrics = compute_metrics(y_true, test_pred_templates)

    predictions_df = test_imgflip_df.copy()
    predictions_df["pred_cluster"] = test_cluster_ids
    predictions_df["pred_template"] = test_pred_templates
    predictions_df["correct"] = predictions_df["pred_template"].to_numpy() == predictions_df["template"].to_numpy()
    predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)

    open_set_summary = None
    if cfg.open_set_dir is not None:
        open_df = collect_open_set_rows(cfg.open_set_dir)
        open_siglip, open_dino, kept_open_paths = encode_dual_backbones(
            siglip.model,
            siglip.processor,
            dino.model,
            dino.processor,
            open_df["image_path"].tolist(),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            device=device,
            desc="encode open-set siglip+dino",
            amp_dtype=encode_dtype,
        )
        open_df = align_dataframe(open_df, kept_open_paths)
        open_reduced = reducer.transform(fused_embedding(open_siglip, open_dino, cfg.alpha))
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

    run_metadata = {
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
            "imgflip_train_images": int(len(train_imgflip_df)),
            "imgflip_test_images": int(len(test_imgflip_df)),
            "unlabeled_images_in_discovery": int(unlabeled_count),
            "unlabeled_corpus_images": int(corpus_count),
            "discovery_images_total": int(len(discovery_df)),
            "small_class_filtering": filtering_summary,
        },
        "clustering": {
            "cluster_method": cfg.cluster_method,
            "requested_kmeans_clusters": int(num_clusters),
            "discovered_cluster_count": int(len({int(c) for c in refined_labels if c >= 0})),
            "labeled_cluster_count": int(len(centroid_ids)),
            "mapped_cluster_count": int(len(cluster_to_template)),
            "unknown_cluster_predictions": int((test_pred_templates == "UNKNOWN_CLUSTER").sum()),
        },
        "ssft": ssft_summary,
        "test_metrics": metrics,
    }
    test_metrics_payload = {
        "test_metrics": metrics,
        "ssft": ssft_summary,
        "run_dir": str(run_dir),
        "cluster_method": cfg.cluster_method,
        "reducer": cfg.reducer,
        "alpha": cfg.alpha,
    }
    summary_row = {
        "run_dir": str(run_dir),
        "seed": int(cfg.random_seed),
        "ssft_enabled": bool(ssft_summary.get("enabled", False)) if isinstance(ssft_summary, dict) else False,
        "loaded_existing_ssft": bool(ssft_summary.get("loaded_existing", False)) if isinstance(ssft_summary, dict) else False,
        "cluster_method": cfg.cluster_method,
        "reducer": cfg.reducer,
        "alpha": cfg.alpha,
        "train_images": int(len(train_imgflip_df)),
        "test_images": int(len(test_imgflip_df)),
        "discovery_images_total": int(len(discovery_df)),
        "discovered_cluster_count": int(len(centroid_ids)),
        "mapped_cluster_count": int(len(cluster_to_template)),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "mcc": float(metrics["mcc"]),
        "cohen_kappa": float(metrics["cohen_kappa"]),
    }
    if open_set_summary is not None and cfg.open_set_labels is not None:
        open_set_summary["metrics"] = score_open_set_predictions(run_dir, cfg.open_set_labels)
    run_metadata["open_set"] = open_set_summary
    timing = finalize_run_timing(run_metadata, summary_row, run_started_at, start_perf)
    test_metrics_payload["timing"] = timing
    (run_dir / "run_config.json").write_text(json.dumps(run_metadata, indent=2))
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics_payload, indent=2))
    pd.DataFrame([summary_row]).to_csv(run_dir / "metrics_summary.csv", index=False)

    print(f"imgflip_train_images={len(train_imgflip_df)} imgflip_test_images={len(test_imgflip_df)}")
    print(f"discovery_images_total={len(discovery_df)}")
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
    args = make_parser().parse_args()
    cfg = build_run_config(args)
    seed_values = args.seeds if args.seeds is not None else (cfg.random_seed,)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if len(seed_values) == 1:
        single_cfg = RunConfig(**resolve_seed_paths({**asdict(cfg), "random_seed": int(seed_values[0])}, seed_values[0]))
        run_dir = cfg.output_dir / timestamp
        run_once(single_cfg, run_dir)
        return

    batch_dir = cfg.output_dir / f"{timestamp}_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_rows: list[dict[str, Any]] = []

    for seed in seed_values:
        print(f"\n=== Running seed {seed} ===")
        seed_cfg = RunConfig(**resolve_seed_paths({**asdict(cfg), "random_seed": int(seed)}, seed))
        seed_run_dir = batch_dir / f"seed_{int(seed)}"
        batch_rows.append(run_once(seed_cfg, seed_run_dir))

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
