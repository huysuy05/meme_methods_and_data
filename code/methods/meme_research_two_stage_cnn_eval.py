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
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from meme_research_eval_utils import (
    OPEN_SET_NON_MEME,
    add_open_set_args,
    collect_open_set_rows,
    score_open_set_predictions,
    write_open_set_predictions,
    VALID_EXTS,
    compute_metrics_with_rejection,
    drop_small_classes,
    filter_valid_image_rows,
    finalize_run_timing,
    load_dataset_rows,
    load_rgb_image,
    load_split_rows,
    now_iso,
    parse_int_list,
    pick_device,
    resolve_seed_paths,
    set_seed,
    stratified_split,
    summarize_batch_timings,
)


REJECT_TOKEN = "__REJECTED_BY_STAGE1__"
NORMALIZE_MEAN = [0.5325, 0.4980, 0.4715]
NORMALIZE_STD = [0.3409, 0.3384, 0.3465]


@dataclass
class RunConfig:
    dataset_root: Path | None
    parquet_path: Path | None
    train_parquet: Path | None
    test_parquet: Path | None
    image_root: Path | None
    negative_root: Path | None
    output_dir: Path
    train_size: float = 0.80
    val_size: float = 0.10
    min_images_per_class: int = 7
    random_seed: int = 42
    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    epochs: int = 10
    patience: int = 2
    freeze_backbone: bool = True
    max_templates: int | None = None
    max_images_per_template: int | None = None
    path_prefix_from: str | None = None
    path_prefix_to: str | None = None
    seeds: tuple[int, ...] | None = None
    checkpoint_dir: Path | None = None
    open_set_dir: Path | None = None
    open_set_reject_label: str = OPEN_SET_NON_MEME
    open_set_labels: Path | None = None


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Closed-set ImgFlip evaluation for a standalone two-stage DenseNet-121 baseline. "
            "Stage 1 is a binary meme gate when a negative dataset is supplied; "
            "Stage 2 is a multiclass template classifier over ImgFlip positives."
        )
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--parquet-path", default=None)
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--test-parquet", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--negative-root", default=None)
    parser.add_argument("--output-dir", default="SEED/runs/meme_research_two_stage_cnn_eval")
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--min-images-per-class", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true")
    parser.add_argument("--train-backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=True)
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--max-images-per-template", type=int, default=None)
    parser.add_argument("--path-prefix-from", default=None)
    parser.add_argument("--path-prefix-to", default=None)
    parser.add_argument("--seeds", type=parse_int_list, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory of a finished run (e.g. .../seed_40) whose weights should be loaded instead of "
             "fine-tuning again. Skips training; the ImgFlip test split is still scored.",
    )
    add_open_set_args(parser, default_reject_label=OPEN_SET_NON_MEME)
    args = parser.parse_args()
    return RunConfig(
        dataset_root=None if args.dataset_root is None else Path(args.dataset_root).expanduser().resolve(),
        parquet_path=None if args.parquet_path is None else Path(args.parquet_path).expanduser().resolve(),
        train_parquet=None if args.train_parquet is None else Path(args.train_parquet).expanduser().resolve(),
        test_parquet=None if args.test_parquet is None else Path(args.test_parquet).expanduser().resolve(),
        image_root=None if args.image_root is None else Path(args.image_root).expanduser().resolve(),
        negative_root=None if args.negative_root is None else Path(args.negative_root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        train_size=args.train_size,
        val_size=args.val_size,
        min_images_per_class=args.min_images_per_class,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        freeze_backbone=args.freeze_backbone,
        max_templates=args.max_templates,
        max_images_per_template=args.max_images_per_template,
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
        seeds=args.seeds,
        checkpoint_dir=None if args.checkpoint_dir is None else Path(args.checkpoint_dir).expanduser().resolve(),
        open_set_dir=None if args.open_set_dir is None else Path(args.open_set_dir).expanduser().resolve(),
        open_set_reject_label=args.open_set_reject_label,
        open_set_labels=None if args.open_set_labels is None else Path(args.open_set_labels).expanduser().resolve(),
    )


def collect_negative_rows(root: Path) -> pd.DataFrame:
    image_paths = sorted([path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VALID_EXTS])
    if not image_paths:
        raise ValueError(f"No negative images found under {root}")
    return pd.DataFrame({
        "image_path": [str(path) for path in image_paths],
        "template": ["NEGATIVE"] * len(image_paths),
        "source": ["negative"] * len(image_paths),
    })


class PositiveTemplateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict[str, int], transform):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.df.iloc[index]
        image = load_rgb_image(row["image_path"])
        if image is None:
            raise ValueError(f"Unreadable image slipped through filtering: {row['image_path']}")
        return self.transform(image), self.label_to_idx[row["template"]], row["image_path"]


class BinaryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float, str]:
        row = self.df.iloc[index]
        image = load_rgb_image(row["image_path"])
        if image is None:
            raise ValueError(f"Unreadable image slipped through filtering: {row['image_path']}")
        return self.transform(image), float(row["binary_label"]), row["image_path"]


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(0, 10)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    return train_transform, eval_transform


class DenseNetClassifier(nn.Module):
    def __init__(self, out_dim: int, freeze_backbone: bool) -> None:
        super().__init__()
        base = models.densenet121(weights="DEFAULT")
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(1024, out_dim)
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    patience: int,
    binary: bool,
) -> tuple[dict[str, torch.Tensor], float, int, list[dict[str, float]]]:
    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, targets, _ in tqdm(train_loader, desc=f"train epoch {epoch + 1}/{epochs}", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if binary:
                logits = logits.squeeze(1)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.shape[0]
            seen += int(images.shape[0])

        val_loss, _ = evaluate_binary(model, val_loader, device, loss_fn) if binary else evaluate_multiclass(model, val_loader, device, loss_fn)
        row = {
            "epoch": float(epoch + 1),
            "train_loss": float(running_loss / max(seen, 1)),
            "val_loss": float(val_loss),
        }
        history.append(row)
        print(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    return best_state, best_val_loss, best_epoch, history


def evaluate_multiclass(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, dict[str, np.ndarray]]:
    model.eval()
    losses: list[float] = []
    probs_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    paths_all: list[str] = []
    with torch.inference_mode():
        for images, targets, paths in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = loss_fn(logits, targets)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            losses.append(float(loss.item()) * images.shape[0])
            probs_all.append(probs.cpu().numpy())
            preds_all.append(preds.cpu().numpy())
            targets_all.append(targets.cpu().numpy())
            paths_all.extend(paths)
    total = max(len(paths_all), 1)
    return float(sum(losses) / total), {
        "probs": np.concatenate(probs_all, axis=0),
        "preds": np.concatenate(preds_all, axis=0),
        "targets": np.concatenate(targets_all, axis=0),
        "paths": np.array(paths_all, dtype=object),
    }


def evaluate_binary(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, dict[str, np.ndarray]]:
    model.eval()
    losses: list[float] = []
    probs_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    paths_all: list[str] = []
    with torch.inference_mode():
        for images, targets, paths in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images).squeeze(1)
            loss = loss_fn(logits, targets)
            probs = torch.sigmoid(logits)
            losses.append(float(loss.item()) * images.shape[0])
            probs_all.append(probs.cpu().numpy())
            targets_all.append(targets.cpu().numpy())
            paths_all.extend(paths)
    total = max(len(paths_all), 1)
    return float(sum(losses) / total), {
        "probs": np.concatenate(probs_all, axis=0),
        "targets": np.concatenate(targets_all, axis=0),
        "paths": np.array(paths_all, dtype=object),
    }


def select_best_threshold(
    targets: np.ndarray,
    probs: np.ndarray,
) -> tuple[float, float, pd.DataFrame]:
    """Pick the stage-1 meme/non-meme cut-off by maximising G-mean on the validation set.

    Returns the chosen threshold, its G-mean, and the full ROC sweep, so the
    operating point can be reported alongside every alternative rather than on its
    own (the rNN baselines expose the same information via ``val_search.csv``).
    """
    fpr, tpr, thresholds = roc_curve(targets, probs)
    tnr = 1.0 - fpr
    gmeans = np.sqrt(tpr * tnr)
    best_idx = int(np.argmax(gmeans))
    sweep = pd.DataFrame(
        {
            "threshold": thresholds,
            "tpr": tpr,
            "fpr": fpr,
            "tnr": tnr,
            "gmean": gmeans,
            "selected": np.arange(len(thresholds)) == best_idx,
        }
    )
    return float(thresholds[best_idx]), float(gmeans[best_idx]), sweep


def load_torch_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class PathOnlyDataset(Dataset):
    """Unlabeled images, for scoring a folder that has no template column."""

    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        image = load_rgb_image(path)
        if image is None:
            raise ValueError(f"Unreadable image slipped through filtering: {path}")
        return self.transform(image), path


def predict_multiclass_paths(
    model: nn.Module,
    paths: list[str],
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage-2 argmax class index and softmax confidence for each path, in input order."""
    loader = make_loader(PathOnlyDataset(paths, transform), batch_size, num_workers, False)
    preds: list[np.ndarray] = []
    confs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc="predict open-set stage 2"):
            probs = torch.softmax(model(images.to(device)), dim=1)
            confs.append(probs.max(dim=1).values.cpu().numpy())
            preds.append(probs.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)


def run_once(cfg: RunConfig, run_dir: Path) -> dict[str, float | int | str]:
    set_seed(cfg.random_seed)
    run_started_at = now_iso()
    start_perf = time.perf_counter()
    device = pick_device()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")
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
    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg.val_size,
        stratify=train_df["template"],
        random_state=cfg.random_seed,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    labels = sorted(train_df["template"].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    train_transform, eval_transform = build_transforms()

    multi_train_loader = make_loader(
        PositiveTemplateDataset(train_df, label_to_idx, train_transform),
        cfg.batch_size,
        cfg.num_workers,
        True,
    )
    multi_val_loader = make_loader(
        PositiveTemplateDataset(val_df, label_to_idx, eval_transform),
        cfg.eval_batch_size,
        cfg.num_workers,
        False,
    )
    multi_test_loader = make_loader(
        PositiveTemplateDataset(test_df, label_to_idx, eval_transform),
        cfg.eval_batch_size,
        cfg.num_workers,
        False,
    )

    multi_model = DenseNetClassifier(out_dim=len(label_to_idx), freeze_backbone=cfg.freeze_backbone).to(device)
    multi_loss = nn.CrossEntropyLoss()
    multi_opt = torch.optim.Adam(
        [param for param in multi_model.parameters() if param.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    if cfg.checkpoint_dir is None:
        multi_state, multi_best_val_loss, multi_best_epoch, multi_history = train_model(
            multi_model,
            multi_train_loader,
            multi_val_loader,
            device,
            multi_opt,
            multi_loss,
            cfg.epochs,
            cfg.patience,
            binary=False,
        )
    else:
        # --checkpoint-dir: reuse a finished run's stage-2 weights instead of fine-tuning again.
        multi_ckpt = cfg.checkpoint_dir / "stage2_multiclass_best.pt"
        if not multi_ckpt.exists():
            raise FileNotFoundError(f"Missing stage-2 checkpoint: {multi_ckpt}")
        print(f"reusing_stage2_checkpoint={multi_ckpt}")
        multi_state = load_torch_state(multi_ckpt)
        multi_best_val_loss, multi_best_epoch, multi_history = float("nan"), 0, []
    multi_model.load_state_dict(multi_state)
    torch.save(multi_state, run_dir / "stage2_multiclass_best.pt")

    stage1_ckpt = None if cfg.checkpoint_dir is None else cfg.checkpoint_dir / "stage1_binary_best.pt"
    reuse_stage1 = stage1_ckpt is not None and stage1_ckpt.exists()
    binary_stage_used = cfg.negative_root is not None or reuse_stage1
    binary_threshold = 0.0
    best_gmean = None
    binary_history: list[dict[str, float]] = []
    if reuse_stage1:
        # Reuse the stage-1 gate together with the threshold that run chose on validation,
        # so the gate behaves exactly as it did in the closed-set evaluation.
        print(f"reusing_stage1_checkpoint={stage1_ckpt}")
        binary_model = DenseNetClassifier(out_dim=1, freeze_backbone=cfg.freeze_backbone).to(device)
        binary_model.load_state_dict(load_torch_state(stage1_ckpt))
        binary_loss = nn.BCEWithLogitsLoss()
        source_stage1 = json.loads((cfg.checkpoint_dir / "run_config.json").read_text())["stage1_binary"]
        binary_threshold = float(source_stage1["threshold"])
        best_gmean = source_stage1.get("best_gmean")
        torch.save(binary_model.state_dict(), run_dir / "stage1_binary_best.pt")
        _, binary_test_outputs = evaluate_binary(
            binary_model,
            make_loader(BinaryDataset(test_df.assign(binary_label=1.0), eval_transform), cfg.eval_batch_size, cfg.num_workers, False),
            device,
            binary_loss,
        )
        meme_probs = binary_test_outputs["probs"]
    elif binary_stage_used:
        neg_df = collect_negative_rows(cfg.negative_root)
        neg_df = filter_valid_image_rows(neg_df)
        neg_train_df, neg_tmp_df = train_test_split(
            neg_df,
            train_size=cfg.train_size,
            random_state=cfg.random_seed,
        )
        neg_train_df, neg_val_df = train_test_split(
            neg_train_df,
            test_size=cfg.val_size,
            random_state=cfg.random_seed,
        )
        pos_train_bin = train_df.copy()
        pos_train_bin["binary_label"] = 1.0
        pos_val_bin = val_df.copy()
        pos_val_bin["binary_label"] = 1.0
        neg_train_df = neg_train_df.copy()
        neg_train_df["binary_label"] = 0.0
        neg_val_df = neg_val_df.copy()
        neg_val_df["binary_label"] = 0.0
        binary_train_df = pd.concat([pos_train_bin, neg_train_df], ignore_index=True)
        binary_val_df = pd.concat([pos_val_bin, neg_val_df], ignore_index=True)

        binary_train_loader = make_loader(
            BinaryDataset(binary_train_df, train_transform),
            cfg.batch_size,
            cfg.num_workers,
            True,
        )
        binary_val_loader = make_loader(
            BinaryDataset(binary_val_df, eval_transform),
            cfg.eval_batch_size,
            cfg.num_workers,
            False,
        )
        binary_model = DenseNetClassifier(out_dim=1, freeze_backbone=cfg.freeze_backbone).to(device)
        binary_loss = nn.BCEWithLogitsLoss()
        binary_opt = torch.optim.Adam(
            [param for param in binary_model.parameters() if param.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        binary_state, binary_best_val_loss, binary_best_epoch, binary_history = train_model(
            binary_model,
            binary_train_loader,
            binary_val_loader,
            device,
            binary_opt,
            binary_loss,
            cfg.epochs,
            cfg.patience,
            binary=True,
        )
        binary_model.load_state_dict(binary_state)
        torch.save(binary_state, run_dir / "stage1_binary_best.pt")
        _, binary_val_outputs = evaluate_binary(binary_model, binary_val_loader, device, binary_loss)
        binary_threshold, best_gmean, threshold_sweep = select_best_threshold(
            binary_val_outputs["targets"],
            binary_val_outputs["probs"],
        )
        threshold_sweep.to_csv(run_dir / "val_threshold_sweep.csv", index=False)
        _, binary_test_outputs = evaluate_binary(
            binary_model,
            make_loader(BinaryDataset(test_df.assign(binary_label=1.0), eval_transform), cfg.eval_batch_size, cfg.num_workers, False),
            device,
            binary_loss,
        )
        meme_probs = binary_test_outputs["probs"]
    else:
        meme_probs = np.ones(len(test_df), dtype=np.float32)

    _, multi_test_outputs = evaluate_multiclass(multi_model, multi_test_loader, device, multi_loss)
    multi_preds = multi_test_outputs["preds"]
    multi_conf = multi_test_outputs["probs"].max(axis=1)
    pred_templates = np.array([idx_to_label[int(pred)] for pred in multi_preds], dtype=object)
    pred_templates = np.where(meme_probs >= binary_threshold, pred_templates, REJECT_TOKEN)
    metrics = compute_metrics_with_rejection(test_df["template"].to_numpy(), pred_templates, REJECT_TOKEN)

    predictions_df = test_df.copy()
    predictions_df["meme_probability"] = meme_probs
    predictions_df["binary_threshold"] = binary_threshold
    predictions_df["stage2_confidence"] = multi_conf
    predictions_df["pred_template"] = pred_templates
    predictions_df["is_rejected"] = predictions_df["pred_template"].eq(REJECT_TOKEN)
    predictions_df["correct"] = predictions_df["pred_template"].to_numpy() == predictions_df["template"].to_numpy()
    predictions_df = predictions_df.loc[:, [
        "image_path",
        "template",
        "pred_template",
        "meme_probability",
        "binary_threshold",
        "stage2_confidence",
        "is_rejected",
        "correct",
        "source",
    ]]
    predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)

    open_set_summary = None
    if cfg.open_set_dir is not None:
        open_df = collect_open_set_rows(cfg.open_set_dir)
        open_paths = open_df["image_path"].tolist()
        if binary_stage_used:
            _, open_binary = evaluate_binary(
                binary_model,
                make_loader(BinaryDataset(open_df.assign(binary_label=0.0), eval_transform), cfg.eval_batch_size, cfg.num_workers, False),
                device,
                binary_loss,
            )
            open_meme_probs = open_binary["probs"]
        else:
            open_meme_probs = np.ones(len(open_df), dtype=np.float32)
        open_preds, open_conf = predict_multiclass_paths(
            multi_model, open_paths, eval_transform, device, cfg.eval_batch_size, cfg.num_workers
        )
        open_templates = np.array([idx_to_label[int(pred)] for pred in open_preds], dtype=object)
        # Stage 1 is a meme/non-meme gate, so its rejection is a Non-Meme call (the default
        # reject label for this script), not a template-free one. Stage 2 never rejects.
        open_labels = np.where(open_meme_probs >= binary_threshold, open_templates, cfg.open_set_reject_label)
        _, open_set_summary = write_open_set_predictions(
            run_dir,
            open_df,
            open_labels,
            extra_columns={
                "meme_probability": open_meme_probs,
                "binary_threshold": binary_threshold,
                "stage2_confidence": open_conf,
            },
        )

    metadata = {
        "config": {
            **asdict(cfg),
            "dataset_root": None if cfg.dataset_root is None else str(cfg.dataset_root),
            "parquet_path": None if cfg.parquet_path is None else str(cfg.parquet_path),
            "train_parquet": None if cfg.train_parquet is None else str(cfg.train_parquet),
            "test_parquet": None if cfg.test_parquet is None else str(cfg.test_parquet),
            "image_root": None if cfg.image_root is None else str(cfg.image_root),
            "negative_root": None if cfg.negative_root is None else str(cfg.negative_root),
            "output_dir": str(cfg.output_dir),
            "checkpoint_dir": None if cfg.checkpoint_dir is None else str(cfg.checkpoint_dir),
            "open_set_dir": None if cfg.open_set_dir is None else str(cfg.open_set_dir),
            "open_set_labels": None if cfg.open_set_labels is None else str(cfg.open_set_labels),
        },
        "dataset": {
            "images_total_after_filtering": int(len(df)),
            "classes_after_filtering": int(df["template"].nunique()),
            "train_images": int(len(train_df)),
            "val_images": int(len(val_df)),
            "test_images": int(len(test_df)),
            "small_class_filtering": filtering_summary,
        },
        "stage1_binary": {
            "used": bool(binary_stage_used),
            "threshold": float(binary_threshold),
            "best_gmean": None if best_gmean is None else float(best_gmean),
            "threshold_sweep_csv": None if not binary_stage_used else str(run_dir / "val_threshold_sweep.csv"),
            "history": binary_history,
        },
        "stage2_multiclass": {
            "best_epoch": int(multi_best_epoch),
            "best_val_loss": float(multi_best_val_loss),
            "history": multi_history,
        },
        "test_metrics": metrics,
    }
    summary_row = {
        "run_dir": str(run_dir),
        "seed": int(cfg.random_seed),
        "binary_stage_used": bool(binary_stage_used),
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

    print(f"binary_stage_used={binary_stage_used}")
    print(f"binary_threshold={binary_threshold:.6f}")
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
