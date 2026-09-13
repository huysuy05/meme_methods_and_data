#!/usr/bin/env python3
"""GPU backend for the unsupervised discovery pipelines.

Everything here replaces a CPU path that dominated wall-clock once the discovery
corpus grew past a few hundred thousand images:

1. ``build_reducer`` / ``fit_clusterer`` — RAPIDS cuML UMAP, PCA, KMeans and HDBSCAN
   in place of scikit-learn and ``umap-learn``. CPU UMAP on ~400k x 1536 embeddings
   runs for hours; cuML does it in minutes.

2. ``compute_centroids`` / ``assign_to_centroids`` — the centroid step used to loop
   over clusters in Python, one full pass over the data per cluster. Both are single
   fused GPU ops here.

3. ``encode_dual_backbones`` — SigLIP and DINO used to each make their own pass over
   the discovery set, so every image was read off disk and JPEG-decoded twice. One
   pass now feeds both models, and decoding happens in DataLoader workers instead of
   the main process, so the GPU stops idling on I/O.

This module is GPU-only by design: ``require_cuda`` raises rather than silently
falling back, so a misconfigured run fails immediately instead of quietly spending
a day on the CPU.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from meme_research_eval_utils import load_rgb_image

# cuML estimators return device arrays by default. Pinning the global output type to
# numpy keeps every call site downstream unchanged from the scikit-learn version.
try:
    import cuml
    from cuml.cluster import HDBSCAN as CumlHDBSCAN
    from cuml.cluster import KMeans as CumlKMeans
    from cuml.decomposition import PCA as CumlPCA
    from cuml.manifold import UMAP as CumlUMAP

    cuml.set_global_output_type("numpy")
    CUML_AVAILABLE = True
    CUML_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local install
    CUML_AVAILABLE = False
    CUML_IMPORT_ERROR = exc


def require_cuda() -> torch.device:
    """Return the CUDA device, or explain why the run cannot proceed."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "These pipelines are GPU-only, but torch.cuda.is_available() is False.\n"
            f"  torch={torch.__version__} built for CUDA {torch.version.cuda}\n"
            "Check that the installed torch matches the driver's CUDA version and the "
            "machine's CPU architecture (see requirements-gpu.txt)."
        )
    capability = torch.cuda.get_device_capability()
    arch = f"sm_{capability[0]}{capability[1]}"
    # A torch built without kernels for this GPU still reports it as available, then
    # dies deep inside the first real kernel launch. Comparing `arch` against
    # get_arch_list() is not a reliable test -- GB10 reports sm_121 and runs happily on
    # the sm_120 binaries via minor-version compatibility -- so just launch something.
    try:
        probe = torch.ones(64, 64, device="cuda")
        torch.mm(probe, probe).sum().item()
    except RuntimeError as exc:
        raise RuntimeError(
            f"Installed torch ({torch.__version__}, CUDA {torch.version.cuda}) cannot run "
            f"kernels on this GPU ({torch.cuda.get_device_name(0)}, {arch}).\n"
            f"  compiled architectures: {torch.cuda.get_arch_list()}\n"
            "Install a build that targets this GPU (see requirements-gpu.txt)."
        ) from exc
    return torch.device("cuda")


def require_cuml() -> None:
    if not CUML_AVAILABLE:
        raise ModuleNotFoundError(
            "RAPIDS cuML is required for GPU clustering but could not be imported: "
            f"{CUML_IMPORT_ERROR}\n"
            "Install cuml-cu13 from https://pypi.nvidia.com (see requirements-gpu.txt)."
        )


def configure_torch_backends() -> None:
    """Allow TF32 for fp32 matmuls.

    Applies to the backbone forward passes and the centroid matmuls only. The pHash
    DCT in ``gpu_hash_utils`` runs in float64, which TF32 does not touch, so hashes
    stay bit-identical to the scipy reference.
    """
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


def resolve_amp_dtype(name: str) -> torch.dtype | None:
    """Map a CLI dtype name to a torch dtype. ``fp32`` means "no autocast"."""
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}
    if name not in mapping:
        raise ValueError(f"Unsupported amp dtype={name}. Choose from {sorted(mapping)}.")
    return mapping[name]


# --------------------------------------------------------------------------------------
# Dimensionality reduction and clustering (cuML)
# --------------------------------------------------------------------------------------


class IdentityReducer:
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


def build_reducer(method: str, n_components: int, random_seed: int):
    """cuML PCA / UMAP. Same defaults as the scikit-learn and umap-learn versions."""
    method = method.lower()
    if method == "none":
        return IdentityReducer()

    require_cuml()
    if method == "pca":
        # cuML's PCA takes no random_state: the default "full" (exact) SVD solver is
        # deterministic, so there is nothing to seed.
        return CumlPCA(n_components=n_components)
    if method == "umap":
        # n_neighbors=15 / min_dist=0.1 are the umap-learn defaults the CPU path used.
        return CumlUMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=random_seed,
        )
    raise ValueError(f"Unsupported reducer={method}")


def fit_kmeans(x: np.ndarray, num_clusters: int, random_seed: int):
    require_cuml()
    # scikit-learn's n_init="auto" resolves to a single init for k-means++, which is
    # what cuML's scalable-k-means++ does with n_init=1. Keeping them equal avoids
    # silently changing the number of restarts when moving to the GPU.
    model = CumlKMeans(n_clusters=num_clusters, random_state=random_seed, n_init=1)
    labels = model.fit_predict(as_float32(x))
    return model, np.asarray(labels).astype(np.int64)


def fit_hdbscan(x: np.ndarray, min_cluster_size: int, min_samples: int | None):
    require_cuml()
    # cuML HDBSCAN supports euclidean only. Callers pass L2-normalised vectors, where
    # euclidean distance is a monotone function of cosine distance, so the clustering
    # is equivalent to the cosine version the CPU path used.
    model = CumlHDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
    )
    labels = model.fit_predict(as_float32(x))
    return model, np.asarray(labels).astype(np.int64)


def fit_clusterer(method: str, x: np.ndarray, num_clusters: int, random_seed: int):
    method = method.lower()
    if method == "kmeans":
        return fit_kmeans(x, num_clusters, random_seed)
    if method == "hdbscan":
        return fit_hdbscan(x, min_cluster_size=10, min_samples=None)
    raise ValueError(f"Unsupported cluster_method={method}")


def as_float32(x: np.ndarray) -> np.ndarray:
    """cuML requires contiguous float32; a float64 array otherwise triggers a silent copy."""
    return np.ascontiguousarray(x, dtype=np.float32)


# --------------------------------------------------------------------------------------
# Centroid maths on the GPU
# --------------------------------------------------------------------------------------


def l2_normalize_torch(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def compute_centroids(
    x: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalised mean vector per non-noise cluster.

    The CPU version looped over clusters and masked the full array once per cluster,
    which is O(n_clusters) passes over the data. This is one scatter-add.
    """
    labels_t = torch.from_numpy(np.asarray(labels).astype(np.int64)).to(device)
    valid_mask = labels_t >= 0
    if not bool(valid_mask.any()):
        raise ValueError("No non-noise clusters were found.")

    valid_clusters = torch.unique(labels_t[valid_mask], sorted=True)
    # Map arbitrary cluster ids onto a contiguous 0..k-1 range so they can index the
    # accumulator directly. searchsorted is valid because valid_clusters is sorted.
    dense = torch.searchsorted(valid_clusters, labels_t[valid_mask])

    features = torch.from_numpy(as_float32(x)).to(device)[valid_mask]
    sums = torch.zeros(len(valid_clusters), features.shape[1], device=device, dtype=features.dtype)
    sums.index_add_(0, dense, features)
    counts = torch.bincount(dense, minlength=len(valid_clusters)).clamp_min(1).unsqueeze(1)

    centroids = l2_normalize_torch(sums / counts)
    return centroids.cpu().numpy(), valid_clusters.cpu().numpy()


def assign_to_centroids(
    x: np.ndarray,
    centroids: np.ndarray,
    centroid_ids: np.ndarray,
    device: torch.device,
    chunk_size: int = 65_536,
) -> np.ndarray:
    """Nearest centroid by cosine similarity, chunked so the score matrix stays bounded."""
    centroids_t = torch.from_numpy(as_float32(centroids)).to(device)
    ids_t = torch.from_numpy(np.asarray(centroid_ids).astype(np.int64)).to(device)
    features = torch.from_numpy(as_float32(x))

    best: list[torch.Tensor] = []
    for start in range(0, len(features), chunk_size):
        block = features[start:start + chunk_size].to(device, non_blocking=True)
        scores = block @ centroids_t.T
        best.append(ids_t[scores.argmax(dim=1)].cpu())
    return torch.cat(best).numpy()


def assign_to_centroids_with_scores(
    x: np.ndarray,
    centroids: np.ndarray,
    centroid_ids: np.ndarray,
    device: torch.device,
    chunk_size: int = 65_536,
) -> tuple[np.ndarray, np.ndarray]:
    """``assign_to_centroids`` that also returns the winning cosine similarity per row.

    Used by the open-set path so the confidence of each cluster assignment is kept
    alongside the label and a distance cut-off can be applied post hoc.
    """
    centroids_t = torch.from_numpy(as_float32(centroids)).to(device)
    ids_t = torch.from_numpy(np.asarray(centroid_ids).astype(np.int64)).to(device)
    features = torch.from_numpy(as_float32(x))

    best_ids: list[torch.Tensor] = []
    best_scores: list[torch.Tensor] = []
    for start in range(0, len(features), chunk_size):
        block = features[start:start + chunk_size].to(device, non_blocking=True)
        top = (block @ centroids_t.T).max(dim=1)
        best_ids.append(ids_t[top.indices].cpu())
        best_scores.append(top.values.float().cpu())
    return torch.cat(best_ids).numpy(), torch.cat(best_scores).numpy()


def refine_clusters(
    x: np.ndarray,
    labels: np.ndarray,
    steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    refined_labels = np.asarray(labels).copy()
    centroids, centroid_ids = compute_centroids(x, refined_labels, device)
    for _ in range(max(steps, 0)):
        refined_labels = assign_to_centroids(x, centroids, centroid_ids, device)
        centroids, centroid_ids = compute_centroids(x, refined_labels, device)
    return refined_labels, centroids, centroid_ids


# --------------------------------------------------------------------------------------
# Parallel image I/O
# --------------------------------------------------------------------------------------


def filter_valid_image_paths(paths: list[str], num_workers: int = 16) -> list[bool]:
    """Readability check for every path, decoded across a thread pool.

    Pillow releases the GIL during decode, so threads scale here without the memory
    cost of forking processes for what is a throwaway check.
    """
    keep: list[bool] = [False] * len(paths)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = pool.map(lambda path: load_rgb_image(path) is not None, paths)
        for index, ok in enumerate(tqdm(results, total=len(paths), desc="verify images")):
            keep[index] = ok
    return keep


class PathDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> str:
        return self.paths[index]


class SingleProcessorCollator:
    """Decode and preprocess a batch inside a DataLoader worker."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: list[str]) -> dict[str, Any] | None:
        images: list[Image.Image] = []
        kept_paths: list[str] = []
        for path in batch:
            image = load_rgb_image(path)
            if image is None:
                continue
            images.append(image)
            kept_paths.append(path)
        if not kept_paths:
            return None
        return {"inputs": dict(self.processor(images=images, return_tensors="pt")), "paths": kept_paths}


class DualProcessorCollator:
    """Decode each image once, then preprocess it for two different backbones.

    SigLIP and DINO want different resolutions and normalisations, but they can share
    the JPEG decode, which is the expensive half.
    """

    def __init__(self, processor_a, processor_b):
        self.processor_a = processor_a
        self.processor_b = processor_b

    def __call__(self, batch: list[str]) -> dict[str, Any] | None:
        images: list[Image.Image] = []
        kept_paths: list[str] = []
        for path in batch:
            image = load_rgb_image(path)
            if image is None:
                continue
            images.append(image)
            kept_paths.append(path)
        if not kept_paths:
            return None
        return {
            "inputs_a": dict(self.processor_a(images=images, return_tensors="pt")),
            "inputs_b": dict(self.processor_b(images=images, return_tensors="pt")),
            "paths": kept_paths,
        }


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


def pooled_output(outputs: Any) -> torch.Tensor:
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is not None:
        return pooled
    return outputs.last_hidden_state[:, 0]


def _autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.inference_mode()
def encode_single_backbone(
    model: torch.nn.Module,
    processor,
    paths: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str,
    amp_dtype: torch.dtype | None = None,
) -> tuple[np.ndarray, list[str]]:
    loader = DataLoader(
        PathDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SingleProcessorCollator(processor),
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    vectors: list[torch.Tensor] = []
    kept_paths: list[str] = []
    model.eval()
    for batch in tqdm(loader, desc=desc):
        if batch is None:
            continue
        inputs = move_batch_to_device(batch["inputs"], device)
        with _autocast(device, amp_dtype):
            features = pooled_output(model(**inputs))
        vectors.append(F.normalize(features.float(), dim=-1).cpu())
        kept_paths.extend(batch["paths"])

    if not vectors:
        raise ValueError(f"No valid images were encoded for {desc}")
    return torch.cat(vectors, dim=0).numpy(), kept_paths


@torch.inference_mode()
def encode_dual_backbones(
    model_a: torch.nn.Module,
    processor_a,
    model_b: torch.nn.Module,
    processor_b,
    paths: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str,
    amp_dtype: torch.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Encode one path list with two backbones in a single pass over the images.

    Returns ``(features_a, features_b, kept_paths)`` already row-aligned, which also
    removes the need for the caller to cross-check two separate kept-path lists.
    """
    loader = DataLoader(
        PathDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=DualProcessorCollator(processor_a, processor_b),
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    vectors_a: list[torch.Tensor] = []
    vectors_b: list[torch.Tensor] = []
    kept_paths: list[str] = []
    model_a.eval()
    model_b.eval()

    for batch in tqdm(loader, desc=desc):
        if batch is None:
            continue
        inputs_a = move_batch_to_device(batch["inputs_a"], device)
        inputs_b = move_batch_to_device(batch["inputs_b"], device)
        with _autocast(device, amp_dtype):
            features_a = pooled_output(model_a(**inputs_a))
            features_b = pooled_output(model_b(**inputs_b))
        vectors_a.append(F.normalize(features_a.float(), dim=-1).cpu())
        vectors_b.append(F.normalize(features_b.float(), dim=-1).cpu())
        kept_paths.extend(batch["paths"])

    if not kept_paths:
        raise ValueError(f"No valid images were encoded for {desc}")
    return (
        torch.cat(vectors_a, dim=0).numpy(),
        torch.cat(vectors_b, dim=0).numpy(),
        kept_paths,
    )
