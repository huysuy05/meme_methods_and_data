#!/usr/bin/env python3
"""GPU helpers for large-scale pHash discovery.

Two pieces, both needed once the discovery corpus grows past ~1M images:

1. ``encode_phashes_gpu`` — batched perceptual hashing. The 2-D DCT becomes a pair of
   matmuls (``D @ X @ D.T``), so whole batches hash on the GPU instead of one image at a
   time through scipy.

2. ``radius_neighbor_graph_hamming_gpu`` — the pairwise Hamming comparison from Zannettou
   et al. (2018). They parallelised it across GPUs with TensorFlow; this is the same idea
   in torch on one GPU. A dense N x N matrix is impossible at this scale (1.43M images
   would need ~2 TB), so distances are computed in tiles and thresholded on the GPU,
   emitting only the sparse set of pairs within ``max_bits``. DBSCAN then consumes that
   sparse graph via ``metric="precomputed"``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.sparse import coo_matrix
from tqdm.auto import tqdm

from meme_research_eval_utils import load_rgb_image

# Exact-duplicate pairs have Hamming distance 0. scipy prunes explicit zeros from sparse
# matrices, which would silently delete those edges, so store a tiny positive value instead.
ZERO_DISTANCE_EPS = 1e-6


def _dct_matrix(size: int, device: torch.device) -> torch.Tensor:
    """Orthonormal DCT-II matrix, so that ``D @ X @ D.T`` equals scipy's 2-D dct(norm='ortho')."""
    n = torch.arange(size, dtype=torch.float64, device=device)
    k = n.reshape(-1, 1)
    matrix = torch.cos(torch.pi * (2.0 * n + 1.0) * k / (2.0 * size))
    scale = torch.full((size, 1), np.sqrt(2.0 / size), dtype=torch.float64, device=device)
    scale[0, 0] = np.sqrt(1.0 / size)
    # Kept in float64: coefficients sitting near the median otherwise flip bits under
    # float32/TF32 rounding, and the hash would not match the scipy reference.
    return matrix * scale


def _load_gray_array(path: str, img_size: int) -> tuple[str, np.ndarray] | None:
    image = load_rgb_image(path)
    if image is None:
        return None
    gray = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    return path, np.asarray(gray, dtype=np.float32)


def encode_phashes_gpu(
    paths: list[str],
    hash_size: int,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 8,
    highfreq_factor: int = 4,
    desc: str = "encode phash (gpu)",
) -> tuple[np.ndarray, list[str]]:
    """Hash images in batches on the GPU. Unreadable images are skipped (fail-soft).

    Returns ``(bits, kept_paths)`` where ``bits`` is uint8 of shape (n_kept, hash_size**2)
    and matches the CPU implementation bit for bit.
    """
    img_size = hash_size * highfreq_factor
    dct = _dct_matrix(img_size, device)

    all_bits: list[np.ndarray] = []
    kept_paths: list[str] = []
    skipped = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for start in tqdm(range(0, len(paths), batch_size), desc=desc):
            chunk = paths[start:start + batch_size]
            loaded = [item for item in pool.map(lambda p: _load_gray_array(p, img_size), chunk) if item is not None]
            skipped += len(chunk) - len(loaded)
            if not loaded:
                continue

            batch_paths = [item[0] for item in loaded]
            pixels = torch.from_numpy(np.stack([item[1] for item in loaded])).to(
                device, dtype=torch.float64, non_blocking=True
            )

            # 2-D DCT: transform columns then rows, exactly as scipy dct(axis=0) -> dct(axis=1).
            coeffs = dct @ pixels @ dct.T
            low_freq = coeffs[:, :hash_size, :hash_size].reshape(len(loaded), -1)
            # A matmul DCT leaves ~1e-29 rounding noise where scipy's FFT returns exact zeros.
            # On a uniform image every non-DC coefficient is zero, so that noise drags the
            # median just below zero and inverts every tie in the `>` comparison. Snap
            # noise-level values to zero, relative to the largest coefficient in the image.
            scale = low_freq.abs().amax(dim=1, keepdim=True)
            low_freq = torch.where(low_freq.abs() < 1e-12 * scale, torch.zeros_like(low_freq), low_freq)
            # Median over every coefficient except DC, matching the CPU path.
            median = torch.quantile(low_freq[:, 1:], 0.5, dim=1, keepdim=True)
            bits = (low_freq > median).to(torch.uint8)

            all_bits.append(bits.cpu().numpy())
            kept_paths.extend(batch_paths)

    if not all_bits:
        raise ValueError(f"No valid images encoded for {desc}")
    if skipped:
        print(f"{desc}: skipped {skipped} unreadable images")
    return np.vstack(all_bits), kept_paths


def radius_neighbor_graph_hamming_gpu(
    bits: np.ndarray,
    max_bits: int,
    device: torch.device,
    tile_size: int = 8192,
    max_neighbors: int | None = None,
) -> coo_matrix:
    """Sparse graph of all pairs within ``max_bits`` Hamming distance, computed on the GPU.

    Maps bits to +/-1 so a dot product yields the Hamming distance directly:
    ``hamming = (n_bits - dot) / 2``. Only the upper triangle is computed and then mirrored.

    ``max_neighbors`` caps the neighbours kept per row. Meme corpora contain very large
    near-duplicate groups, and an uncapped graph can exceed RAM; the cap bounds memory at
    the cost of possibly splitting a huge cluster. Leave as ``None`` for exact behaviour.
    """
    n_points, n_bits = bits.shape
    signs = torch.from_numpy(bits.astype(np.float32) * 2.0 - 1.0)
    dot_threshold = float(n_bits - 2 * max_bits)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    per_row_count = np.zeros(n_points, dtype=np.int64)
    capped_rows = 0

    n_tiles = (n_points + tile_size - 1) // tile_size
    total_blocks = n_tiles * (n_tiles + 1) // 2
    progress = tqdm(total=total_blocks, desc="pairwise hamming (gpu)")

    for i0 in range(0, n_points, tile_size):
        i1 = min(i0 + tile_size, n_points)
        block_i = signs[i0:i1].to(device, non_blocking=True)

        for j0 in range(i0, n_points, tile_size):
            j1 = min(j0 + tile_size, n_points)
            block_j = signs[j0:j1].to(device, non_blocking=True)

            similarity = block_i @ block_j.T
            mask = similarity >= dot_threshold
            if i0 == j0:
                # Diagonal block: keep the upper triangle including the diagonal so each
                # point is its own neighbour (mirrors dense DBSCAN, where d(i,i)=0 <= eps).
                mask = torch.triu(mask)

            hits = mask.nonzero(as_tuple=False)
            if hits.numel() == 0:
                progress.update(1)
                continue

            local_i = hits[:, 0]
            local_j = hits[:, 1]
            hamming = (n_bits - similarity[local_i, local_j]) / 2.0

            r = (local_i + i0).to(torch.int32).cpu().numpy()
            c = (local_j + j0).to(torch.int32).cpu().numpy()
            d = hamming.to(torch.float32).cpu().numpy()
            d[d <= 0.0] = ZERO_DISTANCE_EPS

            if max_neighbors is not None:
                keep_mask = np.ones(len(r), dtype=bool)
                for idx in range(len(r)):
                    row_id = r[idx]
                    if per_row_count[row_id] >= max_neighbors:
                        keep_mask[idx] = False
                        capped_rows += 1
                    else:
                        per_row_count[row_id] += 1
                r, c, d = r[keep_mask], c[keep_mask], d[keep_mask]

            rows.append(r)
            cols.append(c)
            dists.append(d)
            progress.update(1)

    progress.close()

    if not rows:
        raise ValueError("No pairs found within the Hamming threshold.")

    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    values = np.concatenate(dists)

    # Mirror the upper triangle; skip the diagonal so self-loops are not duplicated.
    off_diagonal = row_idx != col_idx
    full_rows = np.concatenate([row_idx, col_idx[off_diagonal]])
    full_cols = np.concatenate([col_idx, row_idx[off_diagonal]])
    full_vals = np.concatenate([values, values[off_diagonal]])

    if capped_rows:
        print(f"WARNING: neighbour cap ({max_neighbors}) dropped {capped_rows} edges")
    print(f"neighbour_graph_edges={len(full_vals)} points={n_points}")

    return coo_matrix((full_vals, (full_rows, full_cols)), shape=(n_points, n_points))
