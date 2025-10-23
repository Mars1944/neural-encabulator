from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def load_vector_field(
    path: "str | Path",
    mmap: bool = True,
) -> np.ndarray:
    """
    Load a large vector field stored as a NumPy array.

    Accepts shapes like (H, W, C) or (C, H, W). Returns float32 array.
    Uses memory mapping by default to avoid loading entire file into RAM.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vector field file not found: {path}")
    arr = np.load(str(path), mmap_mode="r" if mmap else None)
    # Ensure float32
    if arr.dtype != np.float32:
        arr = np.asarray(arr, dtype=np.float32)
    return arr


def ensure_chw(arr: np.ndarray) -> np.ndarray:
    """Ensure array is (C, H, W). Accepts (H, W, C) or already (C, H, W)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (C,H,W) or (H,W,C), got shape {arr.shape}")
    if arr.shape[0] in (1, 2, 3, 4) and arr.shape[1] > 8 and arr.shape[2] > 8:
        # Already (C,H,W)
        return arr
    # Assume (H,W,C)
    return np.transpose(arr, (2, 0, 1))


def add_magnitude_channel(chw: np.ndarray) -> np.ndarray:
    """Append magnitude channel from first 2 or 3 vector components."""
    if chw.ndim != 3:
        raise ValueError("add_magnitude_channel expects (C,H,W)")
    c = chw.shape[0]
    if c < 2:
        return chw
    # Use first 2 or 3 channels for magnitude
    vec = chw[: min(3, c), ...]
    mag = np.sqrt(np.sum(vec * vec, axis=0, dtype=np.float32))  # (H,W)
    mag = np.expand_dims(mag, axis=0)  # (1,H,W)
    return np.concatenate([chw, mag], axis=0)


def normalize_per_channel(chw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-channel z-score normalization (mean/std), robust to large arrays."""
    c = chw.shape[0]
    out = np.empty_like(chw)
    for i in range(c):
        ch = chw[i]
        mean = float(ch.mean())
        std = float(ch.std())
        out[i] = (ch - mean) / (std + eps)
    return out


def iter_tiles(
    chw: np.ndarray,
    tile_size: Tuple[int, int],
    stride: Tuple[int, int] | None = None,
) -> Iterable[np.ndarray]:
    """
    Yield non-overlapping or strided tiles of shape (C, th, tw) from (C,H,W).
    No padding; tiles that don't fit exactly are skipped.
    """
    c, H, W = chw.shape
    th, tw = int(tile_size[0]), int(tile_size[1])
    if stride is None:
        sh, sw = th, tw
    else:
        sh, sw = int(stride[0]), int(stride[1])
    for y in range(0, H - th + 1, sh):
        for x in range(0, W - tw + 1, sw):
            yield chw[:, y : y + th, x : x + tw]


def load_vector_field_tiles(
    path: "str | Path",
    tile_size: Tuple[int, int] = (256, 256),
    stride: Tuple[int, int] | None = None,
    add_magnitude: bool = True,
    normalize: bool = True,
    limit_tiles: int | None = None,
) -> np.ndarray:
    """
    Load a large vector field and slice it into tiles suitable for CNN input.

    Returns a float32 array of shape (N, C, th, tw).
    """
    arr = load_vector_field(path)
    chw = ensure_chw(arr)
    if add_magnitude:
        chw = add_magnitude_channel(chw)
    if normalize:
        chw = normalize_per_channel(chw)

    tiles: List[np.ndarray] = []
    for t in iter_tiles(chw, tile_size=tile_size, stride=stride):
        tiles.append(t)
        if limit_tiles is not None and len(tiles) >= int(limit_tiles):
            break
    if not tiles:
        raise RuntimeError("No tiles produced; check tile_size/stride relative to field size")
    batch = np.stack(tiles, axis=0).astype(np.float32)
    return batch

