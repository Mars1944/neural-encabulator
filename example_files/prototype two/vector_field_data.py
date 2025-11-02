from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _load_csv_vector_field(path: Path) -> np.ndarray:
    """
    Load a sparse vector field from a CSV file with 4 columns:
    x, y, delta_x, delta_y. Builds a dense (H, W, 2) float32 array where
    channel 0 = delta_x and channel 1 = delta_y. Missing positions are 0.

    Notes:
    - x maps to column index, y maps to row index (array indexing [y, x]).
    - If the minimum x/y are not zero, the grid is offset so the minimum
      observed coordinate maps to index 0.
    - Header is allowed and will be skipped if present.
    """
    # Read first line to detect a header; then load data
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
    has_header = any(h in first.lower() for h in ("x", "y", "delta", "dx", "dy"))

    data = np.loadtxt(str(path), delimiter=",", skiprows=1 if has_header else 0)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(
            f"Expected CSV with 4 columns (x,y,delta_x,delta_y); got shape {data.shape}"
        )
    # Keep only first 4 columns in case of extras
    data = data[:, :4]

    x = data[:, 0].astype(np.int64)
    y = data[:, 1].astype(np.int64)
    dx = data[:, 2].astype(np.float32)
    dy = data[:, 3].astype(np.float32)

    # Offset so the grid starts at (0,0)
    min_x, min_y = int(x.min()), int(y.min())
    x0 = x - min_x
    y0 = y - min_y

    H = int(y0.max()) + 1
    W = int(x0.max()) + 1

    grid = np.zeros((H, W, 2), dtype=np.float32)
    grid[y0, x0, 0] = dx
    grid[y0, x0, 1] = dy
    return grid


def load_vector_field(
    path: "str | Path",
    mmap: bool = True,
) -> np.ndarray:
    """
    Load a vector field from disk.

    Supports:
    - NumPy arrays: .npy / .npz containing (H, W, C) or (C, H, W)
    - CSV: 4 columns (x, y, delta_x, delta_y) -> (H, W, 2)

    Returns a float32 array. For NumPy arrays, memory mapping is used by default.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vector field file not found: {path}")

    ext = path.suffix.lower()
    if ext == ".csv":
        return _load_csv_vector_field(path)

    # Fallback to NumPy loader
    arr = np.load(str(path), mmap_mode="r" if mmap else None)
    if isinstance(arr, np.lib.npyio.NpzFile):  # .npz: choose the first array
        # Pick the first item deterministically
        keys = sorted(arr.files)
        if not keys:
            raise ValueError(f"Empty .npz archive: {path}")
        arr = arr[keys[0]]
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

