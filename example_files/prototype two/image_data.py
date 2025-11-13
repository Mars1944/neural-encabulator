from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _to_tuple2(x: Sequence[int] | int, default: Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return int(x[0]), int(x[1])
    if isinstance(x, int):
        return int(x), int(x)
    return default


def _load_pil_image(path: Path, size: Tuple[int, int], channels: int) -> np.ndarray:
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Pillow (PIL) is required for image datasets. Install 'Pillow'.") from e

    with Image.open(path) as img:
        # Convert channels
        if channels == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        # Resize HxW
        # size is (H, W); PIL expects (W, H)
        img = img.resize((int(size[1]), int(size[0])), resample=Image.BILINEAR)
        arr = np.array(img)
        if channels == 1:
            # (H, W) -> (1, H, W)
            arr = np.expand_dims(arr, axis=0)
        else:
            # (H, W, 3) -> (3, H, W)
            arr = np.transpose(arr, (2, 0, 1))
        # Ensure float32 in [0,1]
        arr = arr.astype(np.float32) / 255.0
        return arr


def _normalize(arr_chw: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    c = arr_chw.shape[0]
    # Allow single-value broadcast or exact length; otherwise error early
    if len(mean) not in (1, c) or len(std) not in (1, c):
        raise ValueError("image mean/std length must be 1 or match channels")
    # Broadcast a single value if provided
    if len(mean) == 1:
        mean = [float(mean[0])] * c
    if len(std) == 1:
        std = [float(std[0])] * c
    out = np.empty_like(arr_chw)
    for i in range(c):
        out[i] = (arr_chw[i] - float(mean[i])) / (float(std[i]) + 1e-6)
    return out


def build_class_index(root: Path, class_names: Optional[Sequence[str]] = None) -> Dict[str, int]:
    if class_names is not None and len(class_names) > 0:
        return {str(n): int(i) for i, n in enumerate(class_names)}
    # Discover subdirectories (classes) in alphabetical order
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    return {name: i for i, name in enumerate(classes)}


def discover_image_files(
    root: Path,
    class_to_idx: Dict[str, int],
    extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    for cls, idx in class_to_idx.items():
        d = root / cls
        if not d.exists() or not d.is_dir():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in extensions:
                items.append((p, idx))
    return items


class ImageFolderDataset(torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        root_dir: str | Path,
        *,
        image_size: Sequence[int] | int = (256, 256),
        channels: int = 3,
        mean: Sequence[float] = (0.5,),
        std: Sequence[float] = (0.5,),
        augment: Optional[Dict[str, bool]] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Image root not found: {self.root}")
        self.size = _to_tuple2(image_size, (256, 256))
        self.channels = 1 if int(channels) == 1 else 3
        self.mean = list(mean)
        self.std = list(std)
        self.augment = augment or {}
        self.class_to_idx = build_class_index(self.root, class_names)
        self.samples = discover_image_files(self.root, self.class_to_idx)
        if not self.samples:
            # Fallback: recursively scan and derive class from parent directory name
            all_imgs: List[Path] = [p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
            if not all_imgs:
                raise RuntimeError(f"No images found under {self.root} (expected subfolders per class)")
            # Derive classes from immediate parent folder names
            found_classes = sorted({p.parent.name for p in all_imgs})
            # If class_names provided, respect that order for any overlapping names
            if class_names and len(class_names) > 0:
                order = [str(n) for n in class_names if str(n) in found_classes]
                # Append any remaining discovered classes in alphabetical order
                order += [n for n in found_classes if n not in order]
                mapping = {name: i for i, name in enumerate(order)}
            else:
                mapping = {name: i for i, name in enumerate(found_classes)}
            self.class_to_idx = mapping
            self.samples = [(p, mapping[p.parent.name]) for p in all_imgs if p.parent.name in mapping]
            if not self.samples:
                raise RuntimeError(f"No images found under {self.root} after fallback scan")

    def __len__(self) -> int:
        return len(self.samples)

    def _maybe_augment(self, arr: np.ndarray) -> np.ndarray:
        # arr is (C,H,W)
        if not self.augment:
            return arr
        do_flip = bool(self.augment.get("flip", False))
        do_rotate = bool(self.augment.get("rotate", False))
        if do_flip and np.random.rand() < 0.5:
            # horizontal flip (reverse W)
            arr = arr[:, :, ::-1]
        if do_rotate and np.random.rand() < 0.25:
            # rotate by k*90
            k = np.random.randint(1, 4)
            arr = np.rot90(arr, k=k, axes=(1, 2)).copy()
        return arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        arr = _load_pil_image(path, size=self.size, channels=self.channels)
        arr = self._maybe_augment(arr)
        arr = _normalize(arr, mean=self.mean, std=self.std)
        x = torch.from_numpy(arr.astype(np.float32))  # (C,H,W)
        y = torch.tensor(int(label), dtype=torch.long)
        return x, y


def split_train_val(
    root_dir: str | Path,
    *,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], Dict[str, int]]:
    """Simple deterministic split over discovered samples when no explicit val dir is provided."""
    root = Path(root_dir)
    class_to_idx = build_class_index(root)
    items = discover_image_files(root, class_to_idx)
    if not items:
        raise RuntimeError(f"No images found under {root}")
    rng = np.random.default_rng(int(seed))
    idxs = np.arange(len(items))
    rng.shuffle(idxs)
    n_val = int(round(len(items) * max(0.0, min(0.9, float(val_split)))))
    val_idx = idxs[:n_val]
    tr_idx = idxs[n_val:]
    train_items = [items[i] for i in tr_idx.tolist()]
    val_items = [items[i] for i in val_idx.tolist()]
    return train_items, val_items, class_to_idx
