from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import csv

import numpy as np
import torch

# ----------------------------
# Shared helpers
# ----------------------------

def _to_tuple2(x: Sequence[int] | int, default: Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return int(x[0]), int(x[1])
    if isinstance(x, int):
        return int(x), int(x)
    return default


def _resolve_root_path(raw: str | Path) -> Path:
    """
    Resolve a root that may either be a directory or a text/CSV file containing
    the actual directory path on the first non-empty line.

    CSV handling:
      - if a header row has a column named 'path', use the first non-empty value in that column
      - else use the first non-empty cell in the file
    """
    p = Path(raw)
    if p.is_file() and p.suffix.lower() in {".txt", ".csv"}:
        try:
            if p.suffix.lower() == ".csv":
                with p.open("r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                if rows:
                    header = [h.strip().lower() for h in rows[0]]
                    path_col = header.index("path") if "path" in header else None
                    data_rows = rows[1:] if path_col is not None else rows
                    if path_col is not None:
                        for r in data_rows:
                            if len(r) > path_col and r[path_col].strip():
                                return Path(r[path_col].strip())
                    for r in data_rows:
                        for cell in r:
                            if cell.strip():
                                return Path(cell.strip())
            else:
                lines = p.read_text(encoding="utf-8").splitlines()
                for ln in lines:
                    stripped = ln.strip()
                    if stripped:
                        q = Path(stripped)
                        return q
        except Exception:
            pass
    return p


def _load_pil_image(path: Path, size: Tuple[int, int], channels: int) -> np.ndarray:
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Pillow (PIL) is required for image datasets. Install 'Pillow'.") from e

    with Image.open(path) as img:
        if channels == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img = img.resize((int(size[1]), int(size[0])), resample=Image.BILINEAR)
        arr = np.array(img)
        if channels == 1:
            arr = np.expand_dims(arr, axis=0)
        else:
            arr = np.transpose(arr, (2, 0, 1))
        arr = arr.astype(np.float32) / 255.0
        return arr


def _normalize(arr_chw: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    c = arr_chw.shape[0]
    if len(mean) not in (1, c) or len(std) not in (1, c):
        raise ValueError("image mean/std length must be 1 or match channels")
    if len(mean) == 1:
        mean = [float(mean[0])] * c
    if len(std) == 1:
        std = [float(std[0])] * c
    out = np.empty_like(arr_chw)
    for i in range(c):
        out[i] = (arr_chw[i] - float(mean[i])) / (float(std[i]) + 1e-6)
    return out


# ----------------------------
# Legacy image folder support
# ----------------------------

def build_class_index(root: Path, class_names: Optional[Sequence[str]] = None) -> Dict[str, int]:
    if class_names is not None and len(class_names) > 0:
        return {str(n): int(i) for i, n in enumerate(class_names)}
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
    """Original folder-based classifier dataset (kept for compatibility)."""

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
        self.root = _resolve_root_path(root_dir)
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
            found_classes = sorted({p.parent.name for p in all_imgs})
            if class_names and len(class_names) > 0:
                order = [str(n) for n in class_names if str(n) in found_classes]
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
        if not self.augment:
            return arr
        do_flip = bool(self.augment.get("flip", False))
        do_rotate = bool(self.augment.get("rotate", False))
        if do_flip and np.random.rand() < 0.5:
            arr = arr[:, :, ::-1]
        if do_rotate and np.random.rand() < 0.25:
            k = np.random.randint(1, 4)
            arr = np.rot90(arr, k=k, axes=(1, 2)).copy()
        return arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        arr = _load_pil_image(path, size=self.size, channels=self.channels)
        arr = self._maybe_augment(arr)
        arr = _normalize(arr, mean=self.mean, std=self.std)
        x = torch.from_numpy(arr.astype(np.float32))
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


# ----------------------------
# New paired image+CSV dataset
# ----------------------------

def _read_flow_metadata(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV missing header: {csv_path}")
        for r in reader:
            rows.append({k.strip().lower(): v for k, v in r.items()})
    return rows


def _index_images(root: Path, exts: Iterable[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}

    def _stem_keys(stem: str) -> List[str]:
        keys = {stem}
        lower = stem.lower()
        base = stem
        if lower.endswith("_mask"):
            base = stem[: -len("_mask")]
            keys.add(base)
        # Normalize frame prefix
        if base.lower().startswith("frame"):
            suf = base[5:]
            if suf.isdigit():
                keys.add(suf.lstrip("0") or "0")
                keys.add(str(int(suf)))
        # If purely numeric, include numeric forms
        if base.isdigit():
            keys.add(base.lstrip("0") or "0")
            keys.add(str(int(base)))
        return [k for k in keys if k]

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            for k in _stem_keys(p.stem):
                if k not in mapping:
                    mapping[k] = p
    return mapping


class PairedFlowDataset(torch.utils.data.Dataset[Dict[str, torch.Tensor]]):
    """
    Dataset that pairs images with rows from a CSV containing frame_index,digits,velocity,reynolds_number,(optional split/label).
    - Matches by frame_index == image filename stem.
    - Derives laminar/turbulent label from Reynolds number (<2300 -> laminar=0, else turbulent=1) when no explicit label is provided.
    - Keeps per-item metadata: split, path, frame_index.
    """

    def __init__(
        self,
        image_root: str | Path,
        csv_path: str | Path,
        *,
        image_size: Sequence[int] | int = (256, 256),
        channels: int = 3,
        mean: Sequence[float] = (0.5,),
        std: Sequence[float] = (0.5,),
        augment: Optional[Dict[str, bool]] = None,
        split_filter: Optional[str] = None,
        laminar_threshold: float = 2300.0,
        console: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.root = _resolve_root_path(image_root)
        if not self.root.exists():
            raise FileNotFoundError(f"Image root not found: {self.root}")
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.csv_path}")

        self.size = _to_tuple2(image_size, (256, 256))
        self.channels = 1 if int(channels) == 1 else 3
        self.mean = list(mean)
        self.std = list(std)
        self.augment = augment or {}
        self.split_filter = split_filter.lower() if isinstance(split_filter, str) else None
        self.laminar_threshold = float(laminar_threshold)

        rows = _read_flow_metadata(self.csv_path)
        img_map = _index_images(self.root, exts={".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})

        items = []
        dropped = 0
        matched_images: set[str] = set()
        last_velocity: Optional[float] = None
        last_re: Optional[float] = None
        last_digits: Optional[float] = None
        imputed_count = 0
        impute_warnings: list[tuple[str, str, float]] = []
        impute_messages: list[str] = []
        for r in rows:
            frame_idx = str(r.get("frame_index", "")).strip()
            if not frame_idx:
                dropped += 1
                continue
            img_path = img_map.get(Path(frame_idx).stem)
            if img_path is None:
                dropped += 1
                continue
            try:
                velocity_raw = r.get("velocity", "nan")
                reynolds_raw = r.get("reynolds_number", "nan")
                velocity = float(velocity_raw) if str(velocity_raw).strip() not in ("", "nan", "None") else np.nan
                reynolds = float(reynolds_raw) if str(reynolds_raw).strip() not in ("", "nan", "None") else np.nan
            except Exception:
                velocity, reynolds = np.nan, np.nan

            # Impute missing velocity/Re with last seen values
            missing_v = bool(np.isnan(velocity))
            missing_r = bool(np.isnan(reynolds))
            if missing_v and last_velocity is not None:
                velocity = float(last_velocity)
                imputed_count += 1
                impute_messages.append(f"[impute] velocity missing at frame {frame_idx}; using last value {velocity}")
                impute_warnings.append((frame_idx, "velocity", float(velocity)))
            if missing_r and last_re is not None:
                reynolds = float(last_re)
                imputed_count += 1
                impute_messages.append(f"[impute] Reynolds missing at frame {frame_idx}; using last value {reynolds}")
                impute_warnings.append((frame_idx, "reynolds_number", float(reynolds)))
            if np.isnan(velocity) or np.isnan(reynolds):
                dropped += 1
                continue
            last_velocity = float(velocity)
            last_re = float(reynolds)

            digits = r.get("digits", None)
            try:
                digits_val = float(digits) if digits is not None and str(digits).strip() != "" else None
            except Exception:
                digits_val = None
            if digits_val is None and last_digits is not None:
                digits_val = float(last_digits)
                imputed_count += 1
                impute_messages.append(f"[impute] digits missing at frame {frame_idx}; using last value {digits_val}")
                impute_warnings.append((frame_idx, "digits", float(digits_val)))
            if digits_val is not None:
                last_digits = digits_val

            # Label: explicit column "label" or "laminar_turbulent" (0/1 or laminar/turbulent), else derive from Reynolds
            raw_label = r.get("label", r.get("laminar_turbulent", None))
            if raw_label is None or str(raw_label).strip() == "":
                label = 0 if reynolds < self.laminar_threshold else 1
            else:
                s = str(raw_label).strip().lower()
                if s.isdigit():
                    label = 0 if int(s) == 0 else 1
                else:
                    label = 0 if s.startswith("laminar") else 1

            split = str(r.get("split", "train")).strip().lower()
            if self.split_filter is not None and split != self.split_filter:
                continue

            items.append(
                {
                    "path": img_path,
                    "frame_index": frame_idx,
                    "split": split,
                    "label": int(label),
                    "velocity": float(velocity),
                    "reynolds_number": float(reynolds),
                    "digits": digits_val,
                }
            )
            matched_images.add(Path(frame_idx).stem)

        self.items = items
        self.dropped = dropped
        self.unmatched_images = [p for stem, p in img_map.items() if stem not in matched_images]
        if console is not None:
            console.info(f"PairedFlowDataset matched {len(self.items)} samples (dropped rows={dropped}, unmatched_images={len(self.unmatched_images)})")
            if dropped > 0 or self.unmatched_images:
                console.warn("Dataset drop/flag: rows without images or images without rows were ignored for pairing.")
            if imputed_count > 0:
                console.warn(f"Imputed {imputed_count} missing numeric fields using last observed values.")
            if len(impute_warnings) > 10:
                try:
                    import csv as _csv

                    warn_path = self.csv_path.with_name(f"{self.csv_path.stem}_impute_warnings.csv")
                    warn_path.parent.mkdir(parents=True, exist_ok=True)
                    with warn_path.open("w", newline="", encoding="utf-8") as f:
                        w = _csv.writer(f)
                        w.writerow(["frame_index", "field", "value_used"])
                        for frame_idx, field, value in impute_warnings:
                            w.writerow([frame_idx, field, value])
                    console.warn(f"[impute] Many imputations detected; details saved to {warn_path}")
                except Exception as e:
                    console.warn(f"[impute] Failed to save imputation warnings CSV: {e}")
            else:
                for msg in impute_messages:
                    console.warn(msg)
        if not self.items:
            raise RuntimeError("No paired samples found; check frame_index naming between images and CSV.")

    def __len__(self) -> int:
        return len(self.items)

    def _maybe_augment(self, arr: np.ndarray) -> np.ndarray:
        if not self.augment:
            return arr
        do_flip = bool(self.augment.get("flip", False))
        do_rotate = bool(self.augment.get("rotate", False))
        if do_flip and np.random.rand() < 0.5:
            arr = arr[:, :, ::-1]
        if do_rotate and np.random.rand() < 0.25:
            k = np.random.randint(1, 4)
            arr = np.rot90(arr, k=k, axes=(1, 2)).copy()
        return arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | str]]:
        item = self.items[idx]
        arr = _load_pil_image(item["path"], size=self.size, channels=self.channels)
        arr = self._maybe_augment(arr)
        arr = _normalize(arr, mean=self.mean, std=self.std)
        x = torch.from_numpy(arr.astype(np.float32))
        target: Dict[str, torch.Tensor | str] = {
            "label": torch.tensor(int(item["label"]), dtype=torch.long),
            "velocity": torch.tensor(float(item["velocity"]), dtype=torch.float32),
            "reynolds_number": torch.tensor(float(item["reynolds_number"]), dtype=torch.float32),
            "frame_index": item["frame_index"],
            "path": str(item["path"]),
            "split": item["split"],
        }
        if item.get("digits") is not None:
            target["digits"] = torch.tensor(float(item["digits"]), dtype=torch.float32)
        return x, target
