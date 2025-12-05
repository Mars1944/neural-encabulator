from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, List
import time

import numpy as np
import torch

from vector_field_data import (
    ensure_chw,
    load_vector_field,
    load_vector_field_tiles,
    load_vector_field_tiles_with_labels,
)
from common import load_labels_from_csv, parse_field_paths, PathResolver
from console import console_from_config
from common import PathResolver as _PR, ensure_outputs_ready, plot_training_progress
try:
    from data_loader import ImageFolderDataset, build_class_index, PairedFlowDataset, _resolve_root_path
except Exception:
    ImageFolderDataset = None  # type: ignore
    build_class_index = None  # type: ignore
    PairedFlowDataset = None  # type: ignore
    _resolve_root_path = None  # type: ignore


def _prepare_labels(
    *,
    n: int,
    config: Dict[str, Any],
    split: str,
) -> np.ndarray:
    """Resolve labels for a split: uses <split>_labels_csv | <split>_label_all | <split>_labels | generic keys."""
    # Prefer split-specific keys
    key_csv = f"{split}_labels_csv"
    key_all = f"{split}_label_all"
    key_arr = f"{split}_labels"

    # 1) CSV
    csv_path = config.get(key_csv, None) or config.get("labels_csv", None)
    if isinstance(csv_path, str) and csv_path.strip():
        p = Path(csv_path)
        if not p.is_absolute():
            # Try relative to config dir if available on config
            base = Path(config.get("_config_dir", "."))
            p = (base / p).resolve()
        labels = load_labels_from_csv(p)
        if labels.shape[0] != n:
            raise RuntimeError(f"Labels length {labels.shape[0]} does not match tiles {n}")
        return labels.astype(np.int64)

    # 2) label_all
    la = config.get(key_all, None)
    if la is None:
        la = config.get("label_all", None)
    if la is not None:
        c = int(la)
        return np.full((n,), c, dtype=np.int64)

    # 3) labels array in config
    arr = config.get(key_arr, None)
    if arr is None:
        arr = config.get("labels", None)
    if isinstance(arr, (list, tuple)) and len(arr) == n:
        return np.asarray([int(x) for x in arr], dtype=np.int64)

    raise RuntimeError(
        f"No labels provided for split '{split}'. Set {key_csv} (CSV), {key_all}, or {key_arr} (or generic labels_* keys)."
    )


class _TilesDataset(torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x_np: np.ndarray, y_np: np.ndarray) -> None:
        assert x_np.ndim == 4, f"Expected (N,C,H,W) got {x_np.shape}"
        assert x_np.shape[0] == y_np.shape[0], "Mismatched x/y lengths"
        self.x = torch.from_numpy(x_np)
        self.y = torch.from_numpy(y_np.astype(np.int64))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class Trainer:
    """
    Simple class-based trainer for prototype two.

    Expects training tiles from config.train_field_path (or config.field_path) and labels via
    one of: train_labels_csv | train_label_all | train_labels (or generic labels_* keys).
    Optionally uses config.test_field_path for validation, else splits train into train/val via config.val_split.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device, model: torch.nn.Module) -> None:
        self.config = dict(config)
        # Remember config directory for relative label paths
        if "_config_dir" not in self.config and isinstance(config.get("_config_path"), (str, Path)):
            self.config["_config_dir"] = str(Path(self.config["_config_path"]).parent)
        self.device = device
        self.model = model.to(device)
        task = str(self.config.get("task", "")).lower()
        flow_csv_present = bool(self.config.get("train_flow_csv") or self.config.get("flow_csv"))
        self.is_multitask = task.startswith("multi") or bool(self.config.get("multitask", False)) or flow_csv_present
        self.criterion = torch.nn.CrossEntropyLoss()
        self.reg_loss = torch.nn.MSELoss()
        self.loss_w_cls = float(self.config.get("loss_weight_cls", self.config.get("loss_weights", {}).get("cls", 1.0) if isinstance(self.config.get("loss_weights", {}), dict) else self.config.get("loss_weights", 1.0)))
        self.loss_w_reg = float(self.config.get("loss_weight_reg", self.config.get("loss_weights", {}).get("reg", 1.0) if isinstance(self.config.get("loss_weights", {}), dict) else 1.0))
        self.console = console_from_config(self.config)
        # When True, _load_split will reuse tile_size/tile_stride from config without recomputing
        self._tiling_locked: bool = False
        from cnn_optim import CnnOptim

        self.optim_builder = CnnOptim(self.config)
        self.optimizer, self.scheduler, self.scheduler_step_on = self.optim_builder.build(self.model)

    def _cnn_min_side(self) -> int:
        """Minimum spatial side length required by the CNN given max-pooling depth.

        Uses conv_channels length and pool_every to estimate total number of MaxPool2d layers.
        For N pools with kernel/stride=2, the minimum side is 2**N to preserve at least 1 pixel.
        """
        try:
            conv_channels = self.config.get("conv_channels", [32, 64])
            pool_every = int(self.config.get("pool_every", 1))
            if not isinstance(conv_channels, (list, tuple)) or pool_every <= 0:
                return 1
            n_blocks = len(conv_channels)
            n_pools = n_blocks // max(pool_every, 1)
            return max(1, 2 ** int(n_pools))
        except Exception:
            return 1

    def _is_image_mode(self) -> bool:
        # Vector path removed; force image mode
        return True

    def _flow_collate(self, batch):
        xs = [b[0] for b in batch]
        targets = [b[1] for b in batch]
        x = torch.stack(xs, dim=0)
        out = {
            "label": torch.stack([t["label"] for t in targets]).long(),
            "velocity": torch.stack([t["velocity"] for t in targets]).float(),
            "reynolds": torch.stack([t["reynolds_number"] for t in targets]).float(),
            "frame_index": [t["frame_index"] for t in targets],
            "path": [t["path"] for t in targets],
            "split": [t["split"] for t in targets],
        }
        if "digits" in targets[0]:
            out["digits"] = torch.stack([t.get("digits", torch.tensor(0.0)) for t in targets]).float()
        return x, out

    def _build_image_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        if ImageFolderDataset is None and PairedFlowDataset is None:
            raise RuntimeError("Image support not available; missing data_loader module")
        cfg = self.config
        train_dir_raw = cfg.get("train_image_dir", None)
        val_dir_raw = cfg.get("val_image_dir", None)
        # Anchor paths relative to config
        base_cfg_path = Path(cfg.get("_config_path", Path(cfg.get("_config_dir", ".")) / "config.json"))
        resolver = PathResolver(base_cfg_path, outputs_root=str(cfg.get("outputs_root")) if isinstance(cfg.get("outputs_root"), str) else None)
        train_dir = str(resolver.anchor(train_dir_raw)) if isinstance(train_dir_raw, str) and train_dir_raw.strip() else None
        val_dir = str(resolver.anchor(val_dir_raw)) if isinstance(val_dir_raw, str) and val_dir_raw.strip() else None
        if _resolve_root_path is not None:
            if isinstance(train_dir, str):
                train_dir = str(_resolve_root_path(train_dir))
            if isinstance(val_dir, str):
                val_dir = str(_resolve_root_path(val_dir))
        # If a container folder was provided (e.g., 'training data'), auto-descend into 'image' if present
        try:
            if isinstance(train_dir, str):
                p = Path(train_dir)
                if p.exists() and p.is_dir():
                    cand = p / "image"
                    if cand.exists() and cand.is_dir():
                        train_dir = str(cand.resolve())
        except Exception:
            pass
        try:
            if isinstance(val_dir, str):
                p = Path(val_dir)
                if p.exists() and p.is_dir():
                    cand = p / "image"
                    if cand.exists() and cand.is_dir():
                        val_dir = str(cand.resolve())
        except Exception:
            pass
        if not isinstance(train_dir, str) or not train_dir.strip():
            raise RuntimeError("train_image_dir must be set for image training")
        image_size = tuple(cfg.get("image_size", [256, 256]))
        channels = int(cfg.get("image_channels", 3))
        mean = cfg.get("image_mean", [0.5])
        std = cfg.get("image_std", [0.5])
        augment = cfg.get("augment", None)
        class_names = cfg.get("class_names", None)

        batch_size = int(cfg.get("batch_size", 16))
        num_workers = int(cfg.get("num_workers", 0))
        pin_memory = bool(cfg.get("pin_memory", False))

        # Multitask path: use paired CSV+images
        if self.is_multitask:
            if PairedFlowDataset is None:
                raise RuntimeError("PairedFlowDataset not available; ensure data_loader is present")
            flow_csv = cfg.get("train_flow_csv", cfg.get("flow_csv", None))
            if not isinstance(flow_csv, str) or not flow_csv.strip():
                raise RuntimeError("train_flow_csv (or flow_csv) must be set for multitask image training")
            flow_csv_path = resolver.anchor(flow_csv)
            base_ds = PairedFlowDataset(
                train_dir,
                flow_csv_path,
                image_size=image_size,
                channels=channels,
                mean=mean,
                std=std,
                augment=augment,
                split_filter=None,
                laminar_threshold=float(cfg.get("laminar_threshold", 2300.0)),
                console=self.console,
            )
            splits = [item.get("split", "train") for item in base_ds.items] if hasattr(base_ds, "items") else []
            has_val_rows = any(str(s).lower() == "val" for s in splits)
            if has_val_rows:
                tr_idx = [i for i, s in enumerate(splits) if str(s).lower() == "train"]
                va_idx = [i for i, s in enumerate(splits) if str(s).lower() == "val"]
                if not tr_idx or not va_idx:
                    idx = np.arange(len(base_ds))
                    rng = np.random.default_rng(int(cfg.get("seed", 42)))
                    rng.shuffle(idx)
                    val_split = float(cfg.get("val_split", 0.2))
                    val_split = min(max(val_split, 0.0), 0.9)
                    n_val = int(round(len(base_ds) * val_split))
                    va_idx = idx[:n_val].tolist()
                    tr_idx = idx[n_val:].tolist()
            else:
                idx = np.arange(len(base_ds))
                rng = np.random.default_rng(int(cfg.get("seed", 42)))
                rng.shuffle(idx)
                val_split = float(cfg.get("val_split", 0.2))
                val_split = min(max(val_split, 0.0), 0.9)
                n_val = int(round(len(base_ds) * val_split))
                va_idx = idx[:n_val].tolist()
                tr_idx = idx[n_val:].tolist()
            ds_train = torch.utils.data.Subset(base_ds, tr_idx)
            ds_val = torch.utils.data.Subset(base_ds, va_idx)
            self.console.info(f"[split] multitask image dataset -> train={len(ds_train)}, val={len(ds_val)}")
            limit_frac = float(cfg.get("debug_train_fraction", cfg.get("max_train_fraction", 1.0)))
            debug_enabled = bool(cfg.get("debug_limit_training", False))
            if debug_enabled and limit_frac > 0 and limit_frac < 1.0:
                n_keep = max(1, int(len(ds_train) * limit_frac))
                ds_train = torch.utils.data.Subset(ds_train, list(range(n_keep)))
                self.console.warn(f"[debug] Limiting training samples to {n_keep} ({limit_frac*100:.1f}%) for quick run.")
            train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=self._flow_collate)
            val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=self._flow_collate)
            return train_loader, val_loader

        try:
            if isinstance(val_dir, str) and val_dir.strip():
                ds_train = ImageFolderDataset(train_dir, image_size=image_size, channels=channels, mean=mean, std=std, augment=augment, class_names=class_names)
                ds_val = ImageFolderDataset(val_dir, image_size=image_size, channels=channels, mean=mean, std=std, augment=None, class_names=class_names)
            else:
                ds_all = ImageFolderDataset(train_dir, image_size=image_size, channels=channels, mean=mean, std=std, augment=augment, class_names=class_names)
                n = len(ds_all)
                val_split = float(cfg.get("val_split", 0.2))
                val_split = min(max(val_split, 0.0), 0.9)
                n_val = int(round(n * val_split))
                n_train = max(0, n - n_val)
                idx = np.arange(n)
                rng = np.random.default_rng(int(cfg.get("seed", 42)))
                rng.shuffle(idx)
                val_idx = idx[:n_val]
                tr_idx = idx[n_val:]
                self.console.info(f"[split] image dataset using val_split={val_split:.2f} -> train={n_train}, val={n_val}")
                ds_train = torch.utils.data.Subset(ds_all, tr_idx.tolist())
                ds_val = torch.utils.data.Subset(ds_all, val_idx.tolist())
        except Exception as e:
            if PairedFlowDataset is not None and cfg.get("train_flow_csv"):
                self.console.warn(f"[image] Folder dataset failed ({e}); falling back to paired CSV '{cfg.get('train_flow_csv')}'")
                flow_csv = cfg.get("train_flow_csv", cfg.get("flow_csv", None))
                flow_csv_path = resolver.anchor(flow_csv) if isinstance(flow_csv, str) else None
                base_ds = PairedFlowDataset(
                    train_dir,
                    flow_csv_path,
                    image_size=image_size,
                    channels=channels,
                    mean=mean,
                    std=std,
                    augment=augment,
                    split_filter=None,
                    laminar_threshold=float(cfg.get("laminar_threshold", 2300.0)),
                    console=self.console,
                )
                idx = np.arange(len(base_ds))
                rng = np.random.default_rng(int(cfg.get("seed", 42)))
                rng.shuffle(idx)
                val_split = float(cfg.get("val_split", 0.2))
                val_split = min(max(val_split, 0.0), 0.9)
                n_val = int(round(len(base_ds) * val_split))
                va_idx = idx[:n_val].tolist()
                tr_idx = idx[n_val:].tolist()
                ds_train = torch.utils.data.Subset(base_ds, tr_idx)
                ds_val = torch.utils.data.Subset(base_ds, va_idx)
            else:
                raise

        limit_frac = float(cfg.get("debug_train_fraction", cfg.get("max_train_fraction", 1.0)))
        debug_enabled = bool(cfg.get("debug_limit_training", False))
        if debug_enabled and limit_frac > 0 and limit_frac < 1.0:
            n_keep = max(1, int(len(ds_train) * limit_frac))
            ds_train = torch.utils.data.Subset(ds_train, list(range(n_keep)))
            self.console.warn(f"[debug] Limiting training samples to {n_keep} ({limit_frac*100:.1f}%) for quick run.")

        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, val_loader

    def compute_and_lock_tiling(self, lock: bool = True) -> Tuple[int, int, int, int] | None:
        """Compute adaptive tile_size and stride from the smallest training source.

        Returns (tile_h, tile_w, stride_h, stride_w) for vector mode; returns None for image mode.
        """
        if self._is_image_mode():
            return None

        # Resolve training list from config
        train_field_cfg = self.config.get("train_field_path", self.config.get("field_path", ""))
        if isinstance(train_field_cfg, (list, tuple)):
            train_list = [str(p) for p in train_field_cfg]
        else:
            train_field = str(train_field_cfg or "")
            if ";" in train_field or "," in train_field:
                train_list = [s.strip() for s in train_field.replace(";", ",").split(",") if s.strip()]
            else:
                train_list = [train_field] if train_field else []
        if not train_list:
            raise RuntimeError("'train_field_path' or 'field_path' is required to compute tiling")

        resolver = PathResolver(
            Path(self.config.get("_config_path", Path(self.config.get("_config_dir", ".")) / "config.json")),
            outputs_root=str(self.config.get("outputs_root")) if isinstance(self.config.get("outputs_root"), str) else None,
        )
        patterns = ["*.csv", "*.npy", "*.npz"]
        inspect_paths: List[Path] = []
        for s in train_list:
            if not s:
                continue
            try:
                p = Path(s)
                looks_like_dir = (not p.suffix) or any(ch in str(p) for ch in ("*", "?", "["))
                if looks_like_dir:
                    expanded = resolver.expand_fields([s], recurse=True, patterns=patterns)
                    for e in expanded:
                        pe = Path(e)
                        if pe.is_file():
                            inspect_paths.append(pe)
                else:
                    if not p.is_absolute():
                        p = (Path(self.config.get("_config_dir", ".")) / p).resolve()
                    if p.is_file():
                        inspect_paths.append(p)
            except Exception:
                continue

        # Filter inspect_paths to likely vector field files (skip mapping CSVs)
        def _is_vector_candidate(pp: Path) -> bool:
            ext = pp.suffix.lower()
            if ext in (".npy", ".npz"):
                return True
            if ext == ".csv":
                try:
                    with pp.open("r", encoding="utf-8", errors="ignore") as f:
                        first = f.readline().lower()
                    if ("file_name" in first and "label" in first):
                        return False
                    return any(k in first for k in ("x", "y", "delta", "dx", "dy"))
                except Exception:
                    return False
            return False

        inspect_paths = [p for p in inspect_paths if _is_vector_candidate(p)]
        if not inspect_paths:
            raise RuntimeError("No training files found to compute tiling")

        min_H, min_W = None, None
        for p in inspect_paths:
            arr = load_vector_field(str(p))
            chw = ensure_chw(arr)
            H, W = int(chw.shape[1]), int(chw.shape[2])
            min_H = H if min_H is None else min(min_H, H)
            min_W = W if min_W is None else min(min_W, W)

        desired_th, desired_tw = tuple(self.config.get("tile_size", [256, 256]))
        stride_cfg = self.config.get("tile_stride", None)
        if isinstance(stride_cfg, (list, tuple)):
            sh, sw = int(stride_cfg[0]), int(stride_cfg[1])
        else:
            sh, sw = int(desired_th), int(desired_tw)

        cnn_min = self._cnn_min_side()
        pre_th = max(1, min(int(desired_th), int(min_H or desired_th)))
        pre_tw = max(1, min(int(desired_tw), int(min_W or desired_tw)))
        safe_th = max(cnn_min, pre_th)
        safe_tw = max(cnn_min, pre_tw)
        safe_sh = max(1, min(int(sh), safe_th))
        safe_sw = max(1, min(int(sw), safe_tw))

        if lock:
            # Stash diagnostics for UI/summary tables
            try:
                if min_H is not None and min_W is not None:
                    self.config["_tiling_min_source_hw"] = [int(min_H), int(min_W)]
                self.config["_tiling_cnn_min_side"] = int(cnn_min)
            except Exception:
                pass
            self.config["tile_size"] = [int(safe_th), int(safe_tw)]
            self.config["tile_stride"] = [int(safe_sh), int(safe_sw)]
            self._tiling_locked = True
            self.console.info(
                f"[tiling] Global tile_size=({safe_th},{safe_tw}) stride=({safe_sh},{safe_sw}) from smallest training source across {len(inspect_paths)} source(s)"
            )
        return int(safe_th), int(safe_tw), int(safe_sh), int(safe_sw)

    def _load_split(self, *, field_path: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load tiles and labels for a split. Supports:
        - Single CSV/path (string)
        - Multiple paths provided as list in config, or as a string separated by ';' or ','
        Uses embedded label column if present (via load_vector_field_tiles_with_labels),
        otherwise falls back to config-provided labels.
        """
        tile_size_cfg = tuple(self.config.get("tile_size", [256, 256]))
        stride_cfg = self.config.get("tile_stride", None)
        stride_cfg_tuple: Tuple[int, int] | None = None
        if isinstance(stride_cfg, (list, tuple)):
            stride_cfg_tuple = (int(stride_cfg[0]), int(stride_cfg[1]))
        add_mag = bool(self.config.get("add_magnitude", True))
        normalize = bool(self.config.get("normalize", True))
        limit_tiles = self.config.get("limit_tiles", None)

        paths: List[str] = parse_field_paths(field_path)
        if not paths:
            raise RuntimeError("field_path must be a non-empty string or list of strings")

        # Expand any directories based on data kind and file type patterns using PathResolver
        cfg_path_str = str(self.config.get("_config_path", Path(self.config.get("_config_dir", ".")) / "config.json"))
        try:
            cfg_path = Path(cfg_path_str)
        except Exception:
            cfg_path = Path(self.config.get("_config_dir", ".")) / "config.json"
        resolver = PathResolver(cfg_path, outputs_root=str(self.config.get("outputs_root")) if isinstance(self.config.get("outputs_root"), str) else None)

        # Determine data kind and patterns
        data_kind_key = f"{split}_data_kind"
        data_kind = (self.config.get(data_kind_key) or self.config.get("data_kind") or "vector").strip().lower()
        if data_kind not in ("vector", "image", "video"):
            self.console.warn(f"Unknown data kind '{data_kind}'; defaulting to 'vector'")
            data_kind = "vector"
        if data_kind == "vector":
            patterns = ["*.csv", "*.npy", "*.npz"]
        elif data_kind == "image":
            patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
        else:  # video
            patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv"]

        # If user gave base directories, optionally append the data_kind subfolder
        inputs_for_expand: List[str] = []
        for p in paths:
            ps = str(p)
            q = Path(ps)
            if q.suffix == "":
                # Likely a directory; if so, consider subfolder by data_kind variations
                try:
                    base = q if q.is_absolute() else (Path(self.config.get("_config_dir", ".")) / q).resolve()
                    candidates = [base / data_kind, base / f"{data_kind} data", base / f"{data_kind}s"]
                    chosen = None
                    for cand in candidates:
                        if cand.exists() and cand.is_dir():
                            chosen = cand
                            break
                    inputs_for_expand.append(str(chosen if chosen is not None else base))
                except Exception:
                    inputs_for_expand.append(ps)
            else:
                inputs_for_expand.append(ps)

        expanded = resolver.expand_fields(inputs_for_expand, recurse=True, patterns=patterns)
        if not expanded:
            # Fallback to anchoring raw paths
            base_dir = Path(self.config.get("_config_dir", "."))
            abs_paths: List[Path] = []
            for p in paths:
                pp = Path(p)
                if not pp.is_absolute():
                    pp = (base_dir / pp).resolve()
                abs_paths.append(pp)
        else:
            abs_paths = [Path(s) for s in expanded]

        # Guard: current training pipeline supports only vector fields (CSV/NPY/NPZ)
        if data_kind != "vector":
            raise RuntimeError(
                f"Training data_kind='{data_kind}' is not supported by the current vector-field CNN pipeline. "
                "Use data_kind='vector' or extend loaders to handle images/videos."
            )

        # Filter expanded paths to likely vector field inputs; skip mapping CSVs like file_name,label
        def _is_vector_candidate(pp: Path) -> bool:
            ext = pp.suffix.lower()
            if ext in (".npy", ".npz"):
                return True
            if ext == ".csv":
                try:
                    with pp.open("r", encoding="utf-8", errors="ignore") as f:
                        first = f.readline().lower()
                    if ("file_name" in first and "label" in first):
                        return False
                    return any(k in first for k in ("x", "y", "delta", "dx", "dy"))
                except Exception:
                    return False
            return False

        abs_paths = [pp for pp in abs_paths if pp.is_file() and _is_vector_candidate(pp)]

        # Filter to files only; fail early with a clear message if none found
        only_files = [pp for pp in abs_paths if pp.is_file()]
        if not only_files:
            examples = ", ".join(str(p) for p in abs_paths[:3]) if abs_paths else "<none>"
            patt_desc = ",".join(patterns)
            raise RuntimeError(
                f"No training data files found for split '{split}'. Searched under: {examples} (recurse=True) "
                f"with patterns [{patt_desc}] and data_kind='{data_kind}'. "
                f"If you changed folder structure, set train_field_path to the specific subfolder (e.g., '.../vector data')."
            )

        abs_paths = only_files

        # Determine tiling to use
        if getattr(self, "_tiling_locked", False):
            # Reuse precomputed tiling from config
            safe_th, safe_tw = int(tile_size_cfg[0]), int(tile_size_cfg[1])
            if stride_cfg_tuple is None:
                safe_stride = (safe_th, safe_tw)
            else:
                sh, sw = int(stride_cfg_tuple[0]), int(stride_cfg_tuple[1])
                safe_stride = (max(1, min(sh, safe_th)), max(1, min(sw, safe_tw)))
        else:
            # Compute a safe tiling across these inputs and persist to config
            min_H, min_W = None, None
            for pp in abs_paths:
                try:
                    arr = load_vector_field(str(pp))
                    chw = ensure_chw(arr)
                    H, W = int(chw.shape[1]), int(chw.shape[2])
                    min_H = H if min_H is None else min(min_H, H)
                    min_W = W if min_W is None else min(min_W, W)
                except Exception as e:
                    raise RuntimeError(f"Failed to inspect field size for '{pp}': {e}")

            desired_th, desired_tw = int(tile_size_cfg[0]), int(tile_size_cfg[1])
            # Cap by smallest provided training source; enforce CNN pooling minimum
            cnn_min = self._cnn_min_side()
            pre_th = max(1, min(desired_th, int(min_H or desired_th)))
            pre_tw = max(1, min(desired_tw, int(min_W or desired_tw)))
            safe_th = max(cnn_min, pre_th)
            safe_tw = max(cnn_min, pre_tw)
            if (min_H is not None and min_H < cnn_min) or (min_W is not None and min_W < cnn_min):
                self.console.warn(
                    f"[tiling] Smallest source ({min_H}x{min_W}) is below CNN min side {cnn_min} derived from pooling; proceeding with side=min(source,requested)."
                )
            if stride_cfg_tuple is None:
                safe_stride = (safe_th, safe_tw)
            else:
                sh, sw = int(stride_cfg_tuple[0]), int(stride_cfg_tuple[1])
                safe_stride = (max(1, min(sh, safe_th)), max(1, min(sw, safe_tw)))

            self.config["tile_size"] = [int(safe_th), int(safe_tw)]
            self.config["tile_stride"] = [int(safe_stride[0]), int(safe_stride[1])]
            self.console.info(
                f"[tiling] Using tile_size=({safe_th},{safe_tw}) stride=({safe_stride[0]},{safe_stride[1]}) across {len(abs_paths)} source(s)"
            )

        # Prepare accumulation lists
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        # Optional: per-source labels from config (e.g., train_source_labels / val_source_labels / source_labels)
        src_labels_key = f"{split}_source_labels"
        src_labels = self.config.get(src_labels_key, None)
        if src_labels is None:
            src_labels = self.config.get("source_labels", None)
        # Note: after directory expansion, abs_paths may be larger than 'paths'.
        # Only use src_labels if it matches abs_paths length exactly; otherwise we'll try auto-infer from filename.
        if isinstance(src_labels, (list, tuple)):
            try:
                src_labels = [int(v) for v in src_labels]
            except Exception:
                src_labels = None
        if isinstance(src_labels, list) and len(src_labels) not in (len(paths), len(abs_paths)):
            self.console.warn(
                f"{src_labels_key if f'{split}_source_labels' in self.config else 'source_labels'} length {len(src_labels)} does not match number of inputs (paths={len(paths)} expanded={len(abs_paths)}); will auto-infer by filename when possible"
            )
            src_labels = None

        # Decide a per-source cap to avoid starving later sources when limit_tiles is set
        if limit_tiles is None:
            per_source_cap: int | None = None
        else:
            try:
                total_cap = int(limit_tiles)
            except Exception:
                total_cap = None  # type: ignore[assignment]
            if total_cap is None or total_cap <= 0:
                per_source_cap = None
            else:
                num_sources = max(1, len(abs_paths))
                # Even split across sources; at least 1 tile per source when possible
                per_source_cap = max(1, total_cap // num_sources)

        for idx_p, pp in enumerate(abs_paths):
            used_limit = None if per_source_cap is None else int(per_source_cap)
            # Decide uniform label for this path (priority: exact per-file src_labels -> auto-infer from name)
            label_for_this_path: int | None = None
            if isinstance(src_labels, list) and len(src_labels) == len(abs_paths):
                try:
                    label_for_this_path = int(src_labels[idx_p])
                except Exception:
                    label_for_this_path = None
            else:
                name_l = str(pp).lower()
                # Map to raw labels 1/2 so that after normalization (y-1),
                # 0 -> turbulent, 1 -> laminar to match class_names ["turbulent","laminar"].
                if "turbulent" in name_l:
                    label_for_this_path = 1  # raw 1 -> index 0 (turbulent)
                elif "laminar" in name_l:
                    label_for_this_path = 2  # raw 2 -> index 1 (laminar)

            if label_for_this_path is not None:
                # Use uniform label per source based on config list
                Xp = load_vector_field_tiles(
                    path=str(pp),
                    tile_size=(int(safe_th), int(safe_tw)),
                    stride=(int(safe_stride[0]), int(safe_stride[1])),
                    add_magnitude=add_mag,
                    normalize=normalize,
                    limit_tiles=used_limit,
                )
                lbl = int(label_for_this_path)
                yp = np.full((Xp.shape[0],), lbl, dtype=np.int64)
            else:
                # Prefer embedded labels when available
                try:
                    Xp, yp = load_vector_field_tiles_with_labels(
                        path=str(pp),
                        tile_size=(int(safe_th), int(safe_tw)),
                        stride=(int(safe_stride[0]), int(safe_stride[1])),
                        add_magnitude=add_mag,
                        normalize=normalize,
                        limit_tiles=used_limit,
                        label_aggregation="majority",
                        drop_unlabeled=True,
                    )
                except Exception:
                    # Fallback: no embedded labels; use legacy path
                    Xp = load_vector_field_tiles(
                        path=str(pp),
                        tile_size=(int(safe_th), int(safe_tw)),
                        stride=(int(safe_stride[0]), int(safe_stride[1])),
                        add_magnitude=add_mag,
                        normalize=normalize,
                        limit_tiles=used_limit,
                    )
                    yp = _prepare_labels(n=Xp.shape[0], config=self.config, split=split)
            X_list.append(Xp)
            y_list.append(yp)
            # Respect an overall cap if provided, even with per-source caps
            if limit_tiles is not None and sum(x.shape[0] for x in X_list) >= int(limit_tiles):
                break

        if not X_list:
            raise RuntimeError("No tiles produced for split; check paths/tile_size/stride")
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        # Normalize labels to 0..num_classes-1 when labels are 1/2-based and num_classes=2
        if y.size > 0:
            if y.min() >= 1 and y.max() <= 2 and int(self.config.get("num_classes", 2)) == 2:
                y = (y - 1).astype(np.int64)
        # Class balance sanity check (warn if only one class present)
        try:
            classes, counts = np.unique(y, return_counts=True)
            if classes.size < int(self.config.get("num_classes", 2)):
                self.console.warn(
                    f"Training split '{split}' contains only classes {classes.tolist()} after sampling. "
                    "Consider increasing or removing limit_tiles or balancing sources."
                )
            else:
                # Light info to help diagnose imbalance without being too verbose
                hist = ", ".join([f"c{int(c)}={int(n)}" for c, n in zip(classes, counts)])
                self.console.info(f"[class-balance] {split}: {hist}")
        except Exception:
            pass
        return X, y

    def _train_one_epoch(self, loader: torch.utils.data.DataLoader, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        total_reg_mae = 0.0
        bar = None
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
            bar = _tqdm(loader, desc=f"train {epoch:03d}", unit="batch", dynamic_ncols=True, ascii=True, leave=False)
        except Exception:
            bar = None
        iter_loader = bar if bar is not None else loader
        for batch in iter_loader:
            x, y = batch
            x = x.to(self.device)
            self.optimizer.zero_grad()
            if self.is_multitask:
                logits_reg = self.model(x)
                logits = logits_reg["logits"]
                reg = logits_reg["reg"]
                labels = y["label"].to(self.device)
                reg_tgt = torch.stack([y["velocity"], y["reynolds"]], dim=1).to(self.device)
                loss_cls = self.criterion(logits, labels)
                loss_reg = self.reg_loss(reg, reg_tgt)
                loss = self.loss_w_cls * loss_cls + self.loss_w_reg * loss_reg
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    total_correct += int((pred == labels).sum().item())
                    total_seen += int(labels.numel())
                    total_loss += float(loss.item()) * int(labels.size(0))
                    total_reg_mae += float(torch.abs(reg - reg_tgt).sum().item())
            else:
                y = y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    total_correct += int((pred == y).sum().item())
                    total_seen += int(y.numel())
                    total_loss += float(loss.item()) * int(y.size(0))
            if bar is not None:
                try:
                    bar.set_postfix({"loss": f"{float(loss.item()):.4f}"})
                    bar.update(0)
                except Exception:
                    pass
        try:
            if bar is not None:
                bar.close()
        except Exception:
            pass
        avg_loss = total_loss / max(total_seen, 1)
        acc = total_correct / max(total_seen, 1)
        reg_mae = total_reg_mae / max(total_seen, 1) if self.is_multitask else 0.0
        return avg_loss, acc, reg_mae

    @torch.no_grad()
    def _eval(self, loader: torch.utils.data.DataLoader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        total_reg_mae = 0.0
        bar = None
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
            bar = _tqdm(loader, desc="val", unit="batch", dynamic_ncols=True, ascii=True, leave=False)
        except Exception:
            bar = None
        iter_loader = bar if bar is not None else loader
        for batch in iter_loader:
            x, y = batch
            x = x.to(self.device)
            if self.is_multitask:
                logits_reg = self.model(x)
                logits = logits_reg["logits"]
                reg = logits_reg["reg"]
                labels = y["label"].to(self.device)
                reg_tgt = torch.stack([y["velocity"], y["reynolds"]], dim=1).to(self.device)
                loss_cls = self.criterion(logits, labels)
                loss_reg = self.reg_loss(reg, reg_tgt)
                loss = self.loss_w_cls * loss_cls + self.loss_w_reg * loss_reg
                pred = torch.argmax(logits, dim=1)
                total_correct += int((pred == labels).sum().item())
                total_seen += int(labels.numel())
                total_loss += float(loss.item()) * int(labels.size(0))
                total_reg_mae += float(torch.abs(reg - reg_tgt).sum().item())
            else:
                y = y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                pred = torch.argmax(logits, dim=1)
                total_correct += int((pred == y).sum().item())
                total_seen += int(y.numel())
                total_loss += float(loss.item()) * int(y.size(0))
            if bar is not None:
                try:
                    bar.set_postfix({"loss": f"{float(loss.item()):.4f}"})
                    bar.update(0)
                except Exception:
                    pass
        try:
            if bar is not None:
                bar.close()
        except Exception:
            pass
        avg_loss = total_loss / max(total_seen, 1)
        acc = total_correct / max(total_seen, 1)
        reg_mae = total_reg_mae / max(total_seen, 1) if self.is_multitask else 0.0
        return avg_loss, acc, reg_mae

    def fit(self) -> Path:
        # Ensure folder structure once per training run
        try:
            if isinstance(self.config.get("_config_path"), (str, Path)):
                ensure_outputs_ready(Path(self.config["_config_path"]), self.config)
        except Exception:
            pass
        history: List[Dict[str, float]] = []
        csv_path, png_path = self._training_progress_paths()
        # Save a one-time settings snapshot at start of training
        try:
            self._write_settings_csv(self._training_settings_path())
        except Exception:
            try:
                self.console.warn("[training] Failed to write training settings snapshot.")
            except Exception:
                pass
        run_start = time.time()
        # Image-mode short-circuit: use image datasets and reuse training loop
        if self._is_image_mode():
            train_loader, val_loader = self._build_image_loaders()
            max_epochs = int(self.config.get("max_epochs", 50))
            best_val_metric = -1.0
            best_path: Path | None = None
            save_path = self._resolve_save_path()
            for epoch in range(1, max_epochs + 1):
                epoch_start = time.time()
                tr_loss, tr_acc, tr_mae = self._train_one_epoch(train_loader, epoch)
                va_loss, va_acc, va_mae = self._eval(val_loader)
                epoch_time = time.time() - epoch_start
                total_time = time.time() - run_start
                if self.is_multitask:
                    self.console.info(
                        f"epoch {epoch:03d} | train_loss={tr_loss:.6f} acc={tr_acc:.4f} reg_mae={tr_mae:.4f} | val_loss={va_loss:.6f} acc={va_acc:.4f} reg_mae={va_mae:.4f} | epoch_time={epoch_time:.2f}s total={total_time/60:.2f}m"
                    )
                else:
                    self.console.info(
                        f"epoch {epoch:03d} | train_loss={tr_loss:.6f} acc={tr_acc:.4f} | val_loss={va_loss:.6f} acc={va_acc:.4f} | epoch_time={epoch_time:.2f}s total={total_time/60:.2f}m"
                    )
                history.append(
                    {
                        "epoch": int(epoch),
                        "train_loss": float(tr_loss),
                        "train_acc": float(tr_acc),
                        "val_loss": float(va_loss),
                        "val_acc": float(va_acc),
                        "train_reg_mae": float(tr_mae),
                        "val_reg_mae": float(va_mae),
                        "lr": float(self._current_lr()),
                        "epoch_time_sec": float(epoch_time),
                        "total_time_sec": float(total_time),
                    }
                )
                warm = getattr(self.optimizer, "_warmup_scheduler", None)
                warm_epochs = int(getattr(self.optimizer, "_warmup_epochs", 0))
                if warm is not None and epoch <= warm_epochs:
                    warm.step()
                if self.scheduler is not None and self.scheduler_step_on == "epoch":
                    if hasattr(self.scheduler, "step"):
                        if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                            self.scheduler.step(va_loss)  # type: ignore[arg-type]
                        else:
                            self.scheduler.step()  # type: ignore[misc]
                metric = va_acc if not self.is_multitask else (-va_mae)
                if metric > best_val_metric:
                    best_val_metric = metric
                    best_path = self._save_checkpoint(path=save_path, tag="best")
            final_path = self._save_checkpoint(path=save_path, tag="final")
            self._finalize_training_history(history, csv_path, png_path)
            # Also write run log CSV next to the final checkpoint (matching stem)
            self._write_run_log_csv(history, final_path)
            try:
                total_time = time.time() - run_start
                self.console.info(f"[training] Total training time: {total_time:.2f}s ({total_time/60:.2f} min)")
            except Exception:
                pass
            return best_path or final_path

        # Resolve training/validation fields
        train_field_cfg = self.config.get("train_field_path", self.config.get("field_path", ""))
        # Allow list of paths or delimited string
        if isinstance(train_field_cfg, (list, tuple)):
            train_field = ";".join([str(p) for p in train_field_cfg])
            train_list = [str(p) for p in train_field_cfg]
        else:
            train_field = str(train_field_cfg or "")
            if ";" in train_field or "," in train_field:
                train_list = [s.strip() for s in train_field.replace(";", ",").split(",") if s.strip()]
            else:
                train_list = [train_field] if train_field else []
        if not train_field:
            raise RuntimeError("'train_field_path' or 'field_path' is required for training")
        val_cfg = self.config.get("test_field_path", "")
        if isinstance(val_cfg, (list, tuple)):
            val_field = ";".join([str(p) for p in val_cfg])
            val_list = [str(p) for p in val_cfg]
        else:
            val_field = str(val_cfg or "")
            if ";" in val_field or "," in val_field:
                val_list = [s.strip() for s in val_field.replace(";", ",").split(",") if s.strip()]
            else:
                val_list = [val_field] if val_field else []

        # Compute single global tiling across train+val and lock it
        # Expand any directories/globs to actual vector files to avoid opening directories
        resolver = PathResolver(
            Path(self.config.get("_config_path", Path(self.config.get("_config_dir", ".")) / "config.json")),
            outputs_root=str(self.config.get("outputs_root")) if isinstance(self.config.get("outputs_root"), str) else None,
        )
        patterns = ["*.csv", "*.npy", "*.npz"]
        inspect_paths: List[Path] = []
        for s in train_list:
            if not s:
                continue
            try:
                p = Path(s)
                looks_like_dir = (not p.suffix) or any(ch in str(p) for ch in ("*", "?", "["))
                if looks_like_dir:
                    expanded = resolver.expand_fields([s], recurse=True, patterns=patterns)
                    for e in expanded:
                        pe = Path(e)
                        if pe.is_file():
                            inspect_paths.append(pe)
                else:
                    if not p.is_absolute():
                        p = (Path(self.config.get("_config_dir", ".")) / p).resolve()
                    if p.is_file():
                        inspect_paths.append(p)
            except Exception:
                continue
        if inspect_paths:
            min_H, min_W = None, None
            # Filter inspect_paths to likely vector field files (skip mapping CSVs)
            def _is_vector_candidate(pp: Path) -> bool:
                ext = pp.suffix.lower()
                if ext in (".npy", ".npz"):
                    return True
                if ext == ".csv":
                    try:
                        with pp.open("r", encoding="utf-8", errors="ignore") as f:
                            first = f.readline().lower()
                        if ("file_name" in first and "label" in first):
                            return False
                        return any(k in first for k in ("x", "y", "delta", "dx", "dy"))
                    except Exception:
                        return False
                return False

            inspect_paths = [p for p in inspect_paths if _is_vector_candidate(p)]
            for p in inspect_paths:
                try:
                    arr = load_vector_field(str(p))
                    chw = ensure_chw(arr)
                    H, W = int(chw.shape[1]), int(chw.shape[2])
                    min_H = H if min_H is None else min(min_H, H)
                    min_W = W if min_W is None else min(min_W, W)
                except Exception as e:
                    raise RuntimeError(f"Failed to inspect field size for '{p}': {e}")

            desired_th, desired_tw = tuple(self.config.get("tile_size", [256, 256]))
            stride_cfg = self.config.get("tile_stride", None)
            if isinstance(stride_cfg, (list, tuple)):
                sh, sw = int(stride_cfg[0]), int(stride_cfg[1])
            else:
                sh, sw = int(desired_th), int(desired_tw)
            # Cap by smallest provided training/validation source; enforce CNN pooling minimum
            cnn_min = self._cnn_min_side()
            pre_th = max(1, min(int(desired_th), int(min_H or desired_th)))
            pre_tw = max(1, min(int(desired_tw), int(min_W or desired_tw)))
            safe_th = max(cnn_min, pre_th)
            safe_tw = max(cnn_min, pre_tw)
            if (min_H is not None and min_H < cnn_min) or (min_W is not None and min_W < cnn_min):
                self.console.warn(
                    f"[tiling] Smallest source ({min_H}x{min_W}) is below CNN min side {cnn_min} derived from pooling; proceeding with side=min(source,requested)."
                )
            safe_sh = max(1, min(int(sh), safe_th))
            safe_sw = max(1, min(int(sw), safe_tw))

            self.config["tile_size"] = [int(safe_th), int(safe_tw)]
            self.config["tile_stride"] = [int(safe_sh), int(safe_sw)]
            self._tiling_locked = True
            self.console.info(
                f"[tiling] Global tile_size=({safe_th},{safe_tw}) stride=({safe_sh},{safe_sw}) from smallest training source across {len(inspect_paths)} source(s)"
            )

        # Load tiles + labels
        x_train, y_train = self._load_split(field_path=train_field, split="train")

        # Validation: if explicit field provided and labels available, use it. Else split.
        use_holdout = False
        if val_field:
            try:
                x_val, y_val = self._load_split(field_path=val_field, split="val")
                use_holdout = True
            except Exception as e:
                self.console.warn(f"Validation field specified but labels missing or invalid: {e}. Falling back to split.")
                use_holdout = False

        if not use_holdout:
            val_split = float(self.config.get("val_split", 0.2))
            val_split = min(max(val_split, 0.0), 0.9)
            n = x_train.shape[0]
            n_val = int(round(n * val_split))
            n_train = n - n_val
            idx = np.arange(n)
            rng = np.random.default_rng(int(self.config.get("seed", 42)))
            rng.shuffle(idx)
            val_idx = idx[:n_val]
            tr_idx = idx[n_val:]
            x_val, y_val = x_train[val_idx], y_train[val_idx]
            x_train, y_train = x_train[tr_idx], y_train[tr_idx]

        # DataLoaders
        batch_size = int(self.config.get("batch_size", 16))
        num_workers = int(self.config.get("num_workers", 0))
        pin_memory = bool(self.config.get("pin_memory", False))
        train_loader = torch.utils.data.DataLoader(
            _TilesDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = torch.utils.data.DataLoader(
            _TilesDataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

        max_epochs = int(self.config.get("max_epochs", 50))
        best_val_acc = -1.0
        best_path: Path | None = None
        save_path = self._resolve_save_path()

        for epoch in range(1, max_epochs + 1):
            tr_loss, tr_acc = self._train_one_epoch(train_loader, epoch)
            va_loss, va_acc = self._eval(val_loader)
            self.console.info(
                f"epoch {epoch:03d} | train_loss={tr_loss:.6f} acc={tr_acc:.4f} | val_loss={va_loss:.6f} acc={va_acc:.4f}"
            )
            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(tr_loss),
                    "train_acc": float(tr_acc),
                    "val_loss": float(va_loss),
                    "val_acc": float(va_acc),
                    "lr": float(self._current_lr()),
                }
            )

            # Warmup scheduler hook (if configured by CnnOptim)
            warm = getattr(self.optimizer, "_warmup_scheduler", None)
            warm_epochs = int(getattr(self.optimizer, "_warmup_epochs", 0))
            if warm is not None and epoch <= warm_epochs:
                warm.step()

            if self.scheduler is not None:
                if self.scheduler_step_on == "epoch":
                    if hasattr(self.scheduler, "step"):
                        if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                            self.scheduler.step(va_loss)  # type: ignore[arg-type]
                        else:
                            self.scheduler.step()  # type: ignore[misc]

            # Save best checkpoint
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_path = self._save_checkpoint(path=save_path, tag="best")

        # Always save final
        final_path = self._save_checkpoint(path=save_path, tag="final")
        self._finalize_training_history(history, csv_path, png_path)
        return best_path or final_path

    def _resolve_save_path(self) -> Path:
        p = self.config.get("save_weights", None)
        if isinstance(p, str) and p.strip():
            out = Path(p)
            if not out.is_absolute():
                base = Path(self.config.get("_config_dir", "."))
                out = (base / out).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            return out
        # Default to config_dir/runs/model-YYYYmmdd-HHMMSS.pth (suffix replaced per tag)
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = Path(self.config.get("_config_dir", "."))
        out = (base / "runs" / f"model-{ts}.pth").resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def _save_checkpoint(self, *, path: Path, tag: str) -> Path:
        path = path.with_name(path.stem + f"-{tag}" + path.suffix)
        checkpoint = {
            "model_state": self.model.state_dict(),
            "config": self.config,
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, str(path))
        self.console.success(f"Saved checkpoint: {path}")
        return path

    def _current_lr(self) -> float:
        try:
            return float(self.optimizer.param_groups[0].get("lr", 0.0))
        except Exception:
            return 0.0

    def _training_progress_paths(self) -> Tuple[Path, Path]:
        # Keep training artifacts under outputs_root/<data_kind>/training/*
        cfg_path = Path(self.config.get("_config_path", Path(self.config.get("_config_dir", ".")) / "config.json"))
        outputs_root = str(self.config.get("outputs_root")) if isinstance(self.config.get("outputs_root"), str) else None
        resolver = PathResolver(cfg_path, outputs_root=outputs_root)
        subdir = "image" if self._is_image_mode() else "vector"
        training_dir = (resolver.outputs_root / subdir / "training").resolve()
        training_dir.mkdir(parents=True, exist_ok=True)
        return training_dir / "training_history.csv", training_dir / "training_progress.png"

    def _training_settings_path(self) -> Path:
        cfg_path = Path(self.config.get("_config_path", Path(self.config.get("_config_dir", ".")) / "config.json"))
        outputs_root = str(self.config.get("outputs_root")) if isinstance(self.config.get("outputs_root"), str) else None
        resolver = PathResolver(cfg_path, outputs_root=outputs_root)
        subdir = "image" if self._is_image_mode() else "vector"
        training_dir = (resolver.outputs_root / subdir / "training").resolve()
        training_dir.mkdir(parents=True, exist_ok=True)
        return training_dir / "training_settings.csv"

    def _finalize_training_history(self, history: List[Dict[str, float]], csv_path: Path, png_path: Path) -> None:
        if not history:
            return
        self._write_training_history_csv(history, csv_path)
        self._plot_training_progress(history, png_path)

    def _history_fieldnames(self, history: List[Dict[str, float]]) -> list[str]:
        base_fields = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"]
        extra_fields: list[str] = []
        if history and any("train_reg_mae" in h for h in history):
            extra_fields.append("train_reg_mae")
        if history and any("val_reg_mae" in h for h in history):
            extra_fields.append("val_reg_mae")
        if history and any("epoch_time_sec" in h for h in history):
            extra_fields.append("epoch_time_sec")
        if history and any("total_time_sec" in h for h in history):
            extra_fields.append("total_time_sec")
        return base_fields + extra_fields

    def _write_training_history_csv(self, history: List[Dict[str, float]], csv_path: Path) -> None:
        try:
            import csv as _csv

            csv_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = self._history_fieldnames(history)
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in history:
                    writer.writerow(row)
            self.console.success(f"[training] Saved history CSV to: {csv_path}")
        except Exception as e:
            try:
                self.console.warn(f"[training] Failed to write history CSV: {e}")
            except Exception:
                pass

    def _write_settings_csv(self, path: Path) -> None:
        """Write a CSV snapshot of the training configuration."""
        try:
            import csv as _csv

            flat: List[tuple[str, Any]] = []
            for k, v in sorted(self.config.items(), key=lambda kv: str(kv[0])):
                if str(k).startswith("_"):
                    continue
                flat.append((str(k), v))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = _csv.writer(f)
                writer.writerow(["key", "value"])
                for k, v in flat:
                    writer.writerow([k, v])
            self.console.success(f"[training] Saved training settings to: {path}")
        except Exception as e:
            try:
                self.console.warn(f"[training] Failed to save settings CSV: {e}")
            except Exception:
                pass

    def _write_run_log_csv(self, history: List[Dict[str, float]], model_path: Path) -> None:
        """Write a per-run CSV alongside the model checkpoint (same stem)."""
        try:
            import csv as _csv

            log_path = model_path.with_name(f"{model_path.stem}.log.csv")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = self._history_fieldnames(history)
            with log_path.open("w", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in history:
                    writer.writerow(row)
            self.console.success(f"[training] Saved run log CSV to: {log_path}")
        except Exception as e:
            try:
                self.console.warn(f"[training] Failed to write run log CSV: {e}")
            except Exception:
                pass

    def _plot_training_progress(self, history: List[Dict[str, float]], png_path: Path) -> None:
        model_name = str(self.config.get("model_name", self.config.get("model_type", ""))).strip()
        title = f"Training Progress - {model_name}" if model_name else "Training Progress"
        plot_training_progress(history, png_path, console=self.console, title=title)
