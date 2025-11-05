from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, List

import numpy as np
import torch

from vector_field_data import (
    ensure_chw,
    load_vector_field,
    load_vector_field_tiles,
    load_vector_field_tiles_with_labels,
)
from common import load_labels_from_csv


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
        self.criterion = torch.nn.CrossEntropyLoss()
        # When True, _load_split will reuse tile_size/tile_stride from config without recomputing
        self._tiling_locked: bool = False
        from cnn_optim import CnnOptim

        self.optim_builder = CnnOptim(self.config)
        self.optimizer, self.scheduler, self.scheduler_step_on = self.optim_builder.build(self.model)

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

        paths: List[str] = []
        if isinstance(field_path, str):
            s = field_path.strip()
            if ";" in s or "," in s:
                for part in s.replace(";", ",").split(","):
                    if part.strip():
                        paths.append(part.strip())
            else:
                paths.append(s)
        else:
            # Should not happen given signature, but keep robust
            try:
                paths = list(field_path)  # type: ignore[arg-type]
            except Exception:
                raise RuntimeError("field_path must be a string or list of strings")

        # Anchor all paths relative to config dir
        base_dir = Path(self.config.get("_config_dir", "."))
        abs_paths: List[Path] = []
        for p in paths:
            pp = Path(p)
            if not pp.is_absolute():
                pp = (base_dir / pp).resolve()
            abs_paths.append(pp)

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
            safe_th = max(1, min(desired_th, int(min_H or desired_th)))
            safe_tw = max(1, min(desired_tw, int(min_W or desired_tw)))
            if stride_cfg_tuple is None:
                safe_stride = (safe_th, safe_tw)
            else:
                sh, sw = int(stride_cfg_tuple[0]), int(stride_cfg_tuple[1])
                safe_stride = (max(1, min(sh, safe_th)), max(1, min(sw, safe_tw)))

            self.config["tile_size"] = [int(safe_th), int(safe_tw)]
            self.config["tile_stride"] = [int(safe_stride[0]), int(safe_stride[1])]
            print(
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
        if isinstance(src_labels, (list, tuple)) and len(src_labels) != len(paths):
            print(
                f"[warn] {src_labels_key if f'{split}_source_labels' in self.config else 'source_labels'} length {len(src_labels)} doesn't match number of paths {len(paths)}; ignoring"
            )
            src_labels = None

        for idx_p, pp in enumerate(abs_paths):
            used_limit = None if limit_tiles is None else int(max(0, limit_tiles - sum(x.shape[0] for x in X_list)))
            if isinstance(src_labels, (list, tuple)):
                # Use uniform label per source based on config list
                Xp = load_vector_field_tiles(
                    path=str(pp),
                    tile_size=(int(safe_th), int(safe_tw)),
                    stride=(int(safe_stride[0]), int(safe_stride[1])),
                    add_magnitude=add_mag,
                    normalize=normalize,
                    limit_tiles=used_limit,
                )
                lbl = int(src_labels[idx_p])
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
        return X, y

    def _train_one_epoch(self, loader: torch.utils.data.DataLoader, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                total_correct += int((pred == y).sum().item())
                total_seen += int(y.numel())
                total_loss += float(loss.item()) * int(y.size(0))
        avg_loss = total_loss / max(total_seen, 1)
        acc = total_correct / max(total_seen, 1)
        return avg_loss, acc

    @torch.no_grad()
    def _eval(self, loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            pred = torch.argmax(logits, dim=1)
            total_correct += int((pred == y).sum().item())
            total_seen += int(y.numel())
            total_loss += float(loss.item()) * int(y.size(0))
        avg_loss = total_loss / max(total_seen, 1)
        acc = total_correct / max(total_seen, 1)
        return avg_loss, acc

    def fit(self) -> Path:
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
        base_dir = Path(self.config.get("_config_dir", "."))
        inspect_paths: List[Path] = []
        for s in train_list + val_list:
            if not s:
                continue
            p = Path(s)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            inspect_paths.append(p)
        if inspect_paths:
            min_H, min_W = None, None
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
            safe_th = max(1, min(int(desired_th), int(min_H or desired_th)))
            safe_tw = max(1, min(int(desired_tw), int(min_W or desired_tw)))
            safe_sh = max(1, min(int(sh), safe_th))
            safe_sw = max(1, min(int(sw), safe_tw))

            self.config["tile_size"] = [int(safe_th), int(safe_tw)]
            self.config["tile_stride"] = [int(safe_sh), int(safe_sw)]
            self._tiling_locked = True
            print(
                f"[tiling] Global tile_size=({safe_th},{safe_tw}) stride=({safe_sh},{safe_sw}) across {len(inspect_paths)} source(s)"
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
                print(f"[warn] Validation field specified but labels missing or invalid: {e}. Falling back to split.")
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
            print(
                f"epoch {epoch:03d} | train_loss={tr_loss:.6f} acc={tr_acc:.4f} | val_loss={va_loss:.6f} acc={va_acc:.4f}"
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
        print(f"Saved checkpoint: {path}")
        return path
