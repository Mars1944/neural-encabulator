from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

from vector_field_data import ensure_chw, load_vector_field, load_vector_field_tiles
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
        from cnn_optim import CnnOptim

        self.optim_builder = CnnOptim(self.config)
        self.optimizer, self.scheduler, self.scheduler_step_on = self.optim_builder.build(self.model)

    def _load_split(self, *, field_path: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
        tile_size = tuple(self.config.get("tile_size", [256, 256]))
        stride_cfg = self.config.get("tile_stride", None)
        stride: Tuple[int, int] | None = None
        if isinstance(stride_cfg, (list, tuple)):
            stride = (int(stride_cfg[0]), int(stride_cfg[1]))
        add_mag = bool(self.config.get("add_magnitude", True))
        normalize = bool(self.config.get("normalize", True))
        limit_tiles = self.config.get("limit_tiles", None)

        x_np = load_vector_field_tiles(
            path=field_path,
            tile_size=(int(tile_size[0]), int(tile_size[1])),
            stride=stride,
            add_magnitude=add_mag,
            normalize=normalize,
            limit_tiles=None if limit_tiles is None else int(limit_tiles),
        )
        y_np = _prepare_labels(n=x_np.shape[0], config=self.config, split=split)
        return x_np, y_np

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
        train_field = str(self.config.get("train_field_path", self.config.get("field_path", "")) or "")
        if not train_field:
            raise RuntimeError("'train_field_path' or 'field_path' is required for training")
        val_field = str(self.config.get("test_field_path", "") or "")

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
