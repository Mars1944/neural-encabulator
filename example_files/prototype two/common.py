from __future__ import annotations

"""
common.py — shared utilities for prototype two

Purpose
- Centralize device selection, config loading/validation, small model helpers,
  visualization helpers, and path resolution so behavior can be tuned in one
  place and picked up by infer, viewer, and training scripts.

Sections
- Imports
- Device selection helpers
- Config manager (load/validate/normalize)
- Model utilities (channel inference, labels loader)
- Visualization helpers (HeatmapGenerator)
- Path resolution (PathResolver)
"""

# ============================================================================
# Imports
# ============================================================================
import json
import random
from pathlib import Path
from typing import Any, Dict

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyTorch is not installed. Please install torch to proceed.") from e
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
try:
    import torch.backends.cudnn as cudnn  # type: ignore
except Exception:  # pragma: no cover
    cudnn = None  # type: ignore

from vector_field_data import ensure_chw, load_vector_field


# ---------------------------------------------------------------------------
# Device selection helpers
# ---------------------------------------------------------------------------
class DeviceSelector:
    """Selects the best-available device based on a preference string.

    Preference examples: "auto", "cpu", "cuda", "cuda:0", "mps"
    """
    @staticmethod
    def choose(pref: str | None) -> torch.device:
        p = (pref or "auto").lower()

        def mps_available() -> bool:
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        def cuda_available(idx: int | None = None) -> bool:
            if not torch.cuda.is_available():
                return False
            if idx is None:
                return True
            try:
                return 0 <= idx < torch.cuda.device_count()
            except Exception:
                return False

        if p.startswith("cuda"):
            idx: int | None = None
            if ":" in p:
                try:
                    idx = int(p.split(":", 1)[1])
                except Exception:
                    idx = None
            if cuda_available(idx):
                return torch.device(f"cuda:{idx}" if idx is not None else "cuda")
            return torch.device("cpu")

        if p == "mps":
            if mps_available():
                return torch.device("mps")
            return torch.device("cpu")

        if p == "cpu":
            return torch.device("cpu")

        if cuda_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def format(device: torch.device) -> str:
        if device.type == "cuda":
            try:
                idx = device.index if device.index is not None else (
                    torch.cuda.current_device() if torch.cuda.is_available() else None
                )
                if idx is not None:
                    name = torch.cuda.get_device_name(idx)
                    return f"cuda:{idx} ({name})"
            except Exception:
                pass
            return "cuda"
        if device.type == "mps":
            return "mps (Apple Silicon)"
        return "cpu"


# ---------------------------------------------------------------------------
# Config manager
# ---------------------------------------------------------------------------
class ConfigManager:
    """Load, validate, and normalize a JSON config for the app.

    - Anchors relative paths to the config directory
    - Applies sane defaults and light validation
    - Sets deterministic/runtime flags (best-effort)
    """
    def __init__(self, config_path: Path) -> None:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        self.path = config_path
        self.config: Dict[str, Any] = self._load(self.path)
        self._validate_and_normalize()
        self.device: torch.device = DeviceSelector.choose(str(self.config.get("device", "auto")))
        self._set_seed(int(self.config.get("seed", 42)))
        self._apply_runtime_flags()

    @staticmethod
    def _load(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        if np is not None:
            try:
                np.random.seed(seed)  # type: ignore[attr-defined]
            except Exception:
                pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _apply_runtime_flags(self) -> None:
        deterministic = bool(self.config.get("deterministic", False))
        try:
            torch.use_deterministic_algorithms(deterministic)
        except Exception:
            pass
        if cudnn is not None:
            try:
                cudnn.deterministic = deterministic  # type: ignore[attr-defined]
                cudnn.benchmark = not deterministic  # type: ignore[attr-defined]
            except Exception:
                pass
        prec = str(self.config.get("matmul_precision", "default")).lower()
        try:
            if prec in {"highest", "high", "medium"}:
                torch.set_float32_matmul_precision(prec)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _validate_and_normalize(self) -> None:
        config = self.config
        # Merge known nested sections onto the top-level without overwriting
        # explicit top-level keys. This allows a consolidated config layout.
        for sec in ("paths", "data", "model", "training", "inference", "viewer"):
            sec_dict = config.get(sec, None)
            if isinstance(sec_dict, dict):
                for k, v in sec_dict.items():
                    if k not in config:
                        config[k] = v
        # Expected keys union from training/inference paths
        expected: Dict[str, str] = {
            "seed": "int",
            "device": "str",
            "model_type": "str",
            "num_classes": "int",
            "conv_channels": "list[int]",
            "kernel_size": "int",
            "pool_every": "int",
            "dropout": "float",
            "use_batchnorm": "bool",
            "dilations": "int|list[int]",
            "learning_rate": "float",
            "weight_decay": "float",
            "field_path": "str",
            "tile_size": "list[int,int]",
            "tile_stride": "list[int,int]",
            "add_magnitude": "bool",
            "normalize": "bool",
            "limit_tiles": "int|null",
            "deterministic": "bool",
            "matmul_precision": "str",
            "_doc": "any",
            "_tips": "any",
            "optimizer": "str",
            "scheduler": "str",
            "max_epochs": "int",
            "save_weights": "str",
            "train_field_path": "str",
            "test_field_path": "str",
            "use_test": "bool",
            "no_save": "bool",
            # Training/inference extras
            "batch_size": "int",
            "num_workers": "int",
            "pin_memory": "bool",
            "val_split": "float",
            "train_labels_csv": "str",
            "val_labels_csv": "str",
            "train_label_all": "int",
            "val_label_all": "int",
            "train_labels": "list[int]",
            "val_labels": "list[int]",
            "labels_csv": "str",
            "label_all": "int",
            "labels": "list[int]",
        }
        for k in list(config.keys()):
            if k not in expected:
                # Only warn; do not error to allow forward-compat keys
                print(f"[warn] Unknown config key: '{k}'")

        # Common defaults
        config.setdefault("model_type", "cnn")
        config.setdefault("num_classes", 2)
        config.setdefault("conv_channels", [32, 64, 128])
        config.setdefault("kernel_size", 3)
        config.setdefault("pool_every", 1)
        config.setdefault("dropout", 0.0)
        config.setdefault("use_batchnorm", True)
        config.setdefault("learning_rate", 1e-3)
        config.setdefault("weight_decay", 0.0)
        config.setdefault("tile_size", [256, 256])

        # Resolve paths relative to the config file
        for key in ("field_path", "train_field_path", "test_field_path", "save_weights"):
            fp = config.get(key, "")
            if isinstance(fp, str) and fp.strip():
                p = Path(fp)
                if not p.is_absolute():
                    p = (self.path.parent / p).resolve()
                    config[key] = str(p)

        # Basic validation
        ts = config.get("tile_size", [256, 256])
        if not (isinstance(ts, (list, tuple)) and len(ts) == 2):
            raise ValueError("tile_size must be [h, w]")
        if any(int(x) <= 0 for x in ts):
            raise ValueError("tile_size entries must be positive")
        stride = config.get("tile_stride", None)
        if stride is not None:
            if not (isinstance(stride, (list, tuple)) and len(stride) == 2):
                raise ValueError("tile_stride must be [sh, sw] when provided")
            if any(int(x) <= 0 for x in stride):
                raise ValueError("tile_stride entries must be positive")


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def infer_input_channels_from_field(config: Dict[str, Any], field_path: str) -> int:
    arr = load_vector_field(field_path, mmap=True)
    chw = ensure_chw(arr)
    c = int(chw.shape[0])
    if bool(config.get("add_magnitude", True)) and c >= 2:
        c += 1
    return c


def load_labels_from_csv(path: Path) -> "np.ndarray":
    import csv
    import numpy as _np

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Empty labels CSV: {path}")
    # Detect header by trying to parse the second column
    start = 0
    try:
        _ = int(rows[0][1])
    except Exception:
        start = 1
    labels = []
    for r in rows[start:]:
        if len(r) < 2:
            continue
        try:
            labels.append(int(r[1]))
        except Exception:
            continue
    if not labels:
        raise RuntimeError(f"No labels parsed from {path}")
    return _np.asarray(labels, dtype=_np.int64)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
class HeatmapGenerator:
    """
    Generates per-tile and stitched heatmaps from per-tile scalar values.
    Uses matplotlib if available, otherwise falls back to PIL, else saves .npy.
    """

    def __init__(self, cmap: str = "viridis") -> None:
        self.cmap = cmap
        try:
            import matplotlib.pyplot as plt  # type: ignore

            self._plt = plt
        except Exception:
            self._plt = None
        try:
            from PIL import Image  # type: ignore

            self._PIL_Image = Image
        except Exception:
            self._PIL_Image = None

    def save_per_tile(self, vals: "np.ndarray", out_dir: Path, tile_img_size: int = 16) -> None:
        import numpy as _np

        out_dir.mkdir(parents=True, exist_ok=True)
        if self._plt is not None:
            plt = self._plt
            for i, v in enumerate(vals):
                plt.figure(figsize=(2, 2))
                plt.imshow(_np.array([[v]]), vmin=0.0, vmax=1.0, cmap=self.cmap)
                plt.axis("off")
                plt.savefig(out_dir / f"tile_{i:05d}.png", bbox_inches="tight", pad_inches=0)
                plt.close()
        elif self._PIL_Image is not None:
            Image = self._PIL_Image
            for i, v in enumerate(vals):
                gray = int(_np.clip(v, 0.0, 1.0) * 255.0)
                img = Image.fromarray(_np.full((tile_img_size, tile_img_size), gray, dtype=_np.uint8), mode="L")
                img.save(out_dir / f"tile_{i:05d}.png")
        else:
            _np.save(str(out_dir / "tile_values.npy"), vals)

    def save_stitched(
        self,
        vals: "np.ndarray",
        out_path: Path,
        field_hw: tuple[int, int],
        tile_size: tuple[int, int],
        stride: tuple[int, int] | None,
    ) -> None:
        import numpy as _np

        H, W = field_hw
        th, tw = tile_size
        if stride is None:
            sh, sw = th, tw
        else:
            sh, sw = int(stride[0]), int(stride[1])
        n_rows = (H - th) // sh + 1 if H >= th else 0
        n_cols = (W - tw) // sw + 1 if W >= tw else 0
        if n_rows * n_cols != int(vals.shape[0]) or n_rows <= 0 or n_cols <= 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _np.save(str(out_path.with_suffix(".npy")), vals)
            return

        grid = vals.reshape(n_rows, n_cols)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self._plt is not None:
            plt = self._plt
            plt.figure(figsize=(max(4, n_cols / 4), max(4, n_rows / 4)))
            plt.imshow(grid, vmin=0.0, vmax=1.0, cmap=self.cmap)
            plt.colorbar(label="probability")
            plt.title("Tile confidence heatmap")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
        elif self._PIL_Image is not None:
            Image = self._PIL_Image
            arr = (_np.clip(grid, 0.0, 1.0) * 255.0).astype(_np.uint8)
            Image.fromarray(arr, mode="L").resize((n_cols * 16, n_rows * 16), resample=Image.NEAREST).save(out_path)
        else:
            _np.save(str(out_path.with_suffix(".npy")), grid)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
class PathResolver:
    """Helper for resolving and defaulting paths relative to a config file.

    Optionally honors an outputs_root (relative or absolute) so callers can
    redirect all derived outputs by changing a single config key.
    """

    def __init__(self, cfg_path: Path, outputs_root: str | None = None) -> None:
        self.cfg_path = cfg_path
        self.base_dir = cfg_path.parent
        # Compute outputs_root path
        if outputs_root is None or str(outputs_root).strip() == "":
            self.outputs_root = (self.base_dir / "data" / "outputs").resolve()
        else:
            p = Path(outputs_root)
            self.outputs_root = p if p.is_absolute() else (self.base_dir / p).resolve()

    def anchor(self, p: str | Path | None) -> Path | None:
        if p is None:
            return None
        q = Path(p)
        return q if q.is_absolute() else (self.base_dir / q).resolve()

    def default_heatmaps_dir(self) -> Path:
        return (self.outputs_root / "heatmaps").resolve()

    def default_stitched_heatmap(self) -> Path:
        return (self.outputs_root / "stitched_heatmap.png").resolve()

    def default_cm_csv(self) -> Path:
        return (self.outputs_root / "confusion_matrix.csv").resolve()

    def default_cm_png(self) -> Path:
        return (self.outputs_root / "confusion_matrix.png").resolve()

    def decide_heatmap_paths(
        self,
        *,
        generate: bool,
        heatmaps_dir: str | None,
        stitched: str | None,
    ) -> tuple[Path | None, Path | None]:
        hd = self.anchor(heatmaps_dir)
        sp = self.anchor(stitched)
        if generate:
            if hd is None:
                hd = self.default_heatmaps_dir()
            if sp is None:
                sp = self.default_stitched_heatmap()
        else:
            if heatmaps_dir is None:
                hd = None
            if stitched is None:
                sp = None
        return hd, sp

    def decide_cm_paths(
        self,
        *,
        generate: bool,
        cm_csv: str | None,
        cm_png: str | None,
    ) -> tuple[Path | None, Path | None]:
        cc = self.anchor(cm_csv)
        cp = self.anchor(cm_png)
        if generate:
            if cc is None:
                cc = self.default_cm_csv()
            if cp is None:
                cp = self.default_cm_png()
        else:
            if cm_csv is None:
                cc = None
            if cm_png is None:
                cp = None
        return cc, cp

    def resolve_field_path(self, config: Dict[str, Any]) -> str | None:
        fp = config.get("field_path", None)
        if isinstance(fp, str) and fp.strip():
            return str(self.anchor(fp))
        use_test = bool(config.get("use_test", False))
        key = "test_field_path" if use_test else "train_field_path"
        alt = config.get(key, None)
        if isinstance(alt, str) and alt.strip():
            return str(self.anchor(alt))
        try:
            vec_cfg_path = (self.base_dir / "cnn_vector_config.json").resolve()
            if vec_cfg_path.exists():
                import json as _json
                with vec_cfg_path.open("r", encoding="utf-8") as f:
                    vcfg = _json.load(f)
                fp2 = vcfg.get("field_path", "")
                if isinstance(fp2, str) and fp2.strip():
                    return str(self.anchor(fp2))
                alt2 = vcfg.get(key, "")
                if isinstance(alt2, str) and alt2.strip():
                    return str(self.anchor(alt2))
        except Exception:
            pass
        return None


