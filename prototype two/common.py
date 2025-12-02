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
from console import console_from_config
try:
    from setup_folders import ensure_dirs_from_config  # type: ignore
except Exception:  # pragma: no cover
    ensure_dirs_from_config = None  # type: ignore[misc]


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
# Filesystem helpers
# ---------------------------------------------------------------------------
def ensure_parent(path: Path) -> None:
    """Create parent directories for a path if missing."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def ensure_outputs_ready(cfg_path: Path, cfg: Dict[str, Any]) -> None:
    """
    Centralized hook to ensure expected folder structure exists before writing outputs.

    Uses setup_folders.ensure_dirs_from_config when available; no-ops on failure.
    """
    try:
        if ensure_dirs_from_config is not None:
            ensure_dirs_from_config(cfg_path, cfg)
    except Exception:
        pass


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
        # Allow TF32 on Ampere/ADA when requested
        try:
            allow_tf32 = bool(self.config.get("allow_tf32", False))
            try:
                torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # type: ignore[attr-defined]
            except Exception:
                pass
            if cudnn is not None:
                try:
                    cudnn.allow_tf32 = allow_tf32  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

    def _validate_and_normalize(self) -> None:
        config = self.config
        # Merge known nested sections onto the top-level without overwriting
        # explicit top-level keys. This allows a consolidated config layout.
        for sec in ("paths", "data", "model", "training", "inference", "viewer", "reports"):
            sec_dict = config.get(sec, None)
            if isinstance(sec_dict, dict):
                # For 'inference', avoid lifting nested subsections ('common','image','vector') to top-level here;
                # they are handled explicitly below.
                if sec == "inference" and any(k in sec_dict for k in ("common", "image", "vector")):
                    pass
                else:
                    for k, v in sec_dict.items():
                        if k not in config:
                            config[k] = v

        # Support new inference layout: inference = { common: {..., data_kind: image|vector}, image: {...}, vector: {...} }
        inf = self.config.get("inference", None)
        if isinstance(inf, dict) and any(k in inf for k in ("common", "image", "vector")):
            common = inf.get("common", {}) if isinstance(inf.get("common", {}), dict) else {}
            # Merge common onto top-level (do not overwrite)
            for k, v in common.items():
                if k not in self.config:
                    self.config[k] = v
            # Determine mode from inference.common (prefer explicit), else fall back to existing keys
            mode_raw = str(common.get("data_kind", common.get("mode", self.config.get("data_kind", "vector")))).lower()
            selected = "image" if mode_raw.startswith("image") else "vector"
            # Merge selected subsection
            sub = inf.get(selected, {}) if isinstance(inf.get(selected, {}), dict) else {}
            for k, v in sub.items():
                if k not in self.config:
                    self.config[k] = v
            # Ensure data_kind reflects the selected mode
            self.config.setdefault("data_kind", selected)
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
            # Data selection helpers
            "data_kind": "str",  # one of: vector, image, video
            "train_data_kind": "str",
            "val_data_kind": "str",
            "test_data_kind": "str",
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
            # Newer keys to support multi-source training without CSVs
            "train_source_labels": "list[int]",
            "val_source_labels": "list[int]",
            "source_labels": "list[int]",
            # Common consolidated-config sections and viewer/inference keys
            "paths": "dict",
            "data": "dict",
            "model": "dict",
            "training": "dict",
            "inference": "dict",
            "viewer": "dict",
            "reports": "dict",
            "outputs_root": "str",
            # Directory expansion / discovery
            "recurse": "bool",
            "generate_heatmap": "bool",
            "generate_confusion_matrix": "bool",
            "show_heatmap": "bool",
            "heatmap_class": "int",
            "cm_csv": "str",
            "cm_png": "str",
            "heatmaps_dir": "str",
            "stitched_heatmap": "str",
            "labels_npy": "str",
            "class_names": "list[str]",
            # Inference tiling overrides
            "inference_tile_size": "list[int,int]",
            "inference_tile_stride": "list[int,int]",
            # Additional inference/CLI convenience keys
            "file_labels": "str",
            "file_labels_manifest": "str",
            "file_summary": "str",
            "predictions_csv": "str",
            "image_root": "str",
            "subdir": "str",
            "auto_dump_labels_template": "bool",
            "cm_debug": "bool",
            "heatmap_tile_size": "int",
            # Logging controls
            "log_to_file": "bool",
            "log_file": "str",
            "viewer_values_csv": "str",
            "viewer_title": "str",
            "viewer_cmap": "str",
            "viewer_no_show": "bool",
            "viewer_save": "str",
            "viewer_rows": "int",
            "viewer_cols": "int",
            # Image dataset support
            "train_image_dir": "str",
            "val_image_dir": "str",
            "test_image_dir": "str",
            "image_size": "list[int,int]",
            "image_channels": "int",
            "image_mean": "list[float]",
            "image_std": "list[float]",
            "augment": "dict",
            # Inference performance tuning keys (top-level or under inference)
            "image_infer_batch_size": "int",
            "mixed_precision": "str",
            "channels_last": "bool",
            "prefetch_workers": "int",
            "torch_compile": "bool",
            "use_inference_mode": "bool",
            "allow_tf32": "bool",
            "use_opencv_loader": "bool",
            "use_cuda_graphs": "bool",
            "plot_top_n": "int",
            # Image inference filters
            "image_include_classes": "list[str]",
            "include_classes": "list[str]",
            # Multitask/image + debug support
            "task": "str",
            "multitask": "bool",
            "train_flow_csv": "str",
            "flow_csv": "str",
            "debug_limit_training": "bool",
            "debug_train_fraction": "float",
            "max_train_fraction": "float",

        }
        for k in list(config.keys()):
            if k not in expected:
                # Only warn; do not error to allow forward-compat keys
                console_from_config(config).warn(f"Unknown config key: '{k}'")

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
        # Image defaults
        config.setdefault("image_size", [256, 256])
        config.setdefault("image_channels", 3)
        config.setdefault("image_mean", [0.5])
        config.setdefault("image_std", [0.5])

        # Resolve paths relative to the config file (support str and list[str])
        for key in ("field_path", "train_field_path", "test_field_path", "save_weights",
                    "train_image_dir", "val_image_dir", "test_image_dir", "train_flow_csv", "flow_csv"):
            fp = config.get(key, "")
            if isinstance(fp, str) and fp.strip():
                p = Path(fp)
                if not p.is_absolute():
                    p = (self.path.parent / p).resolve()
                    config[key] = str(p)
            elif isinstance(fp, (list, tuple)):
                resolved_list = []
                for item in fp:
                    if isinstance(item, str) and item.strip():
                        q = Path(item)
                        if not q.is_absolute():
                            q = (self.path.parent / q).resolve()
                        resolved_list.append(str(q))
                if resolved_list:
                    config[key] = resolved_list

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

        # Validate image_size
        isz = config.get("image_size", [256, 256])
        if not (isinstance(isz, (list, tuple)) and len(isz) == 2):
            raise ValueError("image_size must be [h, w]")
        if any(int(x) <= 0 for x in isz):
            raise ValueError("image_size entries must be positive")


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
        # Optional progress bar
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
            _iter = _tqdm(range(len(vals)), desc="heatmaps", unit="tile")
        except Exception:
            _iter = range(len(vals))

        if self._plt is not None:
            plt = self._plt
            for i in _iter:
                v = vals[i]
                plt.figure(figsize=(2, 2))
                plt.imshow(_np.array([[v]]), vmin=0.0, vmax=1.0, cmap=self.cmap)
                plt.axis("off")
                plt.savefig(out_dir / f"tile_{i:05d}.png", bbox_inches="tight", pad_inches=0)
                plt.close()
        elif self._PIL_Image is not None:
            Image = self._PIL_Image
            for i in _iter:
                v = vals[i]
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

    # Data-kind helper
    @staticmethod
    def resolve_split_kind(config: Dict[str, Any], split: str) -> str:
        key = f"{split}_data_kind"
        # Allow inference.common.data_kind to guide selection when present
        inf = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}
        inf_common = inf.get("common", {}) if isinstance(inf.get("common", {}), dict) else {}
        v = str(config.get(key, inf_common.get("data_kind", config.get("data_kind", "vector")))).lower()
        return "image" if v.startswith("image") else "vector"

    def expand_fields(
        self,
        inputs: list[str],
        recurse: bool = False,
        patterns: "list[str] | None" = None,
    ) -> list[str]:
        """Anchor and expand a list of file/dir/glob inputs to concrete file paths.

        - Anchors relative to the config directory
        - Expands glob patterns (supports **)
        - If a directory is given, collects files matching patterns
        - Default patterns: ["*.csv", "*.npy", "*.npz"]
        """
        pats = patterns if isinstance(patterns, list) and patterns else ["*.csv", "*.npy", "*.npz"]
        out: list[str] = []
        for raw in inputs:
            p = Path(raw)
            if not p.is_absolute():
                p = (self.base_dir / p).resolve()
            s = str(p)
            # Glob pattern
            if any(ch in s for ch in "*?[]"):
                parent = p.parent
                pattern = p.name
                for m in parent.glob(pattern):
                    if m.is_file():
                        out.append(str(m.resolve()))
                continue
            # Directory expansion
            if p.is_dir():
                for patt in pats:
                    it = p.rglob(patt) if recurse else p.glob(patt)
                    for m in it:
                        if m.is_file():
                            out.append(str(m.resolve()))
                continue
            # File
            if p.exists() and p.is_file():
                out.append(str(p))
        # Deduplicate + stable order
        return sorted({q for q in out})

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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def parse_field_paths(value: Any) -> list[str]:
    """
    Normalize a field path specification into a list of strings.

    Accepts:
    - str: possibly delimited by ',' or ';' -> split and strip empty items
    - list/tuple: keep string items that are non-empty after strip
    - other: returns []
    """
    out: list[str] = []
    try:
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return []
            parts = [p.strip() for p in s.replace(";", ",").split(",")]
            out = [p for p in parts if p]
        elif isinstance(value, (list, tuple)):
            out = [str(p).strip() for p in value if isinstance(p, (str, Path)) and str(p).strip()]
    except Exception:
        return []
    return out


def choose_fields(config: Dict[str, Any], field_path_arg: Any = None) -> list[str]:
    """
    Choose field paths in priority order and return as a list[str].

    Priority:
      1) CLI/explicit argument `field_path_arg`
      2) config["field_path"]
      3) config["test_field_path"] if config["use_test"] else config["train_field_path"]
    """
    fields = parse_field_paths(field_path_arg)
    if fields:
        return fields
    fields = parse_field_paths(config.get("field_path", None))
    if fields:
        return fields
    use_test = bool(config.get("use_test", False))
    key = "test_field_path" if use_test else "train_field_path"
    return parse_field_paths(config.get(key, None))


def write_labels_template(tmpl_path: Path, n_tiles: int, predicted: "np.ndarray | list[int] | None" = None) -> None:
    """
    Write a labels template CSV with header and N rows.

    Columns:
      - tile_index
      - label (pre-filled with -1)
      - predicted_class (optional; included when `predicted` is provided)
    """
    import csv
    import numpy as _np

    tmpl_path.parent.mkdir(parents=True, exist_ok=True)
    preds: _np.ndarray | None = None
    if predicted is not None:
        try:
            preds = _np.asarray(predicted, dtype=_np.int64)
        except Exception:
            preds = None
    with tmpl_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["tile_index", "label"]
        if preds is not None and preds.size >= n_tiles:
            header.append("predicted_class")
        w.writerow(header)
        for i in range(int(n_tiles)):
            row = [i, -1]
            if preds is not None and i < int(preds.shape[0]):
                row.append(int(preds[i]))
            w.writerow(row)


def save_confusion_outputs(
    cm: "np.ndarray",
    class_names: "list[str] | None",
    cm_csv_out: "Path | None",
    cm_png_out: "Path | None",
    console=None,
) -> None:
    """Save confusion matrix artifacts (CSV, WEKA-style .txt, PNG) with robust fallbacks.

    - CSV: counts with header row/col labels
    - TXT: WEKA-style labeled matrix saved alongside CSV/PNG (with same stem)
    - PNG: column-normalized heatmap using matplotlib if available, else PIL, else saves NPY
    """
    import numpy as _np
    from pathlib import Path as _Path

    def _log_ok(msg: str) -> None:
        try:
            if console is not None:
                console.success(msg)
            else:
                print(msg)
        except Exception:
            pass

    def _log_warn(msg: str) -> None:
        try:
            if console is not None:
                console.warn(msg)
            else:
                print(f"[warn] {msg}")
        except Exception:
            pass

    num_classes = int(cm.shape[0]) if cm.ndim == 2 else 0
    # CSV
    if cm_csv_out is not None:
        try:
            import csv as _csv

            cm_csv_out.parent.mkdir(parents=True, exist_ok=True)
            with cm_csv_out.open("w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["true\\pred"] + [f"{j}" for j in range(num_classes)])
                for i in range(num_classes):
                    w.writerow([f"{i}"] + [int(cm[i, j]) for j in range(num_classes)])
            _log_ok(f"Saved confusion matrix CSV to: {cm_csv_out}")
        except Exception as e:
            _log_warn(f"Failed to save cm_csv: {e}")

    # WEKA-style TXT
    try:
        weka_txt_out: _Path | None = None
        if cm_csv_out is not None:
            weka_txt_out = cm_csv_out.with_suffix(".txt")
        elif cm_png_out is not None:
            weka_txt_out = cm_png_out.with_suffix(".txt")
        if weka_txt_out is not None:
            weka_txt_out.parent.mkdir(parents=True, exist_ok=True)
            labels = class_names if (class_names and len(class_names) == num_classes) else [f"class{i}" for i in range(num_classes)]

            def code(k: int) -> str:
                s = ""
                while True:
                    s = chr(ord('a') + (k % 26)) + s
                    k //= 26
                    if k == 0:
                        break
                    k -= 1
                return s

            max_count = int(cm.max()) if cm.size > 0 else 0
            cell_w = max(3, len(str(max_count)))
            lines = [
                "=== Confusion Matrix ===",
                "  " + " ".join(code(j).rjust(cell_w) for j in range(num_classes)) + "   <-- classified as",
            ]
            for i in range(num_classes):
                row = " ".join(str(int(cm[i, j])).rjust(cell_w) for j in range(num_classes))
                lines.append(f" {row} | {code(i)} = {labels[i]}")
            with weka_txt_out.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            _log_ok(f"Saved WEKA-style CM to: {weka_txt_out}")
    except Exception as e:
        _log_warn(f"Failed to save WEKA-style CM: {e}")

    # PNG (column-normalized)
    if cm_png_out is not None:
        try:
            col_sums = cm.sum(axis=0, keepdims=True).astype(_np.float32)
            norm = _np.divide(cm.astype(_np.float32), _np.maximum(col_sums, 1.0), out=_np.zeros_like(cm, dtype=_np.float32), where=col_sums > 0)
            labels = class_names if (class_names and len(class_names) == num_classes) else [f"{i}" for i in range(num_classes)]
            try:
                import matplotlib.pyplot as _plt  # type: ignore

                _plt.figure(figsize=(max(4, num_classes), max(3, num_classes * 0.6)))
                im = _plt.imshow(norm, vmin=0.0, vmax=1.0, cmap="viridis")
                _plt.colorbar(im, fraction=0.046, pad=0.04, label="col-normalized")
                _plt.xticks(range(num_classes), labels, rotation=45, ha="right")
                _plt.yticks(range(num_classes), labels)
                _plt.xlabel("Predicted")
                _plt.ylabel("True")
                _plt.title("Confusion Matrix (column-normalized)")
                _plt.tight_layout()
                cm_png_out.parent.mkdir(parents=True, exist_ok=True)
                _plt.savefig(cm_png_out)
                _plt.close()
                _log_ok(f"Saved confusion matrix image to: {cm_png_out}")
            except Exception:
                from PIL import Image as _Image  # type: ignore

                arr = (_np.clip(norm, 0.0, 1.0) * 255.0).astype(_np.uint8)
                img = _Image.fromarray(arr, mode="L").resize((num_classes * 32, num_classes * 32), resample=_Image.NEAREST)
                cm_png_out.parent.mkdir(parents=True, exist_ok=True)
                img.save(cm_png_out)
                _log_ok(f"Saved confusion matrix image (grayscale) to: {cm_png_out}")
        except Exception as e:
            try:
                cm_png_out.parent.mkdir(parents=True, exist_ok=True)
                _np.save(str(cm_png_out.with_suffix(".npy")), cm.astype(_np.float32))
                _log_warn(f"Could not save CM image; saved counts as NPY: {cm_png_out.with_suffix('.npy')} (error: {e})")
            except Exception as e2:
                _log_warn(f"Failed to save confusion matrix image or NPY: {e2}")


def plot_training_progress(
    history: "list[dict]",
    out_png: Path,
    *,
    console=None,
) -> None:
    """Plot train/val accuracy and loss curves from a history dict list."""
    if not history:
        return
    try:
        import matplotlib.pyplot as _plt  # type: ignore
        import numpy as _np  # type: ignore
    except Exception as e:  # pragma: no cover
        try:
            if console is not None:
                console.warn(f"[training] matplotlib unavailable; skipping training progress plot: {e}")
        except Exception:
            pass
        return

    epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(history)]
    tr_loss = [float(r.get("train_loss", 0.0)) for r in history]
    va_loss = [float(r.get("val_loss", 0.0)) for r in history]
    tr_acc = [float(r.get("train_acc", 0.0)) * 100.0 for r in history]
    va_acc = [float(r.get("val_acc", 0.0)) * 100.0 for r in history]

    _plt.figure(figsize=(8, 6))
    ax1 = _plt.subplot(2, 1, 1)
    ax1.plot(epochs, tr_acc, label="Train Accuracy", color="#2C7BB6", marker="o", markersize=3)
    if va_acc:
        ax1.plot(epochs, va_acc, label="Validation Accuracy", color="#D7191C", marker="^", markersize=3)
        try:
            best_idx = int(_np.nanargmax(_np.asarray(va_acc)))  # type: ignore[arg-type]
            ax1.scatter([epochs[best_idx]], [va_acc[best_idx]], color="#D7191C", s=35, zorder=5, label="Best Val Acc")
        except Exception:
            pass
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Training Progress")
    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax1.legend(loc="lower right")

    ax2 = _plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(epochs, tr_loss, label="Train Loss", color="#2C7BB6", marker="o", markersize=3)
    if va_loss:
        ax2.plot(epochs, va_loss, label="Validation Loss", color="#D7191C", marker="^", markersize=3)
        try:
            best_loss_idx = int(_np.nanargmin(_np.asarray(va_loss)))  # type: ignore[arg-type]
            ax2.scatter([epochs[best_loss_idx]], [va_loss[best_loss_idx]], color="#D7191C", s=35, zorder=5, label="Min Val Loss")
        except Exception:
            pass
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax2.legend(loc="upper right")

    _plt.tight_layout()
    try:
        ensure_parent(out_png)
        _plt.savefig(out_png, dpi=150)
        if console is not None:
            try:
                console.success(f"[training] Saved progress plot to: {out_png}")
            except Exception:
                pass
    except Exception as e:
        try:
            if console is not None:
                console.warn(f"[training] Failed to save progress plot: {e}")
        except Exception:
            pass
    finally:
        _plt.close()
