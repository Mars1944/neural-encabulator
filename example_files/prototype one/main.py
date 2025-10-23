import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

# Single imports per library
try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyTorch is not installed. Please install torch to proceed.") from e

from cnn_model import SimpleCNN, count_parameters
from vector_field_data import (
    ensure_chw,
    load_vector_field,
    load_vector_field_tiles,
)


class AppSetup:
    def __init__(self, config_path: Path) -> None:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        self.config_path = config_path
        self.cfg: Dict[str, Any] = self._load_config(self.config_path)
        self._validate_and_normalize_cfg()
        self.device: torch.device = self.choose_device(str(self.cfg.get("device", "auto")))
        self.set_seed(int(self.cfg.get("seed", 42)))
        self._apply_runtime_flags()
        print(f"Loaded config from: {self.config_path}")
        print(f"Using device: {self.format_device_info(self.device)}")

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def choose_device(pref: str | None) -> torch.device:
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
            print(f"[warn] Requested CUDA device '{pref}' not available; falling back to CPU")
            return torch.device("cpu")

        if p == "mps":
            if mps_available():
                return torch.device("mps")
            print("[warn] Requested MPS device but it's not available; using CPU")
            return torch.device("cpu")

        if p == "cpu":
            return torch.device("cpu")

        # auto preference order
        if cuda_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def format_device_info(device: torch.device) -> str:
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

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        # Also seed NumPy if available (useful for any pre-tensor randomness)
        try:
            import numpy as np  # type: ignore

            np.random.seed(seed)
        except Exception:
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _apply_runtime_flags(self) -> None:
        # Determinism and precision controls (optional)
        deterministic = bool(self.cfg.get("deterministic", False))
        try:
            torch.use_deterministic_algorithms(deterministic)
        except Exception:
            pass
        try:
            import torch.backends.cudnn as cudnn  # type: ignore

            cudnn.deterministic = deterministic
            cudnn.benchmark = not deterministic
        except Exception:
            pass
        # Float32 matmul precision (PyTorch 2.x): 'high' | 'medium' | 'default'
        prec = str(self.cfg.get("matmul_precision", "default")).lower()
        try:
            if prec in {"high", "medium", "default"}:
                torch.set_float32_matmul_precision(prec)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _validate_and_normalize_cfg(self) -> None:
        # Minimal schema checks + defaults + warnings for unknown keys
        cfg = self.cfg
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
        }
        # Warn on unknown keys
        for k in list(cfg.keys()):
            if k not in expected:
                print(f"[warn] Unknown config key: '{k}'")
        # Fill defaults commonly needed
        cfg.setdefault("model_type", "cnn")
        cfg.setdefault("num_classes", 2)
        cfg.setdefault("conv_channels", [32, 64, 128])
        cfg.setdefault("kernel_size", 3)
        cfg.setdefault("pool_every", 1)
        cfg.setdefault("dropout", 0.0)
        cfg.setdefault("use_batchnorm", True)
        cfg.setdefault("learning_rate", 1e-3)
        cfg.setdefault("weight_decay", 0.0)
        cfg.setdefault("tile_size", [256, 256])
        # Basic type/shape validations (non-fatal -> raise with context)
        try:
            ts = cfg.get("tile_size", [256, 256])
            if not (isinstance(ts, (list, tuple)) and len(ts) == 2):
                raise ValueError("tile_size must be [h, w]")
            if any(int(x) <= 0 for x in ts):
                raise ValueError("tile_size entries must be positive")
            stride = cfg.get("tile_stride", None)
            if stride is not None:
                if not (isinstance(stride, (list, tuple)) and len(stride) == 2):
                    raise ValueError("tile_stride must be [sh, sw] when provided")
                if any(int(x) <= 0 for x in stride):
                    raise ValueError("tile_stride entries must be positive")
        except Exception as e:
            raise ValueError(f"Config validation error: {e}")


class ModelManager:
    def __init__(self, cfg: Dict[str, Any], device: str) -> None:
        self.cfg = cfg
        self.device = device
        self.model = self._build_model()

    def _build_model(self) -> "torch.nn.Module":
        # Infer channels from vector-field file if provided
        field_path = self.cfg.get("field_path", "")
        if isinstance(field_path, str) and field_path.strip():
            try:
                arr = load_vector_field(field_path, mmap=True)
                chw = ensure_chw(arr)
                c = int(chw.shape[0])
                if bool(self.cfg.get("add_magnitude", True)) and c >= 2:
                    c = c + 1
                self.cfg["input_channels"] = c
            except Exception as e:
                print(f"[warn] Could not infer input_channels from field_path: {e}")

        model = SimpleCNN(self.cfg).to(self.device)
        print(model)
        print(f"Trainable parameters: {count_parameters(model):,}")
        return model

    def make_dummy_input(self) -> "torch.Tensor":
        img_h, img_w = self.cfg.get("image_size", [256, 256])
        in_ch = int(self.cfg.get("input_channels", 3))
        return torch.randn(4, in_ch, img_h, img_w, device=self.device)

    def load_inputs(self, field_path: str | None = None) -> "torch.Tensor":
        # Vector-field mode via large .npy/.npz with tiling (CLI overrides config)
        use_field = field_path if field_path is not None else self.cfg.get("field_path", "")
        if isinstance(use_field, str) and use_field.strip():
            tile_size = tuple(self.cfg.get("tile_size", [256, 256]))
            stride_cfg = self.cfg.get("tile_stride", None)
            stride = None
            if isinstance(stride_cfg, (list, tuple)):
                stride = (int(stride_cfg[0]), int(stride_cfg[1]))
            add_mag = bool(self.cfg.get("add_magnitude", True))
            normalize = bool(self.cfg.get("normalize", True))
            limit_tiles = self.cfg.get("limit_tiles", None)
            np_batch = load_vector_field_tiles(
                path=use_field,
                tile_size=(int(tile_size[0]), int(tile_size[1])),
                stride=stride,
                add_magnitude=add_mag,
                normalize=normalize,
                limit_tiles=None if limit_tiles is None else int(limit_tiles),
            )
            print(f"Loaded tiles: shape={np_batch.shape}")
            return torch.from_numpy(np_batch).to(self.device)
        return self.make_dummy_input()

    def create_optimizer(self) -> "torch.optim.Optimizer":
        lr = float(self.cfg.get("learning_rate", 1e-3))
        wd = float(self.cfg.get("weight_decay", 0.0))
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        print(f"Optimizer ready (AdamW, lr={lr}, weight_decay={wd}).")
        return opt


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural Encabulator Runner (Vector-Field CNN)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("cnn_vector_config.json")),
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--field_path",
        type=str,
        default=None,
        help="Override vector field path (.npy/.npz) from config",
    )
    args = parser.parse_args()

    setup = AppSetup(Path(args.config))
    if args.field_path:
        setup.cfg["field_path"] = args.field_path
    manager = ModelManager(setup.cfg, setup.device)

    try:
        inputs = manager.load_inputs(field_path=args.field_path)
        logits = manager.model(inputs)
        print(f"Forward pass OK. Logits shape: {tuple(logits.shape)}")
        _ = manager.create_optimizer()
    except Exception as e:
        raise RuntimeError(f"Error during forward/optimizer setup: {e}") from e


if __name__ == "__main__":
    main()
