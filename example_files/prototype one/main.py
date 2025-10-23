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
        self.device: str = self.choose_device(str(self.cfg.get("device", "auto")))
        self.set_seed(int(self.cfg.get("seed", 42)))
        print(f"Loaded config from: {self.config_path}")
        print(f"Using device: {self.device}")

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def choose_device(pref: str) -> str:
        p = pref.lower()
        if p == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if p == "cpu":
            return "cpu"
        if p == "mps":
            return (
                "mps"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        # NumPy not required here; seeding torch covers most stochasticity
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


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
