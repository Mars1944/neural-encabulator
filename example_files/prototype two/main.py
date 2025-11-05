import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

# Centralized optional imports
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

from cnn_model import SimpleCNN, count_parameters
from console import console_from_config
from cnn_optim import CnnOptim
from common import DeviceSelector, ConfigManager, PathResolver, infer_input_channels_from_field
from vector_field_data import (
    ensure_chw,
    load_vector_field,
    load_vector_field_tiles,
)


class ModelManager:
    """Builds the model and optimizer and provides input loading helpers."""

    def __init__(self, config: Dict[str, Any], device: "torch.device") -> None:
        self.config = config
        self.device = device
        self.model = self._build_model()

    def _select_path(self, value: Any) -> str:
        """Pick a single file path from config entry that may be str or list[str]."""
        if isinstance(value, (list, tuple)):
            for v in value:
                if isinstance(v, str) and v.strip():
                    return v
            return ""
        return str(value or "")

    def _build_model(self) -> "torch.nn.Module":
        # Infer channels from vector-field file if provided
        field_path = self._select_path(self.config.get("field_path", ""))
        if not field_path.strip():
            use_test = bool(self.config.get("use_test", False))
            field_path = self._select_path(
                self.config.get("test_field_path" if use_test else "train_field_path", "")
            )
        if isinstance(field_path, str) and field_path.strip():
            src = (
                "config.field_path"
                if str(self.config.get("field_path", "") or "").strip()
                else ("config.test_field_path" if bool(self.config.get("use_test", False)) else "config.train_field_path")
            )
            c = console_from_config(self.config)
            c.info(f"Inferring input_channels from: {field_path} [source={src}]")
            try:
                inferred = infer_input_channels_from_field(self.config, field_path)
                self.config["input_channels"] = int(inferred)
            except Exception as e:
                c.warn(f"Could not infer input_channels from field_path: {e}")

        model = SimpleCNN(self.config).to(self.device)
        c = console_from_config(self.config)
        c.info(str(model))
        c = console_from_config(self.config)
        c.info(f"Trainable parameters: {count_parameters(model):,}")
        return model

    def make_dummy_input(self) -> "torch.Tensor":
        b = max(int(self.config.get("dummy_batch", 4)), 1)
        c = int(self.config.get("input_channels", 3))
        h, w = tuple(self.config.get("tile_size", [256, 256]))
        return torch.randn(b, c, int(h), int(w), device=self.device)

    def load_inputs(self, field_path: Optional[str] = None) -> "torch.Tensor":
        # CLI arg overrides config
        use_field: Optional[str] = None
        src = ""
        if isinstance(field_path, str) and field_path.strip():
            use_field = field_path
            src = "arg.field_path"
        else:
            config_field_path = self._select_path(self.config.get("field_path", ""))
            if config_field_path.strip():
                use_field = config_field_path
                src = "config.field_path"
            else:
                use_test = bool(self.config.get("use_test", False))
                key = "test_field_path" if use_test else "train_field_path"
                use_field = self._select_path(self.config.get(key, "") or "")
                src = f"config.{key}"
        if isinstance(use_field, str) and use_field.strip():
            c = console_from_config(self.config)
            c.info(f"Using vector field: {use_field} [source={src}]")
            tile_size = tuple(self.config.get("tile_size", [256, 256]))
            stride_config = self.config.get("tile_stride", None)
            stride = None
            if isinstance(stride_config, (list, tuple)):
                stride = (int(stride_config[0]), int(stride_config[1]))
            add_mag = bool(self.config.get("add_magnitude", True))
            normalize = bool(self.config.get("normalize", True))
            limit_tiles = self.config.get("limit_tiles", None)
            tile_batch = load_vector_field_tiles(
                path=use_field,
                tile_size=(int(tile_size[0]), int(tile_size[1])),
                stride=stride,
                add_magnitude=add_mag,
                normalize=normalize,
                limit_tiles=None if limit_tiles is None else int(limit_tiles),
            )
            c = console_from_config(self.config)
            c.info(f"Loaded tiles: shape={tile_batch.shape}")
            return torch.from_numpy(tile_batch).to(self.device)
        return self.make_dummy_input()

    def create_optimizer(self) -> "torch.optim.Optimizer":
        builder = CnnOptim(self.config)
        optimizer, scheduler, scheduler_step_on = builder.build(self.model)
        setattr(self, "optim_builder", builder)
        setattr(self, "scheduler", scheduler)
        setattr(self, "scheduler_step_on", scheduler_step_on)
        return optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural Encabulator Runner (Vector-Field CNN) [prototype two]")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.json")),
        help="Path to JSON config file (consolidated or model-only)",
    )
    parser.add_argument(
        "--field_path",
        type=str,
        default=None,
        help="Override vector field path (.npy/.npz/.csv) from config",
    )
    parser.add_argument(
        "--use_test",
        action="store_true",
        help="Use test_field_path instead of train_field_path (ignored if --field_path is set)",
    )
    parser.add_argument(
        "--save_weights",
        type=str,
        default=None,
        help="Path to save model checkpoint (.pth/.pt). Includes config and optional optimizer.",
    )
    parser.add_argument(
        "--include_optim",
        action="store_true",
        help="Include optimizer state_dict in the saved checkpoint",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Skip saving checkpoint (overrides save_weights/default)",
    )
    args = parser.parse_args()

    config_manager = ConfigManager(Path(args.config))
    if args.field_path:
        config_manager.config["field_path"] = args.field_path
    if args.use_test:
        config_manager.config["use_test"] = True
    manager = ModelManager(config_manager.config, config_manager.device)

    try:
        inputs = manager.load_inputs(field_path=args.field_path)
        logits = manager.model(inputs)
        c = console_from_config(config_manager.config)
        c.info(f"Forward pass OK. Logits shape: {tuple(logits.shape)}")
        optimizer = manager.create_optimizer()
    except Exception as e:
        raise RuntimeError(f"Error during forward/optimizer setup: {e}") from e

    # Optional: save model weights/checkpoint
    if bool(args.no_save) or bool(config_manager.config.get("no_save", False)):
        c = console_from_config(config_manager.config)
        c.info("Skipping save: no_save is set")
    else:
        save_path = args.save_weights if args.save_weights else config_manager.config.get("save_weights", None)
        if not (isinstance(save_path, str) and save_path.strip()):
            # Default under outputs_root/checkpoints using PathResolver
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            oroot = config_manager.config.get("outputs_root", None)
            resolver = PathResolver(config_manager.path, outputs_root=oroot if isinstance(oroot, str) else None)
            default_sp = (resolver.outputs_root / "checkpoints" / f"model-{ts}.pth").resolve()
            c = console_from_config(config_manager.config)
            c.info(f"No save_weights provided; defaulting to: {default_sp}")
            save_path = str(default_sp)
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": manager.model.state_dict(),
            "config": config_manager.config,
        }
        if bool(args.include_optim):
            try:
                checkpoint["optimizer_state"] = optimizer.state_dict()  # type: ignore[name-defined]
            except Exception:
                pass
        torch.save(checkpoint, str(sp))
        c = console_from_config(config_manager.config)
        c.success(f"Saved checkpoint to: {sp}")


if __name__ == "__main__":
    main()
