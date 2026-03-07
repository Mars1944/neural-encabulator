import argparse
import json
from pathlib import Path

import torch

from common import ConfigManager, DeviceSelector
from console import console_from_config
from main import ModelManager
from trainer import Trainer
from cnn_model import count_parameters
from common import PathResolver as _PR, ensure_outputs_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype two trainer")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.json")),
        help="Path to JSON config file (consolidated or training-only)",
    )
    args = parser.parse_args()

    cfg_mgr = ConfigManager(Path(args.config))
    # Stash config file location for relative label resolution
    cfg_mgr.config["_config_path"] = str(cfg_mgr.path)
    cfg_mgr.config["_config_dir"] = str(cfg_mgr.path.parent)

    # Pretty-print settings tables
    c = console_from_config(cfg_mgr.config)
    model_cfg = cfg_mgr.config.get("model", {}) if isinstance(cfg_mgr.config.get("model"), dict) else {}
    train_cfg = cfg_mgr.config.get("training", {}) if isinstance(cfg_mgr.config.get("training"), dict) else {}
    data_cfg = cfg_mgr.config.get("data", {}) if isinstance(cfg_mgr.config.get("data"), dict) else {}

    # Ensure base folder structure exists (centralized helper)
    ensure_outputs_ready(cfg_mgr.path, cfg_mgr.config)

    # Report the resolved device (GPU/CPU name)
    try:
        c.info(f"Using device: {DeviceSelector.format(cfg_mgr.device)}")
    except Exception:
        pass

    def _val_str(v: object) -> str:
        try:
            if isinstance(v, (dict, list, tuple)):
                return json.dumps(v, separators=(",", ":"))
            return str(v)
        except Exception:
            return str(v)

    # Data/Image settings
    if data_cfg:
        rows = [[str(k), _val_str(v)] for k, v in sorted(data_cfg.items(), key=lambda kv: str(kv[0]))]
        c.table(["Data Key", "Value"], rows, align=["l", "l"], title="Data Settings")
        # Focused image-related subset for quick scan
        image_keys = [
            "train_data_kind", "test_data_kind", "train_image_dir", "val_image_dir", "test_image_dir",
            "image_size", "image_channels", "image_mean", "image_std", "augment",
        ]
        img_rows = []
        for k in image_keys:
            if k in data_cfg:
                img_rows.append([k, _val_str(data_cfg[k])])
        if img_rows:
            c.table(["Image Key", "Value"], img_rows, align=["l", "l"], title="Image Settings")

    # Model/Training settings
    if model_cfg:
        rows = [[str(k), _val_str(v)] for k, v in sorted(model_cfg.items(), key=lambda kv: str(kv[0]))]
        c.table(["Model Key", "Value"], rows, align=["l", "l"], title="Model Settings")
    if train_cfg:
        rows = [[str(k), _val_str(v)] for k, v in sorted(train_cfg.items(), key=lambda kv: str(kv[0]))]
        c.table(["Training Key", "Value"], rows, align=["l", "l"], title="Training Settings")

    # Ensure training infers channels from train data, not test directory
    try:
        cfg_mgr.config["use_test"] = False
    except Exception:
        pass
    manager = ModelManager(cfg_mgr.config, cfg_mgr.device)
    trainer = Trainer(cfg_mgr.config, cfg_mgr.device, manager.model)

    # Precompute and lock adaptive tiling for vector mode so we can display it in Derived Settings
    try:
        _ = trainer.compute_and_lock_tiling(lock=True)
    except Exception:
        pass

    # Derived values table
    try:
        # Effective kinds
        train_kind = _PR.resolve_split_kind(cfg_mgr.config, "train") if hasattr(_PR, "resolve_split_kind") else str(cfg_mgr.config.get("data_kind", "vector")).lower()
        test_kind = _PR.resolve_split_kind(cfg_mgr.config, "test") if hasattr(_PR, "resolve_split_kind") else str(cfg_mgr.config.get("data_kind", "vector")).lower()
        # Input channels: read from first Conv2d
        in_ch = None
        for m in manager.model.modules():
            if isinstance(m, torch.nn.Conv2d):
                in_ch = int(m.in_channels)
                break
        num_params = count_parameters(manager.model)
        num_classes = getattr(manager.model, "num_classes", None)
        # Outputs root
        resolver = _PR(cfg_mgr.path, outputs_root=str(cfg_mgr.config.get("outputs_root")) if isinstance(cfg_mgr.config.get("outputs_root"), str) else None)
        out_root = str(resolver.outputs_root)
        # Optimizer/Scheduler names
        opt_name = trainer.optimizer.__class__.__name__ if hasattr(trainer, "optimizer") else "?"
        sch_name = trainer.scheduler.__class__.__name__ if getattr(trainer, "scheduler", None) is not None else "none"
        rows = [
            ["device", str(cfg_mgr.device)],
            ["train_data_kind", str(train_kind)],
            ["test_data_kind", str(test_kind)],
            ["input_channels", str(in_ch) if in_ch is not None else "unknown"],
            ["num_classes", str(num_classes) if num_classes is not None else "unknown"],
            ["parameters", f"{num_params:,}"],
            ["optimizer", opt_name],
            ["scheduler", sch_name],
            ["outputs_root", out_root],
            ["tile_size", str(cfg_mgr.config.get("tile_size", [256, 256]))],
            ["tile_stride", str(cfg_mgr.config.get("tile_stride", cfg_mgr.config.get("tile_size", [256, 256])))]
        ]
        # Optional tiling diagnostics
        try:
            min_hw = cfg_mgr.config.get("_tiling_min_source_hw", None)
            cnn_min = cfg_mgr.config.get("_tiling_cnn_min_side", None)
            if isinstance(min_hw, (list, tuple)) and len(min_hw) == 2:
                rows.append(["min_source_hw", f"{int(min_hw[0])}x{int(min_hw[1])}"])
            if cnn_min is not None:
                rows.append(["cnn_min_side", str(int(cnn_min))])
        except Exception:
            pass
        c.table(["Derived", "Value"], rows, align=["l", "l"], title="Derived Settings")
    except Exception:
        pass
    best_path = trainer.fit()
    c = console_from_config(cfg_mgr.config)
    c.success(f"Training complete. Best/Final checkpoint: {best_path}")


if __name__ == "__main__":
    main()
