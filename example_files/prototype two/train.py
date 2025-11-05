import argparse
from pathlib import Path

import torch

from common import ConfigManager
from console import console_from_config
from main import ModelManager
from trainer import Trainer


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

    manager = ModelManager(cfg_mgr.config, cfg_mgr.device)
    trainer = Trainer(cfg_mgr.config, cfg_mgr.device, manager.model)
    best_path = trainer.fit()
    c = console_from_config(cfg_mgr.config)
    c.success(f"Training complete. Best/Final checkpoint: {best_path}")


if __name__ == "__main__":
    main()
