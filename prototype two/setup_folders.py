from __future__ import annotations

"""
setup_folders.py - ensure the expected prototype-two folder structure exists.

- Reads the config (default: config.json beside this script)
- Creates training/testing inputs, outputs roots, and report/training subfolders
- Exits cleanly if directories already exist
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Set


def _anchor(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p).resolve()


def _add_path(path_str: str | None, base_dir: Path, dest: Set[Path]) -> None:
    if not isinstance(path_str, str) or not path_str.strip():
        return
    p = _anchor(path_str, base_dir)
    dest.add(p.parent if p.suffix else p)


def collect_dirs(cfg: Dict, base_dir: Path) -> Set[Path]:
    dirs: Set[Path] = set()

    paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    outputs_root_str = (
        paths_cfg.get("outputs_root")
        or cfg.get("outputs_root")
        or "data/outputs"
    )
    outputs_root = _anchor(outputs_root_str, base_dir)
    dirs.add(outputs_root)

    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    _add_path(data_cfg.get("train_image_dir"), base_dir, dirs)
    _add_path(data_cfg.get("val_image_dir"), base_dir, dirs)
    _add_path(data_cfg.get("test_image_dir"), base_dir, dirs)
    _add_path(data_cfg.get("train_vector_dir"), base_dir, dirs)
    _add_path(data_cfg.get("test_vector_dir"), base_dir, dirs)

    # Training / inference fields (vector mode)
    _add_path(cfg.get("train_field_path"), base_dir, dirs)
    _add_path(cfg.get("test_field_path"), base_dir, dirs)

    inf = cfg.get("inference", {}) if isinstance(cfg.get("inference"), dict) else {}
    inf_common = inf.get("common", {}) if isinstance(inf.get("common"), dict) else {}
    for key in ("file_labels", "file_summary", "cm_csv", "cm_png"):
        _add_path(inf_common.get(key), base_dir, dirs)

    reports_cfg = cfg.get("reports", {}) if isinstance(cfg.get("reports"), dict) else {}
    for key in ("predictions_csv", "image_root", "cm_csv", "cm_png"):
        _add_path(reports_cfg.get(key), base_dir, dirs)

    # Training progress artifacts (matches Trainer defaults)
    data_kind = (
        str(data_cfg.get("data_kind", "")).lower()
        or str(inf_common.get("data_kind", "")).lower()
        or "image"
    )
    subdir = "image" if data_kind.startswith("image") else "vector"
    training_dir = outputs_root / subdir / "training"
    dirs.add(training_dir)

    # Outputs subfolders commonly used
    dirs.add(outputs_root / "image")
    dirs.add(outputs_root / "image" / "plots")

    return dirs


def ensure_dirs(dirs: Iterable[Path]) -> None:
    for d in sorted({p.resolve() for p in dirs}):
        d.mkdir(parents=True, exist_ok=True)
        print(f"[ok] {d}")


def ensure_dirs_from_config(cfg_path: Path, cfg: Dict | None = None) -> None:
    """
    Load config (if not provided), collect expected directories, and create them.

    This can be called from any script before writing outputs to guarantee the
    folder structure exists in one place.
    """
    cfg_path = cfg_path.resolve()
    if cfg is None:
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    base_dir = cfg_path.parent
    dirs = collect_dirs(cfg, base_dir)
    print(f"[info] base_dir={base_dir}")
    print(f"[info] ensuring {len(dirs)} directories ...")
    ensure_dirs(dirs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create expected folder structure for prototype two.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.json")),
        help="Path to config JSON (defaults to config.json beside this script)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    ensure_dirs_from_config(cfg_path, cfg)
    print("[done] folder structure ready.")


if __name__ == "__main__":
    main()
