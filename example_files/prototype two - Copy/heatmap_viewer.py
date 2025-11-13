import argparse
import json
from pathlib import Path
from typing import Tuple, Optional, Union, Any, Dict

import numpy as np
from common import ConfigManager, PathResolver
from console import console_from_config, get_console
from vector_field_data import load_vector_field, ensure_chw


def show_stitched_heatmap(
    values: np.ndarray,
    grid_shape: Tuple[int, int],
    title: Optional[str] = None,
    cmap: str = "viridis",
    *,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    c = get_console()
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        c.warn(f"matplotlib not available for showing heatmap: {e}")
        return

    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    if values.ndim != 1 or rows * cols != int(values.shape[0]):
        c.warn(f"Cannot reshape values of length {values.shape[0]} into grid {rows}x{cols}")
        return
    grid = values.reshape(rows, cols)
    plt.figure(figsize=(max(4, cols / 4), max(4, rows / 4)))
    im = plt.imshow(grid, vmin=0.0, vmax=1.0, cmap=cmap)
    plt.colorbar(im, label="probability")
    if title:
        plt.title(title)
    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(sp)
        c.success(f"Saved stitched heatmap to: {sp}")

    if show:
        plt.show()
    else:
        plt.close()


def _load_values_from_csv(path: Path) -> np.ndarray:
    import csv

    # Robust decoding: try common encodings, then lenient fallback
    rows = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with path.open("r", encoding=enc, errors="strict") as f:
                reader = csv.reader(f)
                rows = list(reader)
            break
        except UnicodeDecodeError:
            rows = None
            continue
        except Exception:
            rows = None
            continue
    if rows is None:
        # Last resort: ignore undecodable bytes
        with path.open("r", encoding="latin-1", errors="ignore") as f:
            reader = csv.reader(f)
            rows = list(reader)
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    # Detect header by attempting to parse the second column as float
    start = 0
    try:
        _ = float(rows[0][1])
    except Exception:
        start = 1
    vals = []
    for r in rows[start:]:
        if len(r) < 2:
            continue
        try:
            vals.append(float(r[1]))
        except Exception:
            continue
    return np.asarray(vals, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitched heatmap viewer")
    # Values can come from CLI or config; do not require on CLI
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--values_npy", type=str, help="Path to .npy 1D array of per-tile values")
    src.add_argument("--values_csv", type=str, help="Path to CSV with tile_index,value")

    # Optional: derive rows/cols from field + tiling
    parser.add_argument("--rows", type=int, default=None, help="Number of tile rows in the grid")
    parser.add_argument("--cols", type=int, default=None, help="Number of tile columns in the grid")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON; can provide values/tiling/field")
    parser.add_argument("--field_path", type=str, default=None, help="Override vector field path (.npy/.npz/.csv)")
    parser.add_argument("--tile_size", type=int, nargs=2, metavar=("TH", "TW"), default=None, help="Tile size [th tw]")
    parser.add_argument(
        "--tile_stride",
        type=int,
        nargs=2,
        metavar=("SH", "SW"),
        default=None,
        help="Tile stride [sh sw]; defaults to tile_size when omitted",
    )

    parser.add_argument("--title", type=str, default=None, help="Optional window title")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name")
    parser.add_argument("--save_stitched", type=str, default=None, help="Optional path to save stitched image (relative to this script if not absolute)")
    parser.add_argument("--no_show", action="store_true", help="Do not open a window; only save if --save_stitched is given")
    args = parser.parse_args()

    # Load config if provided (for defaults and path anchoring)
    cfg_dict: Dict[str, Any] | None = None
    cfg_path: Optional[Path] = None
    if args.config:
        cfg_path = Path(args.config)
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
        except Exception as e:
            c = get_console()
            c.warn(f"Could not parse config '{cfg_path}': {e}")
            cfg_dict = None
    else:
        # Auto-discover default config alongside the script
        auto_cfg = Path(__file__).with_name("heatmap_viewer_config.json")
        if auto_cfg.exists():
            cfg_path = auto_cfg
            try:
                with cfg_path.open("r", encoding="utf-8") as f:
                    cfg_dict = json.load(f)
                c = console_from_config(cfg_dict)
                c.info(f"Using viewer config: {cfg_path}")
            except Exception as e:
                c = get_console()
                c.warn(f"Could not parse default config '{cfg_path}': {e}")
                cfg_dict = None

    # Use PathResolver for consistent anchoring
    # Honor optional outputs_root from config to anchor saves consistently
    outputs_root = None
    if isinstance(cfg_dict, dict):
        oroot = cfg_dict.get("outputs_root", None)
        if isinstance(oroot, str) and oroot.strip():
            outputs_root = oroot
    resolver = PathResolver(cfg_path if cfg_path is not None else Path(__file__), outputs_root=outputs_root)
    def _anchor(p: Optional[Union[str, Path]]) -> Optional[Path]:
        return resolver.anchor(p) if p is not None else None

    # Load values (CLI has priority, else config viewer_values_npy/csv or values_npy/csv)
    values: np.ndarray
    viewer_section = cfg_dict.get("viewer") if isinstance(cfg_dict, dict) else None
    values_npy = args.values_npy or (
        viewer_section.get("viewer_values_npy") if isinstance(viewer_section, dict) else None
    ) or (cfg_dict.get("viewer_values_npy") if isinstance(cfg_dict, dict) else None) or (
        cfg_dict.get("values_npy") if isinstance(cfg_dict, dict) else None
    )
    values_csv = args.values_csv or (
        viewer_section.get("viewer_values_csv") if isinstance(viewer_section, dict) else None
    ) or (cfg_dict.get("viewer_values_csv") if isinstance(cfg_dict, dict) else None) or (
        cfg_dict.get("values_csv") if isinstance(cfg_dict, dict) else None
    )
    if values_npy:
        resolved = _anchor(values_npy)
        c = console_from_config(cfg_dict if isinstance(cfg_dict, dict) else {})
        c.info(f"Loading values from NPY: {resolved}")
        values = np.load(str(resolved))
    elif values_csv:
        resolved = _anchor(values_csv)
        c = console_from_config(cfg_dict if isinstance(cfg_dict, dict) else {})
        c.info(f"Loading values from CSV: {resolved}")
        values = _load_values_from_csv(resolved)
    elif args.values_npy:
        values = np.load(args.values_npy)
    elif args.values_csv:
        values = _load_values_from_csv(Path(args.values_csv))
    else:
        raise SystemExit(
            "No values provided. Pass --values_npy/--values_csv or provide viewer_values_npy/viewer_values_csv in the config."
        )

    # Basic sanity check on values
    if values.size == 0:
        raise SystemExit("Loaded 0 values. Check your CSV/NPY path and delimiter/encoding.")

    # Determine grid shape
    rows, cols = args.rows, args.cols
    if (rows is None or cols is None) and isinstance(cfg_dict, dict):
        # Optional rows/cols in config for viewer (prefer viewer section)
        r_cfg = (
            (viewer_section.get("viewer_rows") if isinstance(viewer_section, dict) else None)
            or cfg_dict.get("viewer_rows")
            or cfg_dict.get("rows")
        )
        c_cfg = (
            (viewer_section.get("viewer_cols") if isinstance(viewer_section, dict) else None)
            or cfg_dict.get("viewer_cols")
            or cfg_dict.get("cols")
        )
        if isinstance(r_cfg, int) and isinstance(c_cfg, int):
            rows, cols = r_cfg, c_cfg
    if rows is None or cols is None:
        # Try to derive from field + tiling
        config_path = Path(args.config) if args.config else None
        config_manager: Optional[ConfigManager] = None
        if config_path is not None:
            try:
                config_manager = ConfigManager(config_path)
            except Exception as e:
                c = get_console()
                c.warn(f"Could not load config '{config_path}': {e}")

        # Resolve field path
        field_path: Optional[str] = args.field_path
        if field_path is None and config_manager is not None:
            cfg = config_manager.config
            # Prefer explicit field_path; fallback to train_field_path
            field_path = (
                str(cfg.get("field_path", ""))
                or str(cfg.get("train_field_path", ""))
                or str(cfg.get("test_field_path", ""))
            )
        if not field_path:
            raise SystemExit("--rows/--cols not provided and no field_path derivable from --config; specify grid or tiling inputs")

        # Resolve tiling parameters
        if args.tile_size is not None:
            th, tw = int(args.tile_size[0]), int(args.tile_size[1])
        elif config_manager is not None:
            t = config_manager.config.get("tile_size", [256, 256])
            th, tw = int(t[0]), int(t[1])
        else:
            th, tw = 256, 256

        if args.tile_stride is not None:
            sh, sw = int(args.tile_stride[0]), int(args.tile_stride[1])
        elif config_manager is not None and config_manager.config.get("tile_stride") is not None:
            s = config_manager.config.get("tile_stride", [th, tw])
            sh, sw = int(s[0]), int(s[1])
        else:
            sh, sw = th, tw

        # Compute grid from field shape
        field_arr = load_vector_field(field_path)
        chw = ensure_chw(field_arr)
        H, W = int(chw.shape[1]), int(chw.shape[2])
        rows = (H - th) // sh + 1 if H >= th else 0
        cols = (W - tw) // sw + 1 if W >= tw else 0
        if rows * cols != int(values.shape[0]) or rows <= 0 or cols <= 0:
            raise SystemExit(
                f"Derived grid {rows}x{cols} doesn't match values length {values.shape[0]}; provide --rows --cols explicitly"
            )

    # Resolve save path; anchor relative to config (preferred) or script directory
    save_path: Optional[Path] = None
    if args.save_stitched:
        sp = Path(args.save_stitched)
        if not sp.is_absolute():
            base = cfg_path.parent if cfg_path is not None else Path(__file__).parent
            sp = (base / sp).resolve()
        save_path = sp

    # Allow config overrides for title/cmap/no_show/save if not provided via CLI
    title = args.title if args.title is not None else (
        (viewer_section.get("viewer_title") if isinstance(viewer_section, dict) else None)
        or (cfg_dict.get("viewer_title") if isinstance(cfg_dict, dict) else None)
    )
    cmap = args.cmap if args.cmap is not None else (
        (viewer_section.get("viewer_cmap") if isinstance(viewer_section, dict) else None)
        or (cfg_dict.get("viewer_cmap") if isinstance(cfg_dict, dict) else "viridis")
    )
    no_show = bool(
        args.no_show
        or (
            (viewer_section.get("viewer_no_show", False) if isinstance(viewer_section, dict) else False)
            or (cfg_dict.get("viewer_no_show", False) if isinstance(cfg_dict, dict) else False)
        )
    )
    if save_path is None and isinstance(cfg_dict, dict):
        v_save = (
            (viewer_section.get("viewer_save") if isinstance(viewer_section, dict) else None)
            or cfg_dict.get("viewer_save")
        )
        if v_save:
            save_path = _anchor(v_save)

    show_stitched_heatmap(
        values,
        (int(rows), int(cols)),
        title=title,
        cmap=cmap,
        show=not no_show,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
