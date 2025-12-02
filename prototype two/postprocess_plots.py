import argparse
import io
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from common import (
    ConfigManager,
    PathResolver,
    HeatmapGenerator,
    load_labels_from_csv,
    save_confusion_outputs,
)
from heatmap_viewer import show_stitched_heatmap
from vector_field_data import load_vector_field, ensure_chw
from console import console_from_config


class ConsoleArgumentParser(argparse.ArgumentParser):
    """Argument parser that reports errors through the console formatter."""

    def __init__(self, console, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._console = console

    def print_help(self, file=None) -> None:  # type: ignore[override]
        buf = io.StringIO()
        super().print_help(file=buf)
        try:
            for line in buf.getvalue().splitlines():
                self._console.line(line)
        except Exception:
            print(buf.getvalue())

    def error(self, message: str) -> None:  # type: ignore[override]
        try:
            self._console.error(message)
        except Exception:
            pass
        self.print_help()
        raise SystemExit(2)


def _derive_tile_params(cfg: Dict[str, Any]) -> Tuple[Tuple[int, int], Tuple[int, int] | None]:
    inf_ts = cfg.get("inference_tile_size", None)
    inf_tr = cfg.get("inference_tile_stride", None)
    tile_size = tuple(inf_ts if isinstance(inf_ts, (list, tuple)) else cfg.get("tile_size", [256, 256]))
    stride_cfg = inf_tr if isinstance(inf_tr, (list, tuple)) else cfg.get("tile_stride", None)
    stride: Tuple[int, int] | None = None
    if isinstance(stride_cfg, (list, tuple)):
        stride = (int(stride_cfg[0]), int(stride_cfg[1]))
    return (int(tile_size[0]), int(tile_size[1])), stride


def _load_probs(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    if arr.ndim != 2:
        raise RuntimeError(f"Expected (N,C) probabilities; got shape {arr.shape}")
    return arr.astype(np.float32)


def _compute_cm(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t < 0 or p < 0 or t >= num_classes or p >= num_classes:
            continue
        cm[t, p] += 1
    return cm


def main() -> None:
    bootstrap_console = console_from_config({})
    parser = ConsoleArgumentParser(bootstrap_console, description="Post-process inference outputs into plots/images")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("config.json")), help="Config file for defaults")
    parser.add_argument("--probs_npy", type=str, required=True, help="Path to per-tile probabilities saved by infer.py (.npy, shape N x C)")
    parser.add_argument("--field_path", type=str, default=None, help="Vector field file used during inference (required for stitched heatmap)")
    parser.add_argument("--heatmap_class", type=int, default=None, help="Class index to visualize; defaults to predicted confidence")
    parser.add_argument("--heatmaps_dir", type=str, default=None, help="Directory to save per-tile heatmap images")
    parser.add_argument("--stitched_heatmap", type=str, default=None, help="Path to save stitched heatmap image")
    parser.add_argument("--show_heatmap", action="store_true", help="Show stitched heatmap window after saving (requires matplotlib)")
    parser.add_argument("--labels_csv", type=str, default=None, help="CSV with tile_index,label for confusion matrix")
    parser.add_argument("--cm_csv", type=str, default=None, help="Output CSV for confusion matrix counts")
    parser.add_argument("--cm_png", type=str, default=None, help="Output PNG for confusion matrix heatmap")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg_mgr = ConfigManager(cfg_path)
    cfg = cfg_mgr.config
    console = console_from_config(cfg)
    console.header("Postprocess")
    console.info(f"Using config: {cfg_path}")
    resolver = PathResolver(cfg_path, outputs_root=str(cfg.get("outputs_root")) if isinstance(cfg.get("outputs_root"), str) else None)

    probs_path = resolver.anchor(args.probs_npy)
    field_path = resolver.anchor(args.field_path) if args.field_path else None
    heatmaps_dir = resolver.anchor(args.heatmaps_dir) if args.heatmaps_dir else None
    stitched_heatmap = resolver.anchor(args.stitched_heatmap) if args.stitched_heatmap else None
    labels_csv = resolver.anchor(args.labels_csv) if args.labels_csv else None
    cm_csv = resolver.anchor(args.cm_csv) if args.cm_csv else None
    cm_png = resolver.anchor(args.cm_png) if args.cm_png else None

    tile_size, stride = _derive_tile_params(cfg)
    console.info(f"Loading probabilities from: {probs_path}")
    probs = _load_probs(probs_path)
    num_tiles, num_classes = probs.shape
    preds = np.argmax(probs, axis=1).astype(np.int64)
    conf = np.max(probs, axis=1)

    console.info(f"Loaded probabilities: shape={probs.shape}")

    # Heatmaps
    if heatmaps_dir is not None or stitched_heatmap is not None or args.show_heatmap:
        vals = conf
        if args.heatmap_class is not None:
            hc = int(args.heatmap_class)
            if 0 <= hc < num_classes:
                vals = probs[:, hc]
            else:
                console.warn(f"heatmap_class {hc} out of range; using predicted confidence instead")
        hm = HeatmapGenerator(cmap="viridis")
        if heatmaps_dir is not None:
            tile_img_sz = int(cfg.get("heatmap_tile_size", 16))
            hm.save_per_tile(vals, out_dir=heatmaps_dir, tile_img_size=tile_img_sz)
            console.success(f"Saved per-tile heatmaps to: {heatmaps_dir}")
        if stitched_heatmap is not None or args.show_heatmap:
            if field_path is None:
                raise SystemExit("field_path is required for stitched heatmap generation")
            console.info(f"Loading field for stitching: {field_path}")
            field_array = load_vector_field(str(field_path), mmap=True)
            chw = ensure_chw(field_array)
            H, W = int(chw.shape[1]), int(chw.shape[2])
            hm.save_stitched(vals, out_path=stitched_heatmap, field_hw=(H, W), tile_size=tile_size, stride=stride)
            if stitched_heatmap is not None:
                console.success(f"Saved stitched heatmap to: {stitched_heatmap}")
            if args.show_heatmap:
                sh, sw = stride if stride is not None else tile_size
                n_rows = (H - tile_size[0]) // sh + 1 if H >= tile_size[0] else 0
                n_cols = (W - tile_size[1]) // sw + 1 if W >= tile_size[1] else 0
                if n_rows * n_cols == int(vals.shape[0]) and n_rows > 0 and n_cols > 0:
                    show_stitched_heatmap(vals, (n_rows, n_cols), title="Stitched Heatmap", cmap="viridis", show=True, save_path=None)
                else:
                    console.warn(f"Cannot show heatmap: grid ({n_rows}x{n_cols}) doesn't match N={vals.shape[0]}")

    # Confusion matrix
    if labels_csv is not None:
        try:
            console.info(f"Loading labels from: {labels_csv}")
            y_true = load_labels_from_csv(labels_csv)
        except Exception as e:
            raise SystemExit(f"Failed to load labels_csv '{labels_csv}': {e}")
        if y_true.shape[0] != num_tiles:
            console.warn(f"Labels length {y_true.shape[0]} does not match predictions {num_tiles}; skipping confusion matrix")
        else:
            cm = _compute_cm(y_true.astype(np.int64), preds, num_classes)
            save_confusion_outputs(cm, cfg.get("class_names"), cm_csv, cm_png, console=console)
    console.header("Done")


if __name__ == "__main__":
    main()
