import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import numpy as np

# Ensure local imports work whether run or imported
import sys as _sys
_sys.path.append(str(Path(__file__).parent))

from cnn_model import SimpleCNN
from vector_field_data import ensure_chw, load_vector_field, load_vector_field_tiles
from common import DeviceSelector, infer_input_channels_from_field, load_labels_from_csv
from heatmap_viewer import show_stitched_heatmap
    # labels loader available in common.load_labels_from_csv


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    # Normalize field_path relative to config file if present
    field_path_value = config.get("field_path", "")
    if isinstance(field_path_value, str) and field_path_value.strip():
        resolved_path = Path(field_path_value)
        if not resolved_path.is_absolute():
            config["field_path"] = str((path.parent / resolved_path).resolve())
    return config


def choose_device(pref: str | None) -> torch.device:
    return DeviceSelector.choose(pref)


    # Note: prefer common.infer_input_channels_from_field in new code


def build_model(config: Dict[str, Any], inferred_in_ch: int | None) -> SimpleCNN:
    if inferred_in_ch is not None:
        config = dict(config)
        config["input_channels"] = int(inferred_in_ch)
    model = SimpleCNN(config)
    return model


def load_weights(model: torch.nn.Module, weights_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(str(weights_path), map_location=device)
    state: Dict[str, Any] | None = None
    if isinstance(checkpoint, dict):
        # Try common keys
        for key in ("state_dict", "model_state", "model", "ema_state", "weights"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
        if state is None:
            # Might already be a bare state dict
            state = checkpoint if all(isinstance(k, str) for k in checkpoint.keys()) else None
    if state is None:
        raise RuntimeError("Unsupported checkpoint format: could not find state_dict")

    # Strip possible 'module.' prefix (from DataParallel)
    def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not state_dict:
            return state_dict
        if all(k.startswith("module.") for k in state_dict.keys()):
            return {k[len("module."):]: v for k, v in state_dict.items()}
        return state_dict

    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=False)


## HeatmapGenerator moved to common. Import from common above.
def run_inference(
    config: Dict[str, Any],
    device: torch.device,
    model: torch.nn.Module,
    field_path: str,
    output: Path | None,
    probs_out: Path | None = None,
    logits_out: Path | None = None,
    heatmaps_dir: Path | None = None,
    heatmap_class: int | None = None,
    stitch_heatmap_path: Path | None = None,
    labels_csv: Path | None = None,
    label_all: int | None = None,
    labels_array: "list[int] | np.ndarray | None" = None,
    class_names: "list[str] | None" = None,
    cm_csv_out: Path | None = None,
    cm_png_out: Path | None = None,
    show_heatmap: bool = False,
) -> None:
    tile_size = tuple(config.get("tile_size", [256, 256]))
    stride_cfg = config.get("tile_stride", None)
    stride: Tuple[int, int] | None = None
    if isinstance(stride_cfg, (list, tuple)):
        stride = (int(stride_cfg[0]), int(stride_cfg[1]))
    add_mag = bool(config.get("add_magnitude", True))
    normalize = bool(config.get("normalize", True))
    limit_tiles = config.get("limit_tiles", None)

    tile_batch = load_vector_field_tiles(
        path=field_path,
        tile_size=(int(tile_size[0]), int(tile_size[1])),
        stride=stride,
        add_magnitude=add_mag,
        normalize=normalize,
        limit_tiles=None if limit_tiles is None else int(limit_tiles),
    )
    print(f"Loaded tiles: shape={tile_batch.shape}")

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(tile_batch).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    # Summaries
    print(f"Logits shape: {tuple(logits.shape)} | num_classes={logits.shape[1]}")
    topk = min(5, logits.shape[0])
    print("Sample predictions (first N tiles):")
    for i in range(topk):
        print(f"  tile[{i:03d}]: class={pred[i].item()} prob={conf[i].item():.4f}")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Save CSV with tile_index, predicted_class, confidence
        import csv

        with output.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["tile_index", "predicted_class", "confidence"])
            for i in range(pred.shape[0]):
                w.writerow([i, int(pred[i].item()), float(conf[i].item())])
        print(f"Saved predictions to: {output}")

    # Optional: save full probabilities/logits per tile
    if probs_out is not None:
        probs_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(probs_out.with_suffix(".npy")), probs.cpu().numpy())
        print(f"Saved probabilities (N,C) to: {probs_out.with_suffix('.npy')}")
    if logits_out is not None:
        logits_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(logits_out.with_suffix(".npy")), logits.cpu().numpy())
        print(f"Saved logits (N,C) to: {logits_out.with_suffix('.npy')}")

    # Optional: heatmaps per tile and stitched heatmap via HeatmapGenerator
    if heatmaps_dir is not None or stitch_heatmap_path is not None or show_heatmap:
        probs_cpu = probs.cpu().numpy()
        if heatmap_class is not None:
            hc = int(heatmap_class)
            if hc < 0 or hc >= probs_cpu.shape[1]:
                print(f"[warn] heatmap_class {hc} out of range; using predicted class conf")
                vals = conf.cpu().numpy()
            else:
                vals = probs_cpu[:, hc]
        else:
            vals = conf.cpu().numpy()

        hm = HeatmapGenerator(cmap="viridis")

        if heatmaps_dir is not None:
            hm.save_per_tile(vals, out_dir=heatmaps_dir, tile_img_size=16)
            print(f"Saved per-tile heatmaps to: {heatmaps_dir}")

        if stitch_heatmap_path is not None or show_heatmap:
            field_array = load_vector_field(field_path, mmap=True)
            channels_first = ensure_chw(field_array)
            H, W = int(channels_first.shape[1]), int(channels_first.shape[2])
            if stitch_heatmap_path is not None:
                hm.save_stitched(
                    vals,
                    out_path=stitch_heatmap_path,
                    field_hw=(H, W),
                    tile_size=(int(tile_size[0]), int(tile_size[1])),
                    stride=stride,
                )
            # Also show an interactive window if requested and matplotlib is available
            if show_heatmap:
                if isinstance(stride, tuple):
                    sh, sw = int(stride[0]), int(stride[1])
                else:
                    sh, sw = int(tile_size[0]), int(tile_size[1])
                n_rows = (H - int(tile_size[0])) // sh + 1 if H >= int(tile_size[0]) else 0
                n_cols = (W - int(tile_size[1])) // sw + 1 if W >= int(tile_size[1]) else 0
                if n_rows * n_cols == int(vals.shape[0]) and n_rows > 0 and n_cols > 0:
                    show_stitched_heatmap(
                        vals,
                        (n_rows, n_cols),
                        title=f"Stitched Heatmap (class {heatmap_class if heatmap_class is not None else 'pred'})",
                        cmap="viridis",
                    )
                else:
                    print(f"[warn] Cannot show heatmap: grid ({n_rows}x{n_cols}) doesn't match N={vals.shape[0]}")
            # Also show an interactive window if matplotlib is available
            if isinstance(stride, tuple):
                sh, sw = int(stride[0]), int(stride[1])
            else:
                sh, sw = int(tile_size[0]), int(tile_size[1])
            n_rows = (H - int(tile_size[0])) // sh + 1 if H >= int(tile_size[0]) else 0
            n_cols = (W - int(tile_size[1])) // sw + 1 if W >= int(tile_size[1]) else 0
            if n_rows * n_cols == int(vals.shape[0]) and n_rows > 0 and n_cols > 0:
                show_stitched_heatmap(
                    vals,
                    (n_rows, n_cols),
                    title=f"Stitched Heatmap (class {heatmap_class if heatmap_class is not None else 'pred'})",
                    cmap="viridis",
                )

    # Optional: confusion matrix
    # Resolve labels precedence: labels_array > labels_csv > label_all > config keys
    y_true: np.ndarray | None = None
    if labels_array is not None:
        try:
            y_true = np.asarray(labels_array, dtype=np.int64)
        except Exception:
            y_true = None
    if y_true is None and isinstance(labels_csv, Path):
        try:
            y_true = load_labels_from_csv(labels_csv)
        except Exception as e:
            print(f"[warn] Failed to load labels_csv '{labels_csv}': {e}")
    if y_true is None and label_all is not None:
        y_true = np.full((tile_batch.shape[0],), int(label_all), dtype=np.int64)
    if y_true is None:
        cfg_csv = config.get("labels_csv", None)
        if isinstance(cfg_csv, str) and cfg_csv.strip():
            try:
                y_true = load_labels_from_csv(Path(cfg_csv))
            except Exception as e:
                print(f"[warn] Failed to load config.labels_csv '{cfg_csv}': {e}")
        if y_true is None and config.get("label_all", None) is not None:
            try:
                y_true = np.full((tile_batch.shape[0],), int(config.get("label_all")), dtype=np.int64)
            except Exception:
                pass
        if y_true is None:
            cfg_labels = config.get("labels", None)
            if isinstance(cfg_labels, list):
                try:
                    y_true = np.asarray([int(x) for x in cfg_labels], dtype=np.int64)
                except Exception:
                    y_true = None

    if (cm_csv_out is not None or cm_png_out is not None) and y_true is not None:
        y_pred = pred.cpu().numpy().astype(np.int64)
        n = tile_batch.shape[0]
        if y_true.shape[0] != n:
            print(f"[warn] Labels length {y_true.shape[0]} does not match tiles {n}; skipping CM")
        else:
            num_classes = int(logits.shape[1])
            idx = (y_true * num_classes + y_pred).astype(np.int64)
            cm = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

            # Save CSV counts
            if cm_csv_out is not None:
                try:
                    cm_csv_out.parent.mkdir(parents=True, exist_ok=True)
                    import csv

                    with cm_csv_out.open("w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(["true\\pred"] + [f"{j}" for j in range(num_classes)])
                        for i in range(num_classes):
                            w.writerow([f"{i}"] + [int(cm[i, j]) for j in range(num_classes)])
                    print(f"Saved confusion matrix CSV to: {cm_csv_out}")
                except Exception as e:
                    print(f"[warn] Failed to save cm_csv: {e}")

            # Save PNG normalized per-column if matplotlib/PIL available
            if cm_png_out is not None:
                try:
                    col_sums = cm.sum(axis=0, keepdims=True).astype(np.float32)
                    norm = np.divide(
                        cm.astype(np.float32),
                        np.maximum(col_sums, 1.0),
                        out=np.zeros_like(cm, dtype=np.float32),
                        where=col_sums > 0,
                    )
                    class_labels = (
                        class_names if (class_names and len(class_names) == num_classes) else [f"{i}" for i in range(num_classes)]
                    )
                    try:
                        import matplotlib.pyplot as plt  # type: ignore

                        plt.figure(figsize=(max(4, num_classes), max(3, num_classes * 0.6)))
                        im = plt.imshow(norm, vmin=0.0, vmax=1.0, cmap="viridis")
                        plt.colorbar(im, fraction=0.046, pad=0.04, label="col-normalized")
                        plt.xticks(range(num_classes), class_labels, rotation=45, ha="right")
                        plt.yticks(range(num_classes), class_labels)
                        plt.xlabel("Predicted")
                        plt.ylabel("True")
                        plt.title("Confusion Matrix (column-normalized)")
                        plt.tight_layout()
                        cm_png_out.parent.mkdir(parents=True, exist_ok=True)
                        plt.savefig(cm_png_out)
                        plt.close()
                        print(f"Saved confusion matrix image to: {cm_png_out}")
                    except Exception:
                        from PIL import Image  # type: ignore

                        arr = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
                        img = Image.fromarray(arr, mode="L").resize(
                            (num_classes * 32, num_classes * 32), resample=Image.NEAREST
                        )
                        cm_png_out.parent.mkdir(parents=True, exist_ok=True)
                        img.save(cm_png_out)
                        print(f"Saved confusion matrix image (grayscale) to: {cm_png_out}")
                except Exception as e:
                    # Final fallback: save normalized matrix as .npy if image backends fail
                    try:
                        cm_png_out.parent.mkdir(parents=True, exist_ok=True)
                        np.save(str(cm_png_out.with_suffix(".npy")), cm.astype(np.float32))
                        print(
                            f"[warn] Could not save CM image; saved counts as NPY: {cm_png_out.with_suffix('.npy')} (error: {e})"
                        )
                    except Exception as e2:
                        print(f"[warn] Failed to save confusion matrix image or NPY: {e2}")


class InferenceRunner:
    """
    High-level, class-based inference runner for prototype two.
    Keeps CLI thin and supports the same config keys as prototype one.
    """

    def __init__(self, cfg_path: Path, device_pref: str = "auto") -> None:
        self.cfg_path = cfg_path
        self.config = load_config(cfg_path)
        self.device = DeviceSelector.choose(device_pref)
        print(f"Loaded config from: {self.cfg_path}")
        print(f"Using device: {self.device}")

    def run(
        self,
        *,
        weights: str | None,
        field_path: str | None,
        output: str | None,
        probs_out: str | None,
        logits_out: str | None,
        heatmaps_dir: str | None,
        heatmap_class: int | None,
        stitch_heatmap: str | None,
        labels_csv: str | None,
        label_all: int | None,
        labels_array: "list[int] | np.ndarray | None",
        class_names: "list[str] | None",
        cm_csv: str | None,
        cm_png: str | None,
    ) -> None:
        # Resolve field path (arg overrides config)
        use_field = field_path if field_path is not None else self.config.get("field_path", "")
        if not isinstance(use_field, str) or not use_field.strip():
            raise RuntimeError("field_path must be provided via --field_path or config")
        use_field = str(Path(use_field))

        # Resolve weights path
        weights_path: str | None = None
        candidates = []
        if isinstance(weights, str) and weights.strip():
            candidates.append(weights)
        for k in ("weights", "weights_path", "save_weights"):
            v = self.config.get(k, None)
            if isinstance(v, str) and v.strip():
                candidates.append(v)
        resolved_candidates = []
        for w in candidates:
            p = Path(w)
            if not p.is_absolute():
                p = (self.cfg_path.parent / p).resolve()
            resolved_candidates.append(str(p))
            if p.exists():
                weights_path = str(p)
                break
        if weights_path is None:
            runs_dir = (self.cfg_path.parent / "runs").resolve()
            latest: Path | None = None
            if runs_dir.exists() and runs_dir.is_dir():
                files = list(runs_dir.glob("*.pth")) + list(runs_dir.glob("*.pt"))
                if files:
                    latest = max(files, key=lambda f: f.stat().st_mtime)
            if latest is not None:
                weights_path = str(latest)
                print(f"Auto-selected latest weights from runs/: {weights_path}")
        if weights_path is None:
            tried = ", ".join(resolved_candidates) or "<none>"
            raise RuntimeError(
                f"Weights file not provided or not found. Pass --weights, set weights/weights_path/save_weights in config, or place a .pth/.pt in runs/. Tried: {tried}"
            )
        print(f"Using weights: {weights_path}")

        # Infer channels, build model, load weights
        try:
            in_ch = infer_input_channels_from_field(self.config, use_field)
        except Exception as e:
            raise RuntimeError(f"Failed to infer input channels from field: {e}") from e
        model = build_model(self.config, inferred_in_ch=in_ch).to(self.device)
        load_weights(model, Path(weights_path), self.device)
        print("Model and weights loaded.")

        out_path = Path(output) if output else None
        probs_path = Path(probs_out) if probs_out else None
        logits_path = Path(logits_out) if logits_out else None
        heat_dir = Path(heatmaps_dir) if heatmaps_dir else None
        stitch_path = Path(stitch_heatmap) if stitch_heatmap else None

        run_inference(
            self.config,
            self.device,
            model,
            use_field,
            out_path,
            probs_out=probs_path,
            logits_out=logits_path,
            heatmaps_dir=heat_dir,
            heatmap_class=heatmap_class,
            stitch_heatmap_path=stitch_path,
            labels_csv=Path(labels_csv) if labels_csv else None,
            label_all=label_all,
            labels_array=labels_array,
            class_names=class_names,
            cm_csv_out=Path(cm_csv) if cm_csv else None,
            cm_png_out=Path(cm_png) if cm_png else None,
            show_heatmap=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference runner for SimpleCNN on vector fields (prototype two)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("cnn_vector_config.json")),
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained weights (.pt/.pth). If omitted, reads from config keys: weights | weights_path | save_weights",
    )
    parser.add_argument(
        "--field_path",
        type=str,
        default=None,
        help="Override vector field path (.npy/.npz/.csv)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto|cpu|cuda[:i]|mps",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV path to save tile predictions",
    )
    parser.add_argument(
        "--probs_out",
        type=str,
        default=None,
        help="Optional .npy path to save per-tile probabilities (N,C)",
    )
    parser.add_argument(
        "--logits_out",
        type=str,
        default=None,
        help="Optional .npy path to save per-tile logits (N,C)",
    )
    parser.add_argument(
        "--heatmaps_dir",
        type=str,
        default=None,
        help="Directory to save per-tile heatmap images (uses matplotlib or PIL if available)",
    )
    parser.add_argument(
        "--heatmap_class",
        type=int,
        default=None,
        help="Class index for heatmaps; default uses predicted confidence per tile",
    )
    parser.add_argument(
        "--stitch_heatmap",
        type=str,
        default=None,
        help="Path to save a stitched tile-grid heatmap image or .npy",
    )
    parser.add_argument(
        "--show_heatmap",
        action="store_true",
        help="Show stitched heatmap window after inference (requires matplotlib)",
    )
    parser.add_argument(
        "--cm_csv",
        type=str,
        default=None,
        help="Path to save confusion matrix as CSV (counts)",
    )
    parser.add_argument(
        "--cm_png",
        type=str,
        default=None,
        help="Path to save confusion matrix image (PNG); uses column-normalized percentages",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help="CSV with tile_index,label for confusion matrix (header optional)",
    )
    parser.add_argument(
        "--label_all",
        type=int,
        default=None,
        help="Assign a single true label to all tiles for CM",
    )
    args = parser.parse_args()

    runner = InferenceRunner(cfg_path=Path(args.config), device_pref=args.device)
    runner.run(
        weights=args.weights,
        field_path=args.field_path,
        output=args.output,
        probs_out=args.probs_out,
        logits_out=args.logits_out,
        heatmaps_dir=args.heatmaps_dir,
        heatmap_class=args.heatmap_class,
        stitch_heatmap=args.stitch_heatmap,
        labels_csv=args.labels_csv,
        label_all=args.label_all,
        labels_array=None,
        class_names=None,
        cm_csv=args.cm_csv,
        cm_png=args.cm_png,
        show_heatmap=bool(args.show_heatmap),
    )


if __name__ == "__main__":
    main()
