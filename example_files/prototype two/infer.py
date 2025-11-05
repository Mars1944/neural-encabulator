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
from vector_field_data import ensure_chw, load_vector_field, load_vector_field_tiles, load_vector_field_tiles_with_labels
from common import DeviceSelector, infer_input_channels_from_field, load_labels_from_csv, PathResolver
from common import HeatmapGenerator
from heatmap_viewer import show_stitched_heatmap
    # labels loader available in common.load_labels_from_csv


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    # Merge nested sections onto top-level without overwriting explicit top-level keys
    for sec in ("paths", "data", "model", "training", "inference", "viewer"):
        sec_dict = config.get(sec, None)
        if isinstance(sec_dict, dict):
            for k, v in sec_dict.items():
                if k not in config:
                    config[k] = v
    # Resolve path-like keys relative to config directory; support str or list[str]
    def _anchor_one(p: str) -> str:
        q = Path(p)
        return str(q if q.is_absolute() else (path.parent / q).resolve())

    for key in ("field_path", "train_field_path", "test_field_path", "save_weights"):
        fp = config.get(key, None)
        if isinstance(fp, str) and fp.strip():
            config[key] = _anchor_one(fp)
        elif isinstance(fp, (list, tuple)):
            anchored: list[str] = []
            for item in fp:
                if isinstance(item, str) and item.strip():
                    anchored.append(_anchor_one(item))
            if anchored:
                config[key] = anchored

    # If field_path is not set, derive from train/test based on use_test flag
    use_test = bool(config.get("use_test", config.get("inference_use_test", False)))
    if not (isinstance(config.get("field_path", None), str) and str(config.get("field_path")).strip()):
        key = "test_field_path" if use_test else "train_field_path"
        fp = config.get(key, None)
        if isinstance(fp, (list, tuple)) and fp:
            config["field_path"] = str(fp[0])
        elif isinstance(fp, str) and fp.strip():
            config["field_path"] = str(fp)
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
    file_summary_out: Path | None = None,
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
            # (Removed duplicate unconditional heatmap display block)

    # File-level summary prediction: average probabilities across tiles
    try:
        probs_mean = probs.mean(dim=0).cpu().numpy()
        file_pred_idx = int(np.argmax(probs_mean))
        file_conf = float(probs_mean[file_pred_idx])
        # Choose class names from provided list or from config
        names = None
        if class_names and len(class_names) == int(logits.shape[1]):
            names = class_names
        else:
            cfg_names = config.get("class_names", None)
            if isinstance(cfg_names, list) and len(cfg_names) == int(logits.shape[1]):
                names = [str(s) for s in cfg_names]
        pred_name = names[file_pred_idx] if names is not None else f"class{file_pred_idx}"
        fname = Path(field_path).name
        if file_summary_out is not None:
            try:
                file_summary_out.parent.mkdir(parents=True, exist_ok=True)
                import csv
                write_header = not file_summary_out.exists()
                with file_summary_out.open("a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(["file_name", "predicted_class", "predicted_index", "confidence_percent"]) 
                    w.writerow([fname, pred_name, file_pred_idx, round(file_conf * 100.0, 2)])
                print(f"Appended file summary for {fname} -> {pred_name} ({file_conf*100.0:.2f}%) to {file_summary_out}")
            except Exception as e:
                # Fallback: write to a timestamped file if the target is locked (e.g., open in Excel)
                try:
                    from datetime import datetime
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    alt = file_summary_out.with_name(f"{file_summary_out.stem}-{ts}{file_summary_out.suffix}")
                    alt.parent.mkdir(parents=True, exist_ok=True)
                    import csv
                    with alt.open("w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(["file_name", "predicted_class", "predicted_index", "confidence_percent"]) 
                        w.writerow([fname, pred_name, file_pred_idx, round(file_conf * 100.0, 2)])
                    print(
                        f"[info] Summary file locked ('{file_summary_out}'); wrote to '{alt}' instead."
                    )
                except Exception as e2:
                    print(f"[warn] Failed to write file summary (fallback also failed): {e2}")
        else:
            print(f"File-level prediction for {fname}: {pred_name} ({file_conf*100.0:.2f}%)")
    except Exception as e:
        print(f"[warn] Failed to compute file-level summary: {e}")

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

    # If labels still missing, try deriving from embedded CSV labels using the same tiling
    if (cm_csv_out is not None or cm_png_out is not None) and y_true is None:
        try:
            if str(Path(field_path).suffix).lower() == ".csv":
                # Recompute labels per tile without dropping unlabeled to preserve alignment
                _, y_emb = load_vector_field_tiles_with_labels(
                    path=field_path,
                    tile_size=(int(tile_size[0]), int(tile_size[1])),
                    stride=stride,
                    add_magnitude=add_mag,
                    normalize=normalize,
                    limit_tiles=int(tile_batch.shape[0]),
                    label_aggregation="majority",
                    drop_unlabeled=False,
                )
                if y_emb.shape[0] == int(tile_batch.shape[0]):
                    y_true = y_emb.astype(np.int64)
                else:
                    print(
                        f"[warn] Embedded labels length {y_emb.shape[0]} doesn't match tiles {tile_batch.shape[0]}; skipping embedded labels"
                    )
        except Exception as e:
            print(f"[warn] Failed to derive embedded labels from CSV: {e}")

    # If still missing labels and CM is requested, optionally auto-dump a labels template CSV
    if (cm_csv_out is not None or cm_png_out is not None) and y_true is None:
        auto_dump = bool(config.get("auto_dump_labels_template", True))
        if auto_dump:
            try:
                stem = Path(field_path).stem
                base_dir = cm_csv_out.parent if cm_csv_out is not None else (cm_png_out.parent if cm_png_out is not None else Path("."))
                tmpl_path = base_dir / f"labels_template-{stem}.csv"
                tmpl_path.parent.mkdir(parents=True, exist_ok=True)
                import csv

                # Provide a helper template with predicted_class for convenience
                with tmpl_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["tile_index", "label", "predicted_class"])  # fill label with 0..C-1
                    for i in range(int(pred.shape[0])):
                        w.writerow([i, -1, int(pred[i].item())])
                print(
                    f"[info] No labels provided; wrote labels template to: {tmpl_path}. Fill the 'label' column and pass via --labels-csv."
                )
            except Exception as e:
                print(f"[warn] Failed to write labels template CSV: {e}")

    if (cm_csv_out is not None or cm_png_out is not None) and y_true is not None:
        cm_debug = bool(config.get("cm_debug", False))
        y_pred = pred.cpu().numpy().astype(np.int64)
        n = tile_batch.shape[0]
        if y_true.shape[0] != n:
            print(f"[warn] Labels length {y_true.shape[0]} does not match tiles {n}; skipping CM")
        else:
            # Mask out unlabeled entries (-1 or <0)
            if cm_debug:
                try:
                    uniq, cnt = np.unique(y_true, return_counts=True)
                    print(f"[cm_debug] y_true unique raw: {list(zip(uniq.tolist(), cnt.tolist()))}")
                except Exception:
                    pass
            mask = y_true >= 0
            y_true_masked = y_true[mask].astype(np.int64)
            y_pred_masked = y_pred[mask]
            num_classes = int(logits.shape[1])

            # Remap labels to zero-based indices if they are not in [0, num_classes-1]
            # This handles CSV-derived labels like {1,2} or {2} by mapping -> {0,1} or {0}
            if y_true_masked.size > 0:
                uniq = np.unique(y_true_masked)
                expected = np.arange(num_classes, dtype=np.int64)
                if not (uniq.min() >= 0 and uniq.max() < num_classes and np.array_equal(uniq, uniq.astype(np.int64))):
                    # Build a mapping from sorted unique non-negative labels to 0..K-1 (up to num_classes)
                    valid = [int(v) for v in uniq if v >= 0]
                    mapping = {lbl: i for i, lbl in enumerate(sorted(valid)[:num_classes])}
                    y_true_mapped = np.array([mapping.get(int(v), -1) for v in y_true_masked], dtype=np.int64)
                    keep = y_true_mapped >= 0
                    y_true_masked = y_true_mapped[keep]
                    y_pred_masked = y_pred_masked[keep]

            if cm_debug:
                print(f"[cm_debug] kept after mask: {int(y_true_masked.size)} of {int(n)}")
            if y_true_masked.size == 0:
                print("[warn] No valid ground-truth labels available for confusion matrix; skipping")
                cm = np.zeros((num_classes, num_classes), dtype=np.int64)
            else:
                idx = (y_true_masked * num_classes + y_pred_masked).astype(np.int64)
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

            # Print simple tile-level accuracy if any labels present
            total = int(cm.sum())
            correct = int(np.trace(cm)) if cm.size > 0 else 0
            if total > 0:
                acc = correct / total
                print(f"Tile accuracy: {acc*100.0:.2f}% ({correct}/{total})")

            # Save WEKA-style text confusion matrix
            try:
                # Decide path: prefer alongside cm_csv_out, else alongside cm_png_out
                weka_txt_out = None
                if cm_csv_out is not None:
                    weka_txt_out = cm_csv_out.with_suffix(".txt")
                elif cm_png_out is not None:
                    weka_txt_out = cm_png_out.with_suffix(".txt")
                if weka_txt_out is not None:
                    weka_txt_out.parent.mkdir(parents=True, exist_ok=True)

                    # Build class label names
                    labels = (
                        class_names if (class_names and len(class_names) == num_classes) else [f"class{i}" for i in range(num_classes)]
                    )

                    # Letter codes: a, b, ..., z, aa, ab, ...
                    def code(k: int) -> str:
                        s = ""
                        k0 = k
                        while True:
                            s = chr(ord('a') + (k % 26)) + s
                            k //= 26
                            if k == 0:
                                break
                            k -= 1
                        return s

                    # Column width based on max count
                    max_count = int(cm.max()) if cm.size > 0 else 0
                    cell_w = max(3, len(str(max_count)))

                    lines = []
                    lines.append("=== Confusion Matrix ===")
                    # Header row with letter codes
                    header = "  " + " ".join(code(j).rjust(cell_w) for j in range(num_classes)) + "   <-- classified as"
                    lines.append(header)
                    for i in range(num_classes):
                        row_counts = " ".join(str(int(cm[i, j])).rjust(cell_w) for j in range(num_classes))
                        lines.append(f" {row_counts} | {code(i)} = {labels[i]}")

                    with weka_txt_out.open("w", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")
                    print(f"Saved WEKA-style CM to: {weka_txt_out}")
            except Exception as e:
                print(f"[warn] Failed to save WEKA-style CM: {e}")

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
        show_heatmap: bool,
        file_summary: str | None,
        file_labels: str | None,
        recurse: bool,
    ) -> None:
        # Resolve field paths (arg overrides config). Support list or delimited string.
        fields: list[str] = []
        if isinstance(field_path, str) and field_path.strip():
            s = field_path.strip()
            parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
            fields = parts if len(parts) > 1 else [s]
        else:
            fp_cfg = self.config.get("field_path", None)
            if isinstance(fp_cfg, (list, tuple)):
                fields = [str(p) for p in fp_cfg if str(p).strip()]
            elif isinstance(fp_cfg, str) and fp_cfg.strip():
                fields = [fp_cfg]
            else:
                use_test = bool(self.config.get("use_test", False))
                key = "test_field_path" if use_test else "train_field_path"
                alt = self.config.get(key, None)
                if isinstance(alt, (list, tuple)):
                    fields = [str(p) for p in alt if str(p).strip()]
                elif isinstance(alt, str) and alt.strip():
                    fields = [alt]
        if not fields:
            raise RuntimeError("field_path or test/train_field_path must be provided via CLI or config")

        # Anchor to config directory and expand directories/globs
        anchored_fields: list[str] = []
        expanded: list[str] = []
        for f in fields:
            raw = f
            p = Path(f)
            if not p.is_absolute():
                p = (self.cfg_path.parent / p).resolve()

            # Expand glob patterns relative to the config dir (supports **)
            if any(ch in raw for ch in "*?[]"):
                try:
                    parent = p.parent
                    pattern = p.name
                    for m in parent.glob(pattern):
                        if m.is_file():
                            expanded.append(str(m.resolve()))
                except Exception:
                    pass
                continue

            # If directory, collect CSV and NumPy files (optionally recursive)
            if p.is_dir():
                try:
                    patterns = ["*.csv", "*.npy", "*.npz"]
                    found = []
                    for patt in patterns:
                        it = p.rglob(patt) if recurse else p.glob(patt)
                        for m in it:
                            if m.is_file():
                                found.append(str(m.resolve()))
                    expanded.extend(found)
                    # Report discovery counts by type
                    def _cnt(ext: str) -> int:
                        return sum(1 for f in found if f.lower().endswith(ext))
                    print(
                        f"Discovered {_cnt('.csv')} CSV, {_cnt('.npy')} NPY, {_cnt('.npz')} NPZ files under: {p}"
                    )
                except Exception:
                    pass
            else:
                anchored_fields.append(str(p))

        if expanded:
            uniq = sorted({x for x in expanded})
            anchored_fields.extend(uniq)
        # Primary field for building model / inferring channels
        use_field = anchored_fields[0]

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

        # Resolve optional outputs from CLI or fallback to config
        cfg = self.config
        # Respect config toggles
        gen_hm = bool(cfg.get("generate_heatmap", True))
        gen_cm = bool(cfg.get("generate_confusion_matrix", True))

        if output is None:
            output = cfg.get("output", None)
        if probs_out is None:
            probs_out = cfg.get("probs_out", None)
        if logits_out is None:
            logits_out = cfg.get("logits_out", None)

        # Heatmap paths
        if heatmaps_dir is None:
            heatmaps_dir = cfg.get("heatmaps_dir", None)
        if stitch_heatmap is None:
            stitch_heatmap = cfg.get("stitched_heatmap", None)
        # CM paths
        if cm_csv is None:
            cm_csv = cfg.get("cm_csv", None)
        if cm_png is None:
            cm_png = cfg.get("cm_png", None)

        # File summary path (per-file prediction)
        if file_summary is None:
            file_summary = cfg.get("file_summary", None)

        # Allow config to enable showing heatmap if CLI flag not set
        if not show_heatmap:
            show_heatmap = bool(cfg.get("show_heatmap", False))

        # Honor generate_* toggles
        if not gen_hm:
            heatmaps_dir = None
            stitch_heatmap = None
        if not gen_cm:
            cm_csv = None
            cm_png = None

        # Use PathResolver to anchor relative paths to the config directory and outputs_root
        resolver = PathResolver(self.cfg_path, outputs_root=str(cfg.get("outputs_root")) if isinstance(cfg.get("outputs_root"), str) else None)

        # Anchor generic outputs
        out_path = resolver.anchor(output) if output else None
        probs_path = resolver.anchor(probs_out) if probs_out else None
        logits_path = resolver.anchor(logits_out) if logits_out else None

        # File labels manifest (for test CSVs without label column)
        labels_manifest_path: Path | None = None
        manifest_cfg = cfg.get("file_labels", cfg.get("file_labels_manifest", None))
        use_manifest = file_labels if (isinstance(file_labels, str) and file_labels.strip()) else manifest_cfg
        if isinstance(use_manifest, str) and use_manifest.strip():
            p = Path(use_manifest)
            if not p.is_absolute():
                p = (self.cfg_path.parent / p).resolve()
            labels_manifest_path = p

        labels_map: dict[str, int] | None = None
        if labels_manifest_path is not None and labels_manifest_path.exists():
            try:
                import csv

                labels_map = {}
                with labels_manifest_path.open("r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                start = 0
                if rows and len(rows[0]) >= 2 and not rows[0][1].strip().isdigit():
                    start = 1
                for r in rows[start:]:
                    if len(r) < 2:
                        continue
                    name = str(r[0]).strip()
                    try:
                        lab = int(str(r[1]).strip())
                    except Exception:
                        continue
                    if name:
                        # store lookup by original, basename, and stem in lowercase
                        labels_map[name.lower()] = lab
                        labels_map[Path(name).name.lower()] = lab
                        labels_map[Path(name).stem.lower()] = lab
                print(f"Loaded file labels manifest: {labels_manifest_path}")
            except Exception as e:
                print(f"[warn] Failed to load file labels manifest '{labels_manifest_path}': {e}")

        # Decide heatmap paths (auto-default under outputs_root when enabled)
        heat_dir, stitch_path = resolver.decide_heatmap_paths(
            generate=gen_hm,
            heatmaps_dir=heatmaps_dir,
            stitched=stitch_heatmap,
        )

        # Decide confusion matrix paths (auto-default under outputs_root when enabled)
        cm_csv_path, cm_png_path = resolver.decide_cm_paths(
            generate=gen_cm,
            cm_csv=cm_csv,
            cm_png=cm_png,
        )

        # Decide per-file summary path
        file_summary_path: Path | None = None
        if isinstance(file_summary, str) and file_summary.strip():
            file_summary_path = resolver.anchor(file_summary)
        # If summary path is locked (e.g., open in Excel), choose a single fallback for this entire run
        effective_summary_path: Path | None = file_summary_path
        if file_summary_path is not None:
            try:
                file_summary_path.parent.mkdir(parents=True, exist_ok=True)
                with file_summary_path.open("a", encoding="utf-8"):
                    pass
            except Exception:
                try:
                    from datetime import datetime
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    alt = file_summary_path.with_name(f"{file_summary_path.stem}-{ts}{file_summary_path.suffix}")
                    alt.parent.mkdir(parents=True, exist_ok=True)
                    effective_summary_path = alt
                    print(f"[info] Summary file locked ('{file_summary_path}'); using '{alt}' for this run.")
                except Exception as e:
                    print(f"[warn] Failed to establish file summary path: {e}")

        # Run inference for each field path; suffix outputs by input stem to avoid overwrites
        for fp in anchored_fields:
            stem = Path(fp).stem
            per_out = out_path.with_name(f"{out_path.stem}-{stem}{out_path.suffix}") if out_path is not None else None
            per_probs = (
                probs_path.with_name(f"{probs_path.stem}-{stem}{probs_path.suffix}") if probs_path is not None else None
            )
            per_logits = (
                logits_path.with_name(f"{logits_path.stem}-{stem}{logits_path.suffix}") if logits_path is not None else None
            )
            per_heat_dir = (heat_dir / stem) if heat_dir is not None else None
            per_stitch = (
                stitch_path.with_name(f"{stitch_path.stem}-{stem}{stitch_path.suffix}") if stitch_path is not None else None
            )
            per_cm_csv = (
                cm_csv_path.with_name(f"{cm_csv_path.stem}-{stem}{cm_csv_path.suffix}") if cm_csv_path is not None else None
            )
            per_cm_png = (
                cm_png_path.with_name(f"{cm_png_path.stem}-{stem}{cm_png_path.suffix}") if cm_png_path is not None else None
            )

            run_inference(
                self.config,
                self.device,
                model,
                fp,
                per_out,
                probs_out=per_probs,
                logits_out=per_logits,
                heatmaps_dir=per_heat_dir,
                heatmap_class=heatmap_class,
                stitch_heatmap_path=per_stitch,
                labels_csv=Path(labels_csv) if labels_csv else None,
                label_all=(
                    (
                        labels_map.get(Path(fp).name.lower(), labels_map.get(Path(fp).stem.lower()))
                        if labels_map is not None
                        else label_all
                    )
                    if label_all is None
                    else label_all
                ),
                labels_array=labels_array,
                class_names=class_names,
                cm_csv_out=per_cm_csv,
                cm_png_out=per_cm_png,
                show_heatmap=show_heatmap,
                file_summary_out=effective_summary_path,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference runner for SimpleCNN on vector fields (prototype two)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.json")),
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
        "--recurse",
        action="store_true",
        help="When --field_path includes directories, search subfolders recursively for *.csv, *.npy, *.npz",
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
        "--file_summary",
        type=str,
        default=None,
        help="CSV path to append per-file predictions: file_name,predicted_class,predicted_index,confidence_percent",
    )
    parser.add_argument(
        "--file_labels",
        type=str,
        default=None,
        help="CSV mapping file_name,label (0-based) to use as ground-truth for confusion matrix when test CSVs have no label column",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help="CSV with tile_index,label for confusion matrix (header optional)",
    )
    parser.add_argument(
        "--cm_debug",
        action="store_true",
        help="Print confusion matrix label stats and filtering info",
    )
    parser.add_argument(
        "--dump_labels_template",
        action="store_true",
        help="When true and labels are missing, write a labels template CSV under outputs",
    )
    parser.add_argument(
        "--label_all",
        type=int,
        default=None,
        help="Assign a single true label to all tiles for CM",
    )
    args = parser.parse_args()

    runner = InferenceRunner(cfg_path=Path(args.config), device_pref=args.device)
    # Allow CLI to opt-in template dumping even if not in config
    if bool(args.dump_labels_template):
        try:
            runner.config["auto_dump_labels_template"] = True
        except Exception:
            pass
    if bool(args.cm_debug):
        try:
            runner.config["cm_debug"] = True
        except Exception:
            pass
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
        file_summary=args.file_summary,
        file_labels=args.file_labels,
        recurse=bool(args.recurse),
    )


if __name__ == "__main__":
    main()
