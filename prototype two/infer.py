import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import time
import numpy as np

# Ensure local imports work whether run or imported
import sys as _sys
_sys.path.append(str(Path(__file__).parent))

from cnn_model import SimpleCNN, MultiTaskCNN
from vector_field_data import ensure_chw, load_vector_field, load_vector_field_tiles, load_vector_field_tiles_with_labels
from common import (
    DeviceSelector,
    infer_input_channels_from_field,
    load_labels_from_csv,
    PathResolver,
    ConfigManager,
    parse_field_paths,
    choose_fields,
    write_labels_template,
    save_confusion_outputs,
    ensure_outputs_ready,
)
from common import HeatmapGenerator
from heatmap_viewer import show_stitched_heatmap
from console import console_from_config
try:
    from data_loader import build_class_index as _build_class_index, _resolve_root_path as _resolve_root_path_img
except Exception:
    _build_class_index = None  # type: ignore
    _resolve_root_path_img = None  # type: ignore


## Config loading and device selection are consolidated in common.ConfigManager and DeviceSelector


    # Note: prefer common.infer_input_channels_from_field in new code


def build_model(config: Dict[str, Any], inferred_in_ch: int | None) -> torch.nn.Module:
    if inferred_in_ch is not None:
        config = dict(config)
        config["input_channels"] = int(inferred_in_ch)
    task = str(config.get("task", "")).lower()
    multitask = task.startswith("multi") or bool(config.get("multitask", False))
    model = MultiTaskCNN(config) if multitask else SimpleCNN(config)
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
    report_list: "list[tuple[str, str, int, float]] | None" = None,
    report_probs_list: "list[tuple[str, list[float]]] | None" = None,
) -> "np.ndarray | None":
    c = console_from_config(config)
    # Plotting/visual outputs are handled separately to keep inference focused on predictions.
    heatmaps_dir = None
    stitch_heatmap_path = None
    cm_csv_out = None
    cm_png_out = None
    show_heatmap = False
    # Ensure return variable is always defined
    cm: np.ndarray | None = None
    # Allow inference-specific overrides for tiling
    inf_ts = config.get("inference_tile_size", None)
    inf_tr = config.get("inference_tile_stride", None)
    tile_size = tuple(inf_ts if isinstance(inf_ts, (list, tuple)) else config.get("tile_size", [256, 256]))
    stride_cfg = inf_tr if isinstance(inf_tr, (list, tuple)) else config.get("tile_stride", None)
    stride: Tuple[int, int] | None = None
    if isinstance(stride_cfg, (list, tuple)):
        stride = (int(stride_cfg[0]), int(stride_cfg[1]))
    add_mag = bool(config.get("add_magnitude", True))
    normalize = bool(config.get("normalize", True))
    limit_tiles = config.get("limit_tiles", None)

    try:
        tile_batch = load_vector_field_tiles(
            path=field_path,
            tile_size=(int(tile_size[0]), int(tile_size[1])),
            stride=stride,
            add_magnitude=add_mag,
            normalize=normalize,
            limit_tiles=None if limit_tiles is None else int(limit_tiles),
        )
    except Exception as e:
        msg = str(e)
        # Fallback: if no tiles produced (field smaller than configured tile), auto-adjust to safe tiling
        if "No tiles produced" in msg or "check tile_size/stride" in msg:
            try:
                field_arr = load_vector_field(field_path, mmap=True)
                chw = ensure_chw(field_arr)
                H, W = int(chw.shape[1]), int(chw.shape[2])
                safe_th = max(1, min(int(tile_size[0]), H))
                safe_tw = max(1, min(int(tile_size[1]), W))
                if isinstance(stride, tuple):
                    sh, sw = int(stride[0]), int(stride[1])
                else:
                    sh, sw = int(tile_size[0]), int(tile_size[1])
                safe_sh = max(1, min(sh, safe_th))
                safe_sw = max(1, min(sw, safe_tw))
                c.warn(
                    f"No tiles with configured tile_size/stride; retrying with safe tile_size=({safe_th},{safe_tw}) stride=({safe_sh},{safe_sw}) for this file"
                )
                tile_batch = load_vector_field_tiles(
                    path=field_path,
                    tile_size=(safe_th, safe_tw),
                    stride=(safe_sh, safe_sw),
                    add_magnitude=add_mag,
                    normalize=normalize,
                    limit_tiles=None if limit_tiles is None else int(limit_tiles),
                )
            except Exception as e2:
                raise e2
        else:
            raise
    c.info(f"Loaded tiles: shape={tile_batch.shape}")

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(tile_batch).to(device)
        logits = model(x)
        # Support both single-logit binary and multi-class heads
        if logits.ndim == 2 and logits.shape[1] == 1:
            p1 = torch.sigmoid(logits.squeeze(1))  # (N,)
            # Synthesize two-class probabilities [P(class0), P(class1)]
            probs = torch.stack([1.0 - p1, p1], dim=1)  # (N, 2)
        else:
            probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    # Summaries
    c.info(f"Logits shape: {tuple(logits.shape)} | num_classes={probs.shape[1]}")
    topk = min(5, logits.shape[0])
    c.debug("Sample predictions (first N tiles):")
    for i in range(topk):
        c.debug(f"  tile[{i:03d}]: class={pred[i].item()} prob={conf[i].item():.4f}")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Save CSV with tile_index, predicted_class, confidence
        import csv

        with output.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["tile_index", "predicted_class", "confidence"])
            for i in range(pred.shape[0]):
                w.writerow([i, int(pred[i].item()), float(conf[i].item())])
        c.success(f"Saved predictions to: {output}")

    # Optional: save full probabilities/logits per tile
    if probs_out is not None:
        probs_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(probs_out.with_suffix(".npy")), probs.cpu().numpy())
        c.success(f"Saved probabilities (N,C) to: {probs_out.with_suffix('.npy')}")
    if logits_out is not None:
        logits_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(logits_out.with_suffix(".npy")), logits.cpu().numpy())
        c.success(f"Saved logits (N,C) to: {logits_out.with_suffix('.npy')}")

    # Optional: heatmaps per tile and stitched heatmap via HeatmapGenerator
    if heatmaps_dir is not None or stitch_heatmap_path is not None or show_heatmap:
        probs_cpu = probs.cpu().numpy()
        if heatmap_class is not None:
            hc = int(heatmap_class)
            if hc < 0 or hc >= probs_cpu.shape[1]:
                c.warn(f"heatmap_class {hc} out of range; using predicted class conf")
                vals = conf.cpu().numpy()
            else:
                vals = probs_cpu[:, hc]
        else:
            vals = conf.cpu().numpy()

        hm = HeatmapGenerator(cmap="viridis")

        if heatmaps_dir is not None:
            # Allow configurable per-tile heatmap image size via config (default 16)
            tile_img_sz = 16
            try:
                tile_img_sz = int(config.get("heatmap_tile_size", 16))
            except Exception:
                tile_img_sz = 16
            hm.save_per_tile(vals, out_dir=heatmaps_dir, tile_img_size=tile_img_sz)
            c.success(f"Saved per-tile heatmaps to: {heatmaps_dir}")

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
                    c.warn(f"Cannot show heatmap: grid ({n_rows}x{n_cols}) doesn't match N={vals.shape[0]}")
            # (Removed duplicate unconditional heatmap display block)

    # File-level summary prediction: average probabilities across tiles
    try:
        probs_mean = probs.mean(dim=0).cpu().numpy()
        file_pred_idx = int(np.argmax(probs_mean))
        file_conf = float(probs_mean[file_pred_idx])
        # Choose class names from provided list or from config
        names = None
        if class_names and len(class_names) == int(probs.shape[1]):
            names = class_names
        else:
            cfg_names = config.get("class_names", None)
            if isinstance(cfg_names, list) and len(cfg_names) == int(probs.shape[1]):
                names = [str(s) for s in cfg_names]
        pred_name = names[file_pred_idx] if names is not None else f"class{file_pred_idx}"
        fname = Path(field_path).name
        # Append to in-memory report lists for terminal tables
        try:
            if report_list is not None:
                report_list.append((fname, pred_name, int(file_pred_idx), float(round(file_conf * 100.0, 2))))
            if report_probs_list is not None:
                # Store mean probs as a Python list for table printing
                report_probs_list.append((fname, [float(x) for x in probs_mean.tolist()]))
        except Exception:
            pass

        if file_summary_out is not None:
            try:
                file_summary_out.parent.mkdir(parents=True, exist_ok=True)
                import csv
                write_header = True
                try:
                    if file_summary_out.exists():
                        # If file exists but is empty, still write header
                        write_header = file_summary_out.stat().st_size == 0
                    else:
                        write_header = True
                except Exception:
                    write_header = True
                with file_summary_out.open("a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(["file_name", "predicted_class", "predicted_index", "confidence_percent"]) 
                    w.writerow([fname, pred_name, file_pred_idx, round(file_conf * 100.0, 2)])
                c.success(f"Appended file summary for {fname} -> {pred_name} ({file_conf*100.0:.2f}%) to {file_summary_out}")
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
                    c.info(f"Summary file locked ('{file_summary_out}'); wrote to '{alt}' instead.")
                except Exception as e2:
                    c.warn(f"Failed to write file summary (fallback also failed): {e2}")
        else:
            c.info(f"File-level prediction for {fname}: {pred_name} ({file_conf*100.0:.2f}%)")
    except Exception as e:
        c.warn(f"Failed to compute file-level summary: {e}")

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
            c.warn(f"Failed to load labels_csv '{labels_csv}': {e}")
    if y_true is None and label_all is not None:
        y_true = np.full((tile_batch.shape[0],), int(label_all), dtype=np.int64)
    if y_true is None:
        cfg_csv = config.get("labels_csv", None)
        if isinstance(cfg_csv, str) and cfg_csv.strip():
            try:
                y_true = load_labels_from_csv(Path(cfg_csv))
            except Exception as e:
                c.warn(f"Failed to load config.labels_csv '{cfg_csv}': {e}")
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
                    c.warn(f"Embedded labels length {y_emb.shape[0]} doesn't match tiles {tile_batch.shape[0]}; skipping embedded labels")
        except Exception as e:
            c.warn(f"Failed to derive embedded labels from CSV: {e}")

    # If still missing labels and CM is requested, optionally auto-dump a labels template CSV
    if (cm_csv_out is not None or cm_png_out is not None) and y_true is None:
        auto_dump = bool(config.get("auto_dump_labels_template", True))
        if auto_dump:
            try:
                stem = Path(field_path).stem
                base_dir = cm_csv_out.parent if cm_csv_out is not None else (cm_png_out.parent if cm_png_out is not None else Path("."))
                tmpl_path = base_dir / f"labels_template-{stem}.csv"
                write_labels_template(tmpl_path, int(pred.shape[0]), pred.cpu().numpy())
                c.info(f"No labels provided; wrote labels template to: {tmpl_path}. Fill the 'label' column and pass via --labels-csv.")
            except Exception as e:
                c.warn(f"Failed to write labels template CSV: {e}")

    if (cm_csv_out is not None or cm_png_out is not None) and y_true is not None:
        cm_debug = bool(config.get("cm_debug", False))
        y_pred = pred.cpu().numpy().astype(np.int64)
        n = tile_batch.shape[0]
        if y_true.shape[0] != n:
            c.warn(f"Labels length {y_true.shape[0]} does not match tiles {n}; skipping CM")
        else:
            # Mask out unlabeled entries (-1 or <0)
            if cm_debug:
                try:
                    uniq, cnt = np.unique(y_true, return_counts=True)
                    c.debug(f"[cm_debug] y_true unique raw: {list(zip(uniq.tolist(), cnt.tolist()))}")
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
                c.debug(f"[cm_debug] kept after mask: {int(y_true_masked.size)} of {int(n)}")
            if y_true_masked.size == 0:
                c.warn("No valid ground-truth labels available for confusion matrix; skipping")
                cm = np.zeros((num_classes, num_classes), dtype=np.int64)
            else:
                idx = (y_true_masked * num_classes + y_pred_masked).astype(np.int64)
                cm = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

            # Save CSV/PNG/TXT via common helper
            save_confusion_outputs(cm, class_names, cm_csv_out, cm_png_out, console=c)

            # Print confusion matrix as a console table (counts)
            try:
                names = class_names if isinstance(class_names, list) and class_names else [str(i) for i in range(int(cm.shape[0]))]
                headers = ["true\\pred"] + names
                rows = []
                for i, tn in enumerate(names):
                    rows.append([tn] + [str(int(cm[i, j])) for j in range(len(names))])
                c.table(headers, rows, align=["l"] + ["r"] * len(names), title="Confusion Matrix (counts)")
            except Exception:
                pass

            # Print simple tile-level accuracy if any labels present
            total = int(cm.sum())
            correct = int(np.trace(cm)) if cm.size > 0 else 0
            if total > 0:
                acc = correct / total
                c.info(f"Tile accuracy: {acc*100.0:.2f}% ({correct}/{total})")

    return cm


def _load_single_image(path: Path, *, image_size: Tuple[int, int], channels: int, mean: list[float], std: list[float], use_cv: bool = False) -> torch.Tensor:
    # Lightweight duplicate of image_data loader to avoid heavy imports
    if use_cv:
        try:
            import cv2  # type: ignore
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError("cv2.imread returned None")
            if channels == 1:
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (int(image_size[1]), int(image_size[0])), interpolation=cv2.INTER_AREA)
            arr = img
            if channels == 1:
                arr = np.expand_dims(arr, axis=0)
            else:
                arr = np.transpose(arr, (2, 0, 1))
            arr = arr.astype(np.float32) / 255.0
        except Exception:
            use_cv = False
    if not use_cv:
        try:
            from PIL import Image  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Pillow (PIL) is required for image inference. Install 'Pillow'.") from e
        with Image.open(path) as img:
            if channels == 1:
                img = img.convert("L")
            else:
                img = img.convert("RGB")
            img = img.resize((int(image_size[1]), int(image_size[0])), resample=Image.BILINEAR)
            arr = np.array(img)
            if channels == 1:
                arr = np.expand_dims(arr, axis=0)
            else:
                arr = np.transpose(arr, (2, 0, 1))
            arr = arr.astype(np.float32) / 255.0
        # normalize
        c = arr.shape[0]
        # Validate lengths; allow single-value broadcast or exact match
        if len(mean) not in (1, c) or len(std) not in (1, c):
            raise RuntimeError("image mean/std length must be 1 or match channels")
        if len(mean) == 1:
            mean = [float(mean[0])] * c
        if len(std) == 1:
            std = [float(std[0])] * c
        for i in range(c):
            arr[i] = (arr[i] - float(mean[i])) / (float(std[i]) + 1e-6)
        return torch.from_numpy(arr.astype(np.float32))


def run_image_inference(
    *,
    config: Dict[str, Any],
    device: torch.device,
    model: torch.nn.Module,
    image_root: str,
    base_outputs: Path | None = None,
    file_summary_out: Path | None = None,
    cm_csv_out: Path | None = None,
    cm_png_out: Path | None = None,
    include_classes: "list[str] | None" = None,
    out_subdir: str | None = None,
) -> None:
    c = console_from_config(config)
    task = str(config.get("task", "")).lower()
    multitask = task.startswith("multi") or bool(config.get("multitask", False))
    # Allow pointer files (.txt/.csv) containing the actual image root path
    if _resolve_root_path_img is not None:
        root = _resolve_root_path_img(image_root)
    else:
        root = Path(image_root)
    if not root.exists():
        c.warn(f"Image root not found: {root}; skipping inference for this path.")
        return
    size = tuple(config.get("image_size", [256, 256]))  # type: ignore[assignment]
    channels = int(config.get("image_channels", 3))
    mean = list(config.get("image_mean", [0.5]))
    std = list(config.get("image_std", [0.5]))
    class_names = config.get("class_names", None)
    # Build mapping from subfolders if not provided
    if _build_class_index is not None:
        try:
            mapping = _build_class_index(root, class_names)
        except Exception:
            mapping = None
    else:
        mapping = None
    idx_to_name = None
    if mapping:
        idx_to_name = {v: k for k, v in mapping.items()}

    # Collect images under subfolders
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    all_files: list[Path]
    # Optimization: when include_classes is provided, scan only those class folders directly under root
    if include_classes:
        try:
            allow_set = {str(x).lower() for x in include_classes}
            all_files = []
            # Show a transient scan bar while enumerating many files
            try:
                from tqdm import tqdm as _tqdm  # type: ignore
                scan_bar = _tqdm(desc="scan", unit="file", dynamic_ncols=True, ascii=True, leave=False)
            except Exception:
                scan_bar = None
            for cls in sorted(allow_set):
                cls_dir = (root / cls)
                if not (cls_dir.exists() and cls_dir.is_dir()):
                    continue
                try:
                    for p in cls_dir.rglob("*"):
                        if p.is_file() and p.suffix.lower() in exts:
                            all_files.append(p)
                            if scan_bar is not None:
                                try:
                                    scan_bar.update(1)
                                except Exception:
                                    pass
                except Exception:
                    continue
            try:
                if scan_bar is not None:
                    scan_bar.close()
            except Exception:
                pass
            try:
                c.info(f"Restricted scan to classes: {', '.join(sorted(allow_set))} (files found={len(all_files)})")
            except Exception:
                pass
        except Exception:
            all_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        all_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    def _nearest_class_name(p: Path) -> str | None:
        try:
            for ancestor in [p.parent] + list(p.parents):
                if ancestor.resolve() == root.resolve():
                    break
                nm = ancestor.name
                if mapping and nm in mapping:
                    return nm
        except Exception:
            return None
        return None
    if include_classes is not None and len(include_classes) > 0:
        allow = {str(x).lower() for x in include_classes}
        files = []
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
            filt_bar = _tqdm(
                total=len(all_files),
                desc="filter",
                unit="img",
                dynamic_ncols=True,
                ascii=True,
                leave=False,
            )
        except Exception:
            filt_bar = None
        step = max(1, max(100, len(all_files) // 200))
        last_print = 0.0
        try:
            for idx, p in enumerate(all_files, start=1):
                # Derive nearest class using include list (ignore mapping)
                nm = None
                try:
                    for ancestor in [p.parent] + list(p.parents):
                        if ancestor.resolve() == root.resolve():
                            break
                        name_l = ancestor.name.lower()
                        if name_l in allow:
                            nm = name_l
                            break
                except Exception:
                    nm = None
                if nm is not None:
                    files.append(p)
                if filt_bar is not None:
                    try:
                        filt_bar.update(1)
                    except Exception:
                        pass
                else:
                    now = time.time()
                    if (idx % step == 0) or (now - last_print >= 0.5) or idx == len(all_files):
                        pct = (idx / max(1, len(all_files))) * 100.0
                        c.info(f"[filter] scanned {idx}/{len(all_files)} ({pct:.1f}%) kept={len(files)}")
                        last_print = now
        finally:
            try:
                if filt_bar is not None:
                    filt_bar.close()
            except Exception:
                pass
    else:
        files = all_files
    if not files:
        c.warn(f"No images found under {root}; nothing to infer")
        return
    # Pre-run sanity: show a quick scan summary and a few sample files
    try:
        c.header("Scan")
        c.info(f"Files queued: {len(files)}")
        if mapping:
            # List a few class names in discovery order
            try:
                some = ", ".join(list(mapping.keys())[:6])
                more = " ..." if len(mapping) > 6 else ""
                c.info(f"Classes: {len(mapping)} -> {some}{more}")
            except Exception:
                pass
        # Show a few samples with nearest class name
        try:
            n_show = min(5, len(files))
            for p in files[:n_show]:
                nm = _nearest_class_name(p)
                c.info(f"sample: {p.name}  class={nm if nm is not None else '<unknown>'}")
        except Exception:
            pass
    except Exception:
        pass
    model.eval()
    results: list[tuple[str, str, int, float, float | None, float | None]] = []
    # Batched inference for large folders
    try:
        bs = int(config.get("image_infer_batch_size", 64))
    except Exception:
        bs = 64
    # Read perf options (accept under top-level or under 'inference')
    def _opt(key: str, default=None):
        inf = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}
        return config.get(key, inf.get(key, default))
    mixed_precision = str(_opt("mixed_precision", "off")).lower()
    use_channels_last = bool(_opt("channels_last", False))
    prefetch_workers = int(_opt("prefetch_workers", 0) or 0)
    use_cv = bool(_opt("use_opencv_loader", False))
    use_pin_memory = bool(_opt("pin_memory", False))

    # Loader availability probe: ensure at least one loader works
    loader_ok = False
    if use_cv:
        try:
            import cv2  # type: ignore
            loader_ok = True
        except Exception:
            use_cv = False
    if not loader_ok:
        try:
            from PIL import Image  # type: ignore  # noqa: F401
            loader_ok = True
        except Exception:
            loader_ok = False
    if not loader_ok:
        raise RuntimeError(
            "No image loader available. Install Pillow ('pip install Pillow') or enable use_opencv_loader with OpenCV ('pip install opencv-python')."
        )
    batch_paths: list[Path] = []
    batch_tensors: list[torch.Tensor] = []

    # Progress setup
    total_files = len(files)
    processed = 0
    last_print = 0.0
    # Optional tqdm progress bar (falls back to periodic logs)
    try:
        from tqdm import tqdm  # type: ignore
        # Use conservative redraw settings for Windows consoles; avoid wide glyphs and fit to terminal width
        bar = tqdm(total=total_files, desc="images", unit="img", dynamic_ncols=True, ascii=True, mininterval=0.1, miniters=1, smoothing=0.0, leave=True)
    except Exception:
        bar = None
    # Print an initial line only when tqdm is not available
    if bar is None:
        try:
            c.info(f"Progress: 0/{total_files} (0.0%)")
        except Exception:
            pass
    # Also emit every 'step' images (~200 updates)
    step = max(1, max(100, total_files // 200))

    def _flush_batch() -> None:
        if not batch_tensors:
            return
        nonlocal processed, last_print
        xb = torch.stack(batch_tensors, dim=0)
        if use_channels_last:
            try:
                xb = xb.contiguous(memory_format=torch.channels_last)
            except Exception:
                pass
        if device.type == "cuda" and use_pin_memory:
            try:
                xb = xb.pin_memory()
            except Exception:
                pass
        xb = xb.to(device, non_blocking=(device.type == "cuda"))
        use_inf_mode = bool(_opt("use_inference_mode", True))
        # Forward under inference_mode (slightly faster than no_grad)
        ctx = torch.inference_mode if use_inf_mode else torch.no_grad
        with ctx():
            if device.type == "cuda" and mixed_precision in ("fp16", "bf16"):
                try:
                    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
                    with torch.amp.autocast(device_type="cuda", dtype=dtype):
                        out = model(xb)
                except Exception:
                    out = model(xb)
            else:
                out = model(xb)
            if isinstance(out, dict):
                logits = out.get("logits", None)
                reg = out.get("reg", None)
            else:
                logits = out
                reg = None
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            reg_np = reg.detach().cpu().numpy() if reg is not None else None
        for i, pth in enumerate(batch_paths):
            prob_i = probs[i]
            pred_idx = int(np.argmax(prob_i))
            conf = float(prob_i[pred_idx])
            pred_name = idx_to_name.get(pred_idx, str(pred_idx)) if idx_to_name is not None else str(pred_idx)
            vel = float(reg_np[i, 0]) if (reg_np is not None and reg_np.shape[1] >= 1) else None
            ren = float(reg_np[i, 1]) if (reg_np is not None and reg_np.shape[1] >= 2) else None
            try:
                rel_path = str(pth.relative_to(root))
            except Exception:
                rel_path = pth.name
            results.append((rel_path, pred_name, pred_idx, conf, vel, ren))
        batch_paths.clear()
        batch_tensors.clear()
        # No progress update here; we update per image as they are decoded/queued

    if prefetch_workers > 0:
        # Threaded per-batch image decode/resize
        try:
            from concurrent.futures import ThreadPoolExecutor
            err_counter = {"n": 0}
            def _load(pth: Path) -> tuple[Path, torch.Tensor | None]:
                try:
                    x = _load_single_image(pth, image_size=(int(size[0]), int(size[1])), channels=channels, mean=mean, std=std, use_cv=use_cv)
                    return (pth, x)
                except Exception as e:
                    # Log a few early failures to help diagnose empty outputs
                    try:
                        if err_counter["n"] < 5:
                            err_counter["n"] += 1
                            c.warn(f"decode failed: {pth.name} -> {e}")
                    except Exception:
                        pass
                    return (pth, None)
            for i in range(0, len(files), bs):
                chunk = files[i : i + bs]
                batch_paths = []
                batch_tensors = []
                with ThreadPoolExecutor(max_workers=max(1, prefetch_workers)) as ex:
                    futs = [ex.submit(_load, p) for p in chunk]
                    for fu in futs:
                        pth, x = fu.result()
                        if x is None:
                            continue
                        batch_paths.append(pth)
                        batch_tensors.append(x)
                        # Per-image progress update
                        processed += 1
                        if bar is not None:
                            try:
                                bar.update(1)
                                # Occasionally force a redraw to keep the bar live even if other logs appear
                                if processed % step == 0:
                                    bar.refresh()
                            except Exception:
                                pass
                        now = time.time()
                        if bar is None and (now - last_print >= 0.5 or (processed % step == 0) or processed >= total_files):
                            pct = (processed / max(1, total_files)) * 100.0
                            c.info(f"Progress: {processed}/{total_files} ({pct:.1f}%)")
                            last_print = now
                _flush_batch()
        except Exception:
            # Fallback to sequential
            for p in files:
                try:
                    x = _load_single_image(p, image_size=(int(size[0]), int(size[1])), channels=channels, mean=mean, std=std, use_cv=use_cv)
                    batch_paths.append(p)
                    batch_tensors.append(x)
                    if len(batch_tensors) >= max(1, bs):
                        _flush_batch()
                except Exception:
                    continue
                # Per-image progress update (sequential)
                processed += 1
                if bar is not None:
                    try:
                        bar.update(1)
                        if processed % step == 0:
                            bar.refresh()
                    except Exception:
                        pass
                now = time.time()
                if bar is None and (now - last_print >= 0.5 or (processed % step == 0) or processed >= total_files):
                    pct = (processed / max(1, total_files)) * 100.0
                    c.info(f"Progress: {processed}/{total_files} ({pct:.1f}%)")
                    last_print = now
            _flush_batch()
    else:
        for p in files:
            try:
                x = _load_single_image(p, image_size=(int(size[0]), int(size[1])), channels=channels, mean=mean, std=std, use_cv=use_cv)
                batch_paths.append(p)
                batch_tensors.append(x)
                if len(batch_tensors) >= max(1, bs):
                    _flush_batch()
            except Exception:
                # Skip unreadable files but continue
                continue
            # Per-image progress update (sequential, no prefetch)
            processed += 1
            if bar is not None:
                try:
                    bar.update(1)
                    if processed % step == 0:
                        bar.refresh()
                except Exception:
                    pass
            now = time.time()
            if bar is None and (now - last_print >= 0.5 or (processed % step == 0) or processed >= total_files):
                pct = (processed / max(1, total_files)) * 100.0
                c.info(f"Progress: {processed}/{total_files} ({pct:.1f}%)")
                last_print = now
        _flush_batch()
    # Final progress line at 100% (only when tqdm is not used)
    if bar is None:
        try:
            pct = (processed / max(1, total_files)) * 100.0
            if processed >= total_files:
                c.info(f"Progress: {processed}/{total_files} ({pct:.1f}%)")
        except Exception:
            pass
    # Close bar if used
    try:
        if bar is not None:
            bar.close()
    except Exception:
        pass
    # Decide default image output path under outputs_root/image when not provided
    if file_summary_out is None:
        try:
            if base_outputs is not None:
                sub = out_subdir if isinstance(out_subdir, str) and out_subdir.strip() else "image"
                file_summary_out = (base_outputs / sub / "file_predictions.csv").resolve()
            else:
                # Fallback to ./data/outputs/image
                sub = out_subdir if isinstance(out_subdir, str) and out_subdir.strip() else "image"
                file_summary_out = (Path(".") / "data" / "outputs" / sub / "file_predictions.csv").resolve()
        except Exception:
            file_summary_out = None

    # Write summary CSV with required columns only
    if file_summary_out is not None:
        try:
            file_summary_out.parent.mkdir(parents=True, exist_ok=True)
            import csv

            def _write_csv(path: Path) -> int:
                with path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if multitask:
                        w.writerow(["image_path", "predicted_class", "identifier", "confidence_pct", "velocity_pred", "reynolds_pred"])
                        for (rel_path, pred_name, pred_idx, conf, vel, ren) in results:
                            w.writerow([rel_path, pred_name, int(pred_idx), round(conf * 100.0, 2), vel, ren])
                    else:
                        w.writerow(["image_path", "predicted_class", "identifier", "confidence_pct"])
                        for (rel_path, pred_name, pred_idx, conf, _vel, _ren) in results:
                            w.writerow([rel_path, pred_name, int(pred_idx), round(conf * 100.0, 2)])
                return len(results)

            try:
                nrows = _write_csv(file_summary_out)
                c.info(f"Wrote image predictions: {file_summary_out} (rows={nrows})")
            except Exception as e1:
                # Fallback: write to a timestamped file in same directory
                try:
                    from datetime import datetime as _dt
                    ts = _dt.now().strftime("%Y%m%d-%H%M%S")
                    alt = file_summary_out.with_name(f"{file_summary_out.stem}-{ts}{file_summary_out.suffix}")
                    _write_csv(alt)
                    c.warn(f"Summary file locked ('{file_summary_out}'); wrote to '{alt}' instead.")
                except Exception as e2:
                    c.warn(f"Failed to write file summary (fallback also failed): {e2}")
        except Exception as e:
            c.warn(f"Failed to prepare summary output directory: {e}")


class InferenceRunner:
    """
    Top-level, class-based inference runner for prototype two.
    Uses common helpers for config, paths, and saving artifacts.
    """

    def __init__(self, cfg_path: Path, device_pref: str = "auto") -> None:
        cfg_mgr = ConfigManager(cfg_path)
        self.cfg_path = cfg_mgr.path
        self.config = cfg_mgr.config
        try:
            ensure_outputs_ready(self.cfg_path, self.config)
        except Exception:
            pass
        self.device = DeviceSelector.choose(device_pref)
        self.console = console_from_config(self.config)
        self.console.info(f"Loaded config from: {self.cfg_path}")
        self.console.info(f"Using device: {self.device}")

    def _is_image_mode(self) -> bool:
        # Vector path removed; force image mode
        return True

    def _resolve_weights_path(self, weights: "str | None", cfg: Dict[str, Any]) -> Path:
        candidates: list[str] = []
        if isinstance(weights, str) and weights.strip():
            candidates.append(weights)
        for k in ("weights", "weights_path", "save_weights"):
            v = cfg.get(k, None)
            if isinstance(v, str) and v.strip():
                candidates.append(v)
        for w in candidates:
            p = Path(w)
            if not p.is_absolute():
                p = (self.cfg_path.parent / p).resolve()
            if p.exists():
                return p
        search_dirs: list[Path] = []
        cfg_dir = self.cfg_path.parent
        search_dirs.append((cfg_dir / "runs").resolve())
        out_root = cfg.get("outputs_root", None)
        if isinstance(out_root, str) and out_root.strip():
            p2 = Path(out_root)
            search_dirs.append(p2 if p2.is_absolute() else (cfg_dir / p2).resolve())
        latest: Path | None = None
        for d in search_dirs:
            if not d.exists() or not d.is_dir():
                continue
            for pat in ("model-*.pth", "*.pth", "*.pt"):
                for f in d.rglob(pat):
                    if latest is None or f.stat().st_mtime > latest.stat().st_mtime:
                        latest = f
        if latest is not None:
            self.console.warn(f"[weights] None specified; using latest found: {latest}")
            return latest
        tried = ", ".join(candidates) or "<none>"
        raise FileNotFoundError(
            f"Weights not found. Pass --weights, set weights/weights_path/save_weights in config, or place a .pth/.pt in runs/. Tried: {tried}"
        )

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
        cfg = self.config
        # Branch: image-mode inference uses directory of images instead of vector fields
        is_image = self._is_image_mode()

        if is_image:
            # Resolve test_image_dir (or allow explicit CLI override)
            override_dir = field_path if isinstance(field_path, str) and field_path.strip() else None
            img_dir = override_dir if override_dir is not None else cfg.get("test_image_dir", None)
            if not isinstance(img_dir, str) or not img_dir.strip():
                raise RuntimeError("For image inference set test_image_dir in the config or pass --field_path to point to the test image directory.")
            # Anchor and auto-descend into 'image' subfolder when a container path is provided
            resolver = PathResolver(
                self.cfg_path,
                outputs_root=str(self.config.get("outputs_root")) if isinstance(self.config.get("outputs_root"), str) else None,
            )
            p_img = resolver.anchor(img_dir)
            # If the anchored path does not exist, try parent/image/<basename> as a smart fallback
            if p_img is not None and not p_img.exists():
                try:
                    parent = p_img.parent
                    alt = (parent / "image" / p_img.name).resolve()
                    if alt.exists() and alt.is_dir():
                        self.console.info(f"Resolved missing path via image subfolder: {alt}")
                        p_img = alt
                except Exception:
                    pass
            if p_img is not None and p_img.exists() and p_img.is_dir():
                candidate = p_img / "image"
                try:
                    if candidate.exists() and candidate.is_dir():
                        img_dir = str(candidate.resolve())
                        self.console.info(f"Auto-descended into image subfolder: {img_dir}")
                    else:
                        img_dir = str(p_img)
                except Exception:
                    img_dir = str(p_img)
            # Final sanity print of resolved image root
            try:
                # Inputs section header for image mode
                self.console.header("Inference Inputs")
                self.console.info("Mode: image")
                self.console.info(f"Root: {img_dir}")
            except Exception:
                pass
            # Build model
            inferred_in_ch = int(cfg.get("image_channels", 3))
            model = build_model(cfg, inferred_in_ch)
            # Resolve weights via shared helper and load
            resolved = self._resolve_weights_path(weights, cfg)
            load_weights(model, resolved, self.device)
            # Optional compile and channels-last for speed
            def _opt(key: str, default=None):
                inf = self.config.get("inference", {}) if isinstance(self.config.get("inference"), dict) else {}
                return self.config.get(key, inf.get(key, default))
            want_compile = bool(_opt("torch_compile", False))
            use_channels_last = bool(_opt("channels_last", False))
            if want_compile:
                try:
                    model = torch.compile(model)  # type: ignore[attr-defined]
                except Exception:
                    pass
            if use_channels_last:
                try:
                    model = model.to(memory_format=torch.channels_last)
                except Exception:
                    pass
            model.to(self.device)

            # Model/Weights summary (image mode)
            try:
                self.console.header("Model / Weights")
                rows = [
                    ["device", str(self.device)],
                    ["weights", str(resolved)],
                    ["input_channels", str(inferred_in_ch)],
                    ["num_classes", str(getattr(model, "num_classes", "?"))],
                ]
                self.console.table(["key", "value"], rows, align=["l", "l"], title="Model Info")
            except Exception:
                pass
            # Resolve outputs from CLI or fallback to config, and anchor
            if not (isinstance(file_summary, str) and file_summary.strip()):
                file_summary = cfg.get("file_summary", None)
            if not (isinstance(cm_csv, str) and cm_csv.strip()):
                cm_csv = cfg.get("cm_csv", None)
            if not (isinstance(cm_png, str) and cm_png.strip()):
                cm_png = cfg.get("cm_png", None)
            sum_path = resolver.anchor(file_summary) if isinstance(file_summary, str) and file_summary.strip() else None
            cmc_path = resolver.anchor(cm_csv) if isinstance(cm_csv, str) and cm_csv.strip() else None
            cmp_path = resolver.anchor(cm_png) if isinstance(cm_png, str) and cm_png.strip() else None

            # Outputs summary (image mode)
            try:
                self.console.header("Outputs")
                rows: list[list[str]] = [["outputs_root", str(resolver.outputs_root)]]
                if sum_path is not None:
                    rows.append(["file_summary", str(sum_path)])
                if cmc_path is not None:
                    rows.append(["cm_csv", str(cmc_path)])
                if cmp_path is not None:
                    rows.append(["cm_png", str(cmp_path)])
                self.console.table(["key", "path"], rows, align=["l", "l"], title="Requested Outputs")
            except Exception:
                pass
            # Optional filter: include only specific class folders (e.g., ["laminar","turbulent"]) from config
            try:
                inf_sc = cfg.get("inference", {}) if isinstance(cfg.get("inference"), dict) else {}
                include_classes_cfg = cfg.get("image_include_classes", None)
                if not isinstance(include_classes_cfg, list) or not include_classes_cfg:
                    include_classes_cfg = inf_sc.get("image_include_classes", None)
                if not isinstance(include_classes_cfg, list) or not include_classes_cfg:
                    include_classes_cfg = None
            except Exception:
                include_classes_cfg = None

            run_image_inference(
                config=cfg,
                device=self.device,
                model=model,
                image_root=str(img_dir),
                base_outputs=resolver.outputs_root,
                file_summary_out=sum_path,
                cm_csv_out=cmc_path,
                cm_png_out=cmp_path,
                include_classes=include_classes_cfg,
            )
            # Final summary (image mode)
            try:
                self.console.header("Done")
                rows: list[list[str]] = [["outputs_root", str(resolver.outputs_root)]]
                if sum_path is not None:
                    rows.append(["file_summary", str(sum_path)])
                if cmc_path is not None:
                    rows.append(["cm_csv", str(cmc_path)])
                if cmp_path is not None:
                    rows.append(["cm_png", str(cmp_path)])
                self.console.table(["key", "path"], rows, align=["l", "l"], title="Artifacts")
            except Exception:
                pass
            return

        fields = choose_fields(cfg, field_path)
        if not fields:
            raise RuntimeError("field_path or test/train_field_path must be provided via CLI or config")

        resolver = PathResolver(self.cfg_path, outputs_root=str(cfg.get("outputs_root")) if isinstance(cfg.get("outputs_root"), str) else None)
        # Auto-enable recursion if config requests it or any input is a directory-like path
        cfg_recurse = bool(cfg.get("recurse", False))
        looks_like_dir = False
        try:
            for raw in fields:
                p = Path(raw)
                if (not p.suffix) or any(ch in str(p) for ch in ["*", "?", "["]):
                    looks_like_dir = True
                    break
        except Exception:
            looks_like_dir = False
        effective_recurse = bool(recurse or cfg_recurse or looks_like_dir)
        anchored_fields = resolver.expand_fields(fields, recurse=effective_recurse)

        # Prefer actual vector field inputs for inference; skip mapping CSVs like file_name,label
        def _is_vector_candidate(path_str: str) -> bool:
            try:
                q = Path(path_str)
                if not q.is_file():
                    return False
                ext = q.suffix.lower()
                if ext in (".npy", ".npz"):
                    return True
                if ext == ".csv":
                    with q.open("r", encoding="utf-8", errors="ignore") as f:
                        first = f.readline().lower()
                    if ("file_name" in first and "label" in first):
                        return False
                    return any(k in first for k in ("x", "y", "delta", "dx", "dy"))
                return False
            except Exception:
                return False

        vector_fields = [s for s in anchored_fields if _is_vector_candidate(s)]
        if vector_fields:
            use_field = vector_fields[0]
        else:
            use_field = anchored_fields[0]
            self.console.warn(f"No obvious vector-field files found under inputs; using first match: {use_field}")

        # Inputs section header for vector mode
        try:
            self.console.header("Inference Inputs")
            self.console.info("Mode: vector")
            self.console.info(f"Requested inputs: {len(fields)}")
            self.console.info(f"Resolved files: {len(anchored_fields)}")
            max_show = 12
            show = anchored_fields[:max_show]
            rows = [[f"{i}", s] for i, s in enumerate(show)]
            if rows:
                self.console.table(["idx", "path"], rows, align=["r", "l"], title="Files To Process")
            if len(anchored_fields) > max_show:
                self.console.info(f"... and {len(anchored_fields)-max_show} more")
        except Exception:
            pass

        # Resolve weights (vector path)
        resolved_w = self._resolve_weights_path(weights, cfg)
        weights_path: str = str(resolved_w)
        self.console.info(f"Using weights: {weights_path}")

        # Build model
        try:
            in_ch = infer_input_channels_from_field(cfg, use_field)
        except Exception as e:
            raise RuntimeError(f"Failed to infer input channels from field: {e}") from e
        model = build_model(cfg, inferred_in_ch=in_ch).to(self.device)
        load_weights(model, Path(weights_path), self.device)
        self.console.info("Model and weights loaded.")

        # Options and outputs (heatmaps/CM are deferred to post-processing)
        gen_hm = False
        gen_cm = False
        output = output or cfg.get("output", None)
        probs_out = probs_out or cfg.get("probs_out", None)
        logits_out = logits_out or cfg.get("logits_out", None)
        heatmaps_dir = None
        stitch_heatmap = None
        cm_csv = None
        cm_png = None
        show_heatmap = False

        out_path = resolver.anchor(output) if output else None
        probs_path = resolver.anchor(probs_out) if probs_out else None
        logits_path = resolver.anchor(logits_out) if logits_out else None
        heat_dir, stitch_path = None, None
        cm_csv_path, cm_png_path = None, None

        # File labels manifest
        labels_map: dict[str, int] | None = None
        use_manifest = file_labels if (isinstance(file_labels, str) and file_labels.strip()) else cfg.get("file_labels", cfg.get("file_labels_manifest", None))
        if isinstance(use_manifest, str) and use_manifest.strip():
            p = Path(use_manifest)
            if not p.is_absolute():
                p = (self.cfg_path.parent / p).resolve()
            if p.exists():
                try:
                    import csv
                    labels_map = {}
                    with p.open("r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                    start = 0
                    if rows and len(rows[0]) >= 2 and not str(rows[0][1]).strip().isdigit():
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
                            labels_map[name.lower()] = lab
                            labels_map[Path(name).name.lower()] = lab
                            labels_map[Path(name).stem.lower()] = lab
                    self.console.info(f"Loaded file labels manifest: {p}")
                except Exception as e:
                    self.console.warn(f"Failed to load file labels manifest '{p}': {e}")

        # Summary path
        file_summary_path: Path | None = None
        if isinstance(file_summary, str) and file_summary.strip():
            file_summary_path = resolver.anchor(file_summary)
        effective_summary_path = file_summary_path
        if file_summary_path is not None:
            try:
                file_summary_path.parent.mkdir(parents=True, exist_ok=True)
                # Only probe lock if the file already exists; avoid creating it early so header logic can run
                if file_summary_path.exists():
                    try:
                        with file_summary_path.open("a", encoding="utf-8"):
                            pass
                    except Exception:
                        from datetime import datetime
                        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                        alt = file_summary_path.with_name(f"{file_summary_path.stem}-{ts}{file_summary_path.suffix}")
                        alt.parent.mkdir(parents=True, exist_ok=True)
                        effective_summary_path = alt
                        self.console.info(f"Summary file locked ('{file_summary_path}'); using '{alt}' for this run.")
            except Exception as e:
                self.console.warn(f"Failed to establish file summary path: {e}")

        cm_accum: "np.ndarray | None" = None
        report_rows: list[tuple[str, str, int, float]] = []
        report_probs: list[tuple[str, list[float]]] = []
        # Iterate only over vector-field inputs when possible
        iter_fields = [s for s in anchored_fields if 'vector_fields' in locals() and s in vector_fields] if 'vector_fields' in locals() and vector_fields else anchored_fields
        # Vector per-file progress bar
        _iter = None
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
            _iter = _tqdm(iter_fields, desc="vector", unit="file")
        except Exception:
            _iter = iter_fields
        for fp in _iter:
            stem = Path(fp).stem
            per_out = out_path.with_name(f"{out_path.stem}-{stem}{out_path.suffix}") if out_path is not None else None
            per_probs = probs_path.with_name(f"{probs_path.stem}-{stem}{probs_path.suffix}") if probs_path is not None else None
            per_logits = logits_path.with_name(f"{logits_path.stem}-{stem}{logits_path.suffix}") if logits_path is not None else None
            per_heat_dir = (heat_dir / stem) if heat_dir is not None else None
            per_stitch = stitch_path.with_name(f"{stitch_path.stem}-{stem}{stitch_path.suffix}") if stitch_path is not None else None
            per_cm_csv = cm_csv_path.with_name(f"{cm_csv_path.stem}-{stem}{cm_csv_path.suffix}") if cm_csv_path is not None else None
            per_cm_png = cm_png_path.with_name(f"{cm_png_path.stem}-{stem}{cm_png_path.suffix}") if cm_png_path is not None else None

            ret_cm = run_inference(
                cfg,
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
                report_list=report_rows,
                report_probs_list=report_probs,
            )
            if ret_cm is not None:
                import numpy as _np
                cm_accum = ret_cm.copy() if cm_accum is None else (cm_accum + ret_cm)

        # Pretty-print a table of per-file predictions to terminal
        try:
            if report_rows:
                headers = ["file_name", "predicted_class", "index", "confidence%"]
                rows = [[a, b, str(c), f"{d:.2f}"] for (a, b, c, d) in report_rows]
                self.console.table(headers, rows, align=["l", "l", "r", "r"], title="Per-file Predictions")
        except Exception:
            pass

        # Also print a table of mean probabilities per file (per class)
        try:
            if report_probs:
                # Determine class headers
                cls = class_names if (class_names and len(class_names) == int(model.classifier[-1].out_features)) else [f"p{i}" for i in range(int(model.classifier[-1].out_features))]
                headers = ["file_name"] + [f"{name}%" for name in cls]
                rows: list[list[str]] = []
                for (fname, probs_list) in report_probs:
                    perc = [f"{float(p)*100.0:.2f}" for p in probs_list]
                    rows.append([fname] + perc)
                aligns = ["l"] + ["r"] * (len(headers) - 1)
                self.console.table(headers, rows, align=aligns, title="Per-file Mean Probabilities")
        except Exception:
            pass

        if cm_accum is not None:
            try:
                overall_csv = cm_csv_path.with_name(f"{cm_csv_path.stem}-overall{cm_csv_path.suffix}") if cm_csv_path is not None else None
                overall_png = cm_png_path.with_name(f"{cm_png_path.stem}-overall{cm_png_path.suffix}") if cm_png_path is not None else None
                save_confusion_outputs(cm_accum, class_names, overall_csv, overall_png, console=self.console)
            except Exception as e:
                self.console.warn(f"Failed to save overall confusion matrix: {e}")

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
