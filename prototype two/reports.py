from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common import ConfigManager, PathResolver, save_confusion_outputs, ensure_outputs_ready, plot_training_progress
from console import console_from_config
try:
    from data_loader import build_class_index, _resolve_root_path as _resolve_root_path_img
except Exception:
    build_class_index = None  # type: ignore
    _resolve_root_path_img = None  # type: ignore


@dataclass
class _Record:
    rel_path: Path
    predicted_class: str
    predicted_id: int
    confidence_pct: float
    true_class: str | None
    velocity_pred: float | None
    reynolds_pred: float | None


def _true_class_name(rel_path: Path) -> str | None:
    parts = rel_path.parts
    return parts[0] if parts else None


def _load_labels_map(labels_csv: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    try:
        with labels_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = str(row.get("file_name", row.get("filename", row.get("image_path", "")))).strip()
                label = str(row.get("label", "")).strip()
                if not fname or label == "":
                    continue
                mapping[fname] = label
                mapping[Path(fname).name] = label
    except Exception:
        pass
    return mapping


def _discover_class_dirs(root: Path) -> List[str]:
    try:
        return sorted([p.name for p in root.iterdir() if p.is_dir()])
    except Exception:
        return []


def _load_records(csv_path: Path, root: Path, mapping: Dict[str, int] | None, console, labels_map: Dict[str, str] | None = None) -> List[_Record]:
    records: List[_Record] = []
    if not csv_path.exists():
        console.warn(f"[reports] predictions CSV not found: {csv_path}")
        return records

    def _parse_float(value: str | None) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except Exception:
            return 0.0

    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_str = (
                    row.get("image_path")
                    or row.get("file")
                    or row.get("image_name")
                    or row.get("path")
                )
                if not rel_str:
                    continue
                rel_path = Path(rel_str)
                pred_class = row.get("predicted_class", "").strip()
                pred_id_raw = (
                    row.get("identifier")
                    or row.get("predicted_id")
                    or row.get("predicted_index")
                    or ""
                )
                try:
                    pred_idx = int(pred_id_raw)
                except Exception:
                    if mapping and pred_class in mapping:
                        pred_idx = int(mapping[pred_class])
                    else:
                        try:
                            pred_idx = int(float(pred_id_raw))
                        except Exception:
                            pred_idx = -1
                conf = _parse_float(
                    row.get("confidence_pct")
                    or row.get("%accuracy")
                    or row.get("confidence")
                )
                vel_pred = _parse_float(row.get("velocity_pred"))
                re_pred = _parse_float(row.get("reynolds_pred"))
                true_cls = None
                if labels_map:
                    candidates = [rel_path.name, rel_path.as_posix(), str(rel_path), rel_path.stem]
                    for key in candidates:
                        if key in labels_map:
                            true_cls = labels_map[key]
                            break
                if true_cls is None:
                    true_cls = _true_class_name(rel_path)
                records.append(
                    _Record(
                        rel_path=rel_path,
                        predicted_class=pred_class,
                        predicted_id=pred_idx,
                        confidence_pct=conf,
                        true_class=true_cls,
                        velocity_pred=vel_pred,
                        reynolds_pred=re_pred,
                    )
                )
    except Exception as e:
        console.warn(f"[reports] Failed to parse predictions CSV '{csv_path}': {e}")
    return records


def _compute_confusion_matrix(
    *,
    records: List[_Record],
    mapping: Dict[str, int],
    num_out: int,
    cm_csv: Path | None,
    cm_png: Path | None,
    console,
) -> None:
    y_true: List[int] = []
    y_pred: List[int] = []
    for rec in records:
        if rec.true_class is None or rec.true_class not in mapping:
            continue
        if rec.predicted_id < 0 or rec.predicted_id >= num_out:
            continue
        y_true.append(int(mapping[rec.true_class]))
        y_pred.append(int(rec.predicted_id))

    # Fallback: if no ground-truth labels, produce a diagonal matrix of prediction counts
    if not y_true:
        try:
            console.warn("[reports] No ground-truth labels found; falling back to prediction-count matrix (diagonal only).")
        except Exception:
            pass
        counts = np.zeros((num_out,), dtype=np.int64)
        for rec in records:
            if 0 <= rec.predicted_id < num_out:
                counts[int(rec.predicted_id)] += 1
        cm = np.zeros((num_out, num_out), dtype=np.int64)
        for i in range(num_out):
            cm[i, i] = counts[i]
        class_names = list(mapping.keys())
        save_confusion_outputs(cm, class_names, cm_csv, cm_png, console=console)
        try:
            headers = ["true\\pred"] + class_names
            rows: List[List[str]] = []
            for i, tn in enumerate(class_names):
                rows.append([tn] + [str(int(cm[i, j])) for j in range(num_out)])
            console.table(headers, rows, align=["l"] + ["r"] * len(class_names), title="Prediction count matrix (no labels)")
        except Exception:
            pass
        return
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)[: y_true_arr.shape[0]]
    idx = (y_true_arr * num_out + y_pred_arr).astype(np.int64)
    cm = np.bincount(idx, minlength=num_out * num_out).reshape(num_out, num_out)
    class_names = list(mapping.keys())
    save_confusion_outputs(cm, class_names, cm_csv, cm_png, console=console)
    try:
        headers = ["true\\pred"] + class_names
        rows: List[List[str]] = []
        for i, tn in enumerate(class_names):
            rows.append([tn] + [str(int(cm[i, j])) for j in range(num_out)])
        console.table(headers, rows, align=["l"] + ["r"] * len(class_names), title="Image confusion matrix")
    except Exception:
        pass


def _generate_per_folder_reports(
    *,
    records: List[_Record],
    mapping: Dict[str, int],
    num_out: int,
    base_dir: Path,
    console,
) -> None:
    names = list(mapping.keys())
    records_by_class: Dict[str, List[_Record]] = {name: [] for name in names}
    for rec in records:
        if rec.true_class in records_by_class:
            records_by_class[rec.true_class].append(rec)

    console.info(f"[reports] base folder: {base_dir}")
    for cls_name in names:
        cls_records = records_by_class.get(cls_name, [])
        if not cls_records:
            continue
        cls_dir = (base_dir / cls_name).resolve()
        cls_dir.mkdir(parents=True, exist_ok=True)
        if cls_name.lower() in ("laminar", "turbulent") and num_out == 2:
            try:
                true_idx = int(mapping[cls_name])
                y_true_cls = np.full((len(cls_records),), true_idx, dtype=np.int64)
                y_pred_cls = np.asarray(
                    [max(rec.predicted_id, -1) for rec in cls_records],
                    dtype=np.int64,
                )
                idx = (y_true_cls * num_out + np.clip(y_pred_cls, 0, num_out - 1)).astype(np.int64)
                cm_cls = np.bincount(idx, minlength=num_out * num_out).reshape(num_out, num_out)
                csv_p = cls_dir / f"cm-{cls_name}.csv"
                png_p = cls_dir / f"cm-{cls_name}.png"
                console.info(f"[reports] saving per-folder CM for {cls_name} -> {csv_p}, {png_p}")
                save_confusion_outputs(
                    cm_cls,
                    names if len(names) == num_out else [str(i) for i in range(num_out)],
                    csv_p,
                    png_p,
                    console=console,
                )
                try:
                    label_names = names if len(names) == num_out else [f"class{i}" for i in range(num_out)]
                    headers = ["true\\pred"] + label_names
                    rows: List[List[str]] = []
                    for r_idx, label in enumerate(label_names):
                        row = [label] + [str(int(cm_cls[r_idx, c_idx])) for c_idx in range(num_out)]
                        rows.append(row)
                    console.table(headers, rows, align=["l"] + ["r"] * len(label_names), title=f"{cls_name} confusion matrix")
                except Exception:
                    pass
            except Exception as e:
                console.warn(f"Failed per-folder CM for {cls_name}: {e}")
        else:
            try:
                counts = np.zeros((num_out,), dtype=np.int64)
                for rec in cls_records:
                    if 0 <= rec.predicted_id < num_out:
                        counts[rec.predicted_id] += 1
                dist_csv = cls_dir / f"pred_distribution-{cls_name}.csv"
                with dist_csv.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["predicted_index", "count"])
                    for idx, count in enumerate(counts):
                        writer.writerow([idx, int(count)])
                console.info(f"[reports] saved distribution for {cls_name}: {dist_csv}")
            except Exception as e:
                console.warn(f"Failed prediction distribution for {cls_name}: {e}")


def _generate_plots(
    *,
    records: List[_Record],
    mapping: Dict[str, int] | None,
    base_dir: Path,
    config: Dict[str, Any],
    console,
) -> None:
    try:
        import matplotlib.pyplot as _plt  # type: ignore
        from matplotlib import ticker as _ticker  # type: ignore
    except Exception as e:
        console.warn(f"matplotlib unavailable for plots: {e}")
        return

    plots_dir = (base_dir / "plots").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Regression plots (velocity, Reynolds) with optional rolling average
    vel_vals = [r.velocity_pred for r in records if r.velocity_pred is not None]
    re_vals = [r.reynolds_pred for r in records if r.reynolds_pred is not None]

    def _rolling_mean(arr: List[float], window: int = 3) -> List[float]:
        if window <= 1 or len(arr) < window:
            return arr
        out: List[float] = []
        cumsum = np.cumsum([0.0] + arr, dtype=float)
        for i in range(window, len(arr) + 1):
            out.append(float((cumsum[i] - cumsum[i - window]) / window))
        # align to original length by padding the first (window-1) entries with leading values
        if out:
            pad = [out[0]] * (window - 1)
            return pad + out
        return arr

    def _butter_filter(arr: List[float], order: int = 6, cutoff: float = 0.1) -> List[float] | None:
        """
        Apply a low-pass Butterworth filter (filtfilt) to smooth the sequence.
        cutoff is normalized to Nyquist (0<cutoff<1).
        """
        try:
            # Skip known-incompatible SciPy if NumPy 2.x is installed and SciPy wheels are outdated
            np_ver = tuple(int(p) for p in np.__version__.split(".")[:2] if p.isdigit())
            if np_ver and np_ver[0] >= 2:
                try:
                    import importlib.util as _ilu  # type: ignore
                    if _ilu.find_spec("scipy") is not None:
                        console.warn("[reports] Skipping Butterworth filter: SciPy wheels likely incompatible with NumPy 2.x (saw NumPy "
                                     f"{np.__version__}). Install scipy built for NumPy 2.x or downgrade NumPy (<2).")
                except Exception:
                    pass
                return None
            import scipy.signal as _sig  # type: ignore
            if len(arr) < order + 2:
                return None
            b, a = _sig.butter(order, cutoff, btype="low", analog=False)
            filtered = _sig.filtfilt(b, a, np.asarray(arr, dtype=float))
            return filtered.tolist()
        except Exception as e:
            try:
                console.warn(f"[reports] Butterworth filter unavailable: {e}")
            except Exception:
                pass
            return None

    if vel_vals:
        x_vals = list(range(1, len(vel_vals) + 1))
        try:
            vel_avg = _rolling_mean(vel_vals, window=3)
            vel_avg10 = _rolling_mean(vel_vals, window=10)
            vel_avg100 = _rolling_mean(vel_vals, window=100)
            fit_100 = None
            if len(vel_vals) >= 100:
                try:
                    x_arr = np.arange(1, len(vel_avg100) + 1, dtype=float)
                    y_arr = np.asarray(vel_avg100, dtype=float)
                    # 8th-order polynomial fit on the 100-pt average
                    coeffs = np.polyfit(x_arr, y_arr, deg=8)
                    fit_100 = np.polyval(coeffs, x_arr)
                except Exception:
                    fit_100 = None
            _plt.figure(figsize=(6, 4))
            _plt.plot(x_vals, vel_vals, color="#2C7BB6", linewidth=1.0, alpha=0.6, label="velocity (pred)")
            _plt.plot(x_vals, vel_avg, color="#1B4F72", linewidth=1.4, label="3-pt avg")
            if len(vel_vals) >= 10:
                _plt.plot(x_vals, vel_avg10, color="#0B3C5D", linewidth=1.2, linestyle="--", label="10-pt avg")
            if len(vel_vals) >= 100:
                _plt.plot(x_vals, vel_avg100, color="#062133", linewidth=1.0, linestyle=":", label="100-pt avg")
            if fit_100 is not None:
                _plt.plot(x_vals, fit_100, color="#FFD700", linewidth=1.2, linestyle="-.", label="100-pt fit (8th)")
            _plt.xlabel("image index")
            _plt.ylabel("velocity (predicted)")
            _plt.title("Predicted velocity")
            _plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
            _plt.legend()
            out_vel = plots_dir / "velocity_predictions.png"
            console.info(f"[reports] plotting velocity predictions -> {out_vel}")
            _plt.tight_layout()
            _plt.savefig(out_vel, dpi=150)
            _plt.close()
        except Exception as e:
            console.warn(f"[reports] failed to plot velocity predictions: {e}")
        # Butterworth-smoothed velocity plot
        vel_butter = _butter_filter(vel_vals, order=6, cutoff=0.1)
        if vel_butter:
            try:
                _plt.figure(figsize=(6, 4))
                _plt.plot(x_vals, vel_vals, color="#B0C4DE", linewidth=1.0, alpha=0.5, label="velocity (raw)")
                _plt.plot(x_vals, vel_butter, color="#FFD700", linewidth=1.4, label="velocity (6th-order Butterworth)")
                _plt.xlabel("image index")
                _plt.ylabel("velocity (predicted)")
                _plt.title("Velocity (Butterworth smoothed)")
                _plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
                _plt.legend()
                out_vel_b = plots_dir / "velocity_butterworth.png"
                console.info(f"[reports] plotting Butterworth velocity -> {out_vel_b}")
                _plt.tight_layout()
                _plt.savefig(out_vel_b, dpi=150)
                _plt.close()
            except Exception as e:
                console.warn(f"[reports] failed to plot Butterworth velocity: {e}")
    if re_vals:
        x_vals = list(range(1, len(re_vals) + 1))
        try:
            re_avg = _rolling_mean(re_vals, window=3)
            re_avg10 = _rolling_mean(re_vals, window=10)
            re_avg100 = _rolling_mean(re_vals, window=100)
            fit_100 = None
            if len(re_vals) >= 100:
                try:
                    x_arr = np.arange(1, len(re_avg100) + 1, dtype=float)
                    y_arr = np.asarray(re_avg100, dtype=float)
                    coeffs = np.polyfit(x_arr, y_arr, deg=8)
                    fit_100 = np.polyval(coeffs, x_arr)
                except Exception:
                    fit_100 = None
            _plt.figure(figsize=(6, 4))
            _plt.plot(x_vals, re_vals, color="#D9534F", linewidth=1.0, alpha=0.6, label="Re (pred)")
            _plt.plot(x_vals, re_avg, color="#A52A2A", linewidth=1.4, label="3-pt avg")
            if len(re_vals) >= 10:
                _plt.plot(x_vals, re_avg10, color="#7B241C", linewidth=1.2, linestyle="--", label="10-pt avg")
            if len(re_vals) >= 100:
                _plt.plot(x_vals, re_avg100, color="#4A0D0D", linewidth=1.0, linestyle=":", label="100-pt avg")
            if fit_100 is not None:
                _plt.plot(x_vals, fit_100, color="#FFD700", linewidth=1.2, linestyle="-.", label="100-pt fit (8th)")
            _plt.xlabel("image index")
            _plt.ylabel("Reynolds number (predicted)")
            _plt.title("Predicted Reynolds number")
            _plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
            _plt.legend()
            out_re = plots_dir / "reynolds_predictions.png"
            console.info(f"[reports] plotting Reynolds predictions -> {out_re}")
            _plt.tight_layout()
            _plt.savefig(out_re, dpi=150)
            _plt.close()
        except Exception as e:
            console.warn(f"[reports] failed to plot Reynolds predictions: {e}")
        # Butterworth-smoothed Reynolds plot
        re_butter = _butter_filter(re_vals, order=6, cutoff=0.1)
        if re_butter:
            try:
                _plt.figure(figsize=(6, 4))
                _plt.plot(x_vals, re_vals, color="#F5B7B1", linewidth=1.0, alpha=0.5, label="Re (raw)")
                _plt.plot(x_vals, re_butter, color="#FFD700", linewidth=1.4, label="Re (6th-order Butterworth)")
                _plt.xlabel("image index")
                _plt.ylabel("Reynolds number (predicted)")
                _plt.title("Reynolds (Butterworth smoothed)")
                _plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
                _plt.legend()
                out_re_b = plots_dir / "reynolds_butterworth.png"
                console.info(f"[reports] plotting Butterworth Reynolds -> {out_re_b}")
                _plt.tight_layout()
                _plt.savefig(out_re_b, dpi=150)
                _plt.close()
            except Exception as e:
                console.warn(f"[reports] failed to plot Butterworth Reynolds: {e}")

    if mapping:
        lower_name_map = {name.lower(): name for name in mapping.keys()}
        if "laminar" in lower_name_map and "turbulent" in lower_name_map:
            lam_idx = mapping.get(lower_name_map["laminar"])
            tur_idx = mapping.get(lower_name_map["turbulent"])
            if lam_idx is not None and tur_idx is not None:
                y_vals: List[int] = []
                for rec in records:
                    if rec.predicted_id == lam_idx:
                        y_vals.append(1)
                    elif rec.predicted_id == tur_idx:
                        y_vals.append(0)
                if y_vals:
                    _plt.figure(figsize=(6, 4))
                    x_vals = list(range(1, len(y_vals) + 1))
                    _plt.scatter(x_vals, y_vals, s=10, color="#5B8FF9")
                    _plt.yticks([0, 1], ["turbulent (0)", "laminar (1)"])
                    _plt.xlabel("image index")
                    _plt.ylabel("predicted class")
                    _plt.title("Predicted laminar/turbulent")
                    _plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
                    out_step = plots_dir / "classification_step.png"
                    console.info(f"[reports] plotting laminar/turbulent -> {out_step}")
                    _plt.tight_layout()
                    _plt.savefig(out_step, dpi=150)
                    _plt.close()


def _maybe_plot_training_progress(*, base_dir: Path, config: Dict[str, Any], console) -> None:
    history_candidates: List[Path] = []
    hist_cfg = config.get("training_history_csv") or config.get("training_history")
    if isinstance(hist_cfg, str) and hist_cfg.strip():
        history_candidates.append(Path(hist_cfg))
    history_candidates.append(base_dir / "training" / "training_history.csv")

    csv_path: Path | None = None
    for candidate in history_candidates:
        if candidate is None:
            continue
        p = candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
        if p.exists():
            csv_path = p
            break
    if csv_path is None or not csv_path.exists():
        return

    try:
        history_rows: List[Dict[str, float]] = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    history_rows.append(
                        {
                            "epoch": float(row.get("epoch", len(history_rows) + 1)),
                            "train_loss": float(row.get("train_loss", 0.0)),
                            "val_loss": float(row.get("val_loss", 0.0)),
                            "train_acc": float(row.get("train_acc", 0.0)),
                            "val_acc": float(row.get("val_acc", 0.0)),
                            "epoch_time_sec": float(row.get("epoch_time_sec", row.get("epoch_time", 0.0)) or 0.0),
                            "total_time_sec": float(row.get("total_time_sec", 0.0) or 0.0),
                        }
                    )
                except Exception:
                    continue
        if not history_rows:
            console.warn(f"[reports] training history CSV was empty: {csv_path}")
            return
    except Exception as e:
        console.warn(f"[reports] Failed to parse training history CSV '{csv_path}': {e}")
        return

    training_dir = (base_dir / "training").resolve()
    training_dir.mkdir(parents=True, exist_ok=True)
    out_png = (training_dir / "training_progress.png").resolve()
    model_name = str(config.get("model_name", config.get("model_type", ""))).strip()
    custom_title = f"Training Progress - {model_name}" if model_name else "Training Progress"
    console.info(f"[reports] plotting training progress -> {out_png}")
    plot_training_progress(history_rows, out_png, console=console, title=custom_title)

    # Additional plot: epoch vs accuracy (%) with per-epoch and total time
    try:
        import matplotlib.pyplot as _plt  # type: ignore

        epochs = [r.get("epoch", i + 1) for i, r in enumerate(history_rows)]
        train_acc = [float(r.get("train_acc", 0.0)) * 100.0 for r in history_rows]
        val_acc = [float(r.get("val_acc", 0.0)) * 100.0 for r in history_rows]
        epoch_time = [float(r.get("epoch_time_sec", 0.0)) for r in history_rows]
        total_time = [float(r.get("total_time_sec", 0.0)) for r in history_rows]
        # If total_time not recorded, derive cumulative from epoch_time
        if not any(t > 0 for t in total_time) and any(et > 0 for et in epoch_time):
            cum = 0.0
            total_time = []
            for et in epoch_time:
                cum += et
                total_time.append(cum)
        epoch_time_min = [t / 60.0 for t in epoch_time] if any(t > 0 for t in epoch_time) else None
        total_time_min = [t / 60.0 for t in total_time] if any(t > 0 for t in total_time) else None

        _plt.figure(figsize=(8, 5))
        ax1 = _plt.gca()
        ax1.plot(epochs, train_acc, label="Train acc (%)", color="#1f77b4", marker="o", markersize=3)
        if any(val_acc):
            ax1.plot(epochs, val_acc, label="Val acc (%)", color="#d62728", marker="^", markersize=3)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.grid(alpha=0.3, linestyle="--", linewidth=0.5)

        if total_time_min:
            ax2 = ax1.twinx()
            if total_time_min:
                ax2.plot(epochs, total_time_min, label="Total time (min)", color="#9467bd", linestyle="-.")
            ax2.set_ylabel("Time (minutes)")
            # Combine legends
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="lower right")
        else:
            ax1.legend(loc="lower right")

        extra_out = (training_dir / "training_accuracy_time.png").resolve()
        _plt.title(custom_title + " (Accuracy & Time)")
        _plt.tight_layout()
        _plt.savefig(extra_out, dpi=150)
        _plt.close()
        console.info(f"[reports] plotting training accuracy/time -> {extra_out}")
    except Exception as e:
        console.warn(f"[reports] Failed to plot training accuracy/time: {e}")


def generate_image_reports(
    *,
    csv_path: Path | None,
    root: Path,
    mapping: Dict[str, int] | None,
    base_outputs: Path | None,
    out_subdir: str | None,
    cm_csv: Path | None,
    cm_png: Path | None,
    config: Dict[str, Any],
    console,
    labels_csv: Path | None = None,
) -> None:
    if csv_path is None:
        console.warn("[reports] No predictions CSV provided; skipping reports generation")
        return
    csv_path = Path(csv_path)
    labels_map = _load_labels_map(labels_csv) if labels_csv is not None else {}
    if labels_csv is not None and not labels_map:
        console.warn(f"[reports] labels CSV was empty or unreadable: {labels_csv}")
    records = _load_records(csv_path, root, mapping, console, labels_map=labels_map)
    if not records:
        console.warn("[reports] No prediction rows found; skipping reports generation")
        return
    # If the folder-based mapping is missing/misleading (e.g., single folder like 'set one'),
    # rebuild it from labels or predicted classes so confusion matrices align with laminar/turbulent.
    try:
        needs_remap = mapping is None or len(mapping) <= 1
        candidate_classes: list[str] = []
        # Prefer labels_map values
        if labels_map:
            candidate_classes = sorted({str(v).strip() for v in labels_map.values() if str(v).strip()})
        # Else predicted class strings
        if not candidate_classes:
            candidate_classes = sorted({r.predicted_class for r in records if r.predicted_class})
        # Else true_class fields
        if not candidate_classes:
            candidate_classes = sorted({r.true_class for r in records if r.true_class})
        if needs_remap and candidate_classes:
            # Preserve common ordering when both classes are present
            if {"turbulent", "laminar"} <= set(c.lower() for c in candidate_classes):
                ordered = ["turbulent", "laminar"]
                mapping = {name: i for i, name in enumerate(ordered)}
            else:
                mapping = {name: i for i, name in enumerate(candidate_classes)}
            console.info(f"[reports] using rebuilt class mapping: {mapping}")
    except Exception as e:
        console.warn(f"[reports] could not rebuild class mapping: {e}")
    try:
        console.info(f"[reports] loaded predictions from {csv_path} (rows={len(records)})")
    except Exception:
        pass

    mapping = mapping or {}
    observed_classes = sorted({rec.true_class for rec in records if rec.true_class})
    dir_classes = _discover_class_dirs(root)
    derived_classes = dir_classes if dir_classes else observed_classes
    next_idx = max(mapping.values(), default=-1) + 1 if mapping else 0
    for cls in derived_classes:
        if cls not in mapping:
            mapping[cls] = next_idx
            next_idx += 1

    def _as_int(val: Any, default: int = 0) -> int:
        try:
            return int(val)
        except Exception:
            return default

    cfg_num = _as_int(config.get("num_classes"), 0)
    mapping_span = max(mapping.values(), default=-1) + 1 if mapping else 0
    num_out = max(2, cfg_num, mapping_span)
    base_root = base_outputs if base_outputs is not None else (Path(".") / "data" / "outputs")
    sub = out_subdir if isinstance(out_subdir, str) and out_subdir.strip() else "image"
    base_dir = (base_root / sub).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        console.info(f"[reports] using predictions CSV: {csv_path}")
    except Exception:
        pass

    want_cm = bool(config.get("generate_confusion_matrix", True))
    if mapping and want_cm:
        _compute_confusion_matrix(
            records=records,
            mapping=mapping,
            num_out=num_out,
            cm_csv=cm_csv,
            cm_png=cm_png,
            console=console,
        )

    if mapping:
        _generate_per_folder_reports(
            records=records,
            mapping=mapping,
            num_out=num_out,
            base_dir=base_dir,
            console=console,
        )

    _generate_plots(
        records=records,
        mapping=mapping if mapping else None,
        base_dir=base_dir,
        config=config,
        console=console,
    )

    _maybe_plot_training_progress(base_dir=base_dir, config=config, console=console)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process inference CSV outputs into reports and plots.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.json")),
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Override predictions CSV path (defaults to reports.predictions_csv or file_summary in config)",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Root directory containing class subfolders (defaults to reports.image_root or test_image_dir)",
    )
    parser.add_argument("--labels_csv", type=str, default=None, help="CSV with per-image labels: file_name,label")
    parser.add_argument("--cm_csv", type=str, default=None, help="Optional override for saved confusion-matrix CSV")
    parser.add_argument("--cm_png", type=str, default=None, help="Optional override for saved confusion-matrix PNG")
    args = parser.parse_args()

    cfg_mgr = ConfigManager(Path(args.config))
    cfg = cfg_mgr.config
    ensure_outputs_ready(cfg_mgr.path, cfg)
    console = console_from_config(cfg)
    resolver = PathResolver(cfg_mgr.path, outputs_root=str(cfg.get("outputs_root")) if isinstance(cfg.get("outputs_root"), str) else None)
    reports_cfg = cfg.get("reports", {}) if isinstance(cfg.get("reports"), dict) else {}

    def _anchor(val: str | None) -> Path | None:
        if not isinstance(val, str) or not val.strip():
            return None
        anchored = resolver.anchor(val)
        return anchored if anchored is not None else Path(val).resolve()

    csv_path = _anchor(args.csv) or _anchor(reports_cfg.get("predictions_csv")) or _anchor(cfg.get("file_summary"))
    image_root = _anchor(args.image_root) or _anchor(reports_cfg.get("image_root")) or _anchor(cfg.get("test_image_dir"))
    labels_csv = _anchor(args.labels_csv) or _anchor(reports_cfg.get("labels_csv")) or _anchor(cfg.get("file_labels"))
    if csv_path is None:
        raise SystemExit("No predictions CSV provided. Pass --csv or set reports.predictions_csv in the config.")
    if image_root is None:
        raise SystemExit("Image root not provided. Pass --image_root or set reports.image_root/test_image_dir in the config.")

    # Allow image_root to be a pointer file (txt/csv) containing the actual directory path
    if _resolve_root_path_img is not None:
        try:
            image_root = _resolve_root_path_img(image_root)
        except Exception:
            pass

    if not image_root.exists():
        raise SystemExit(f"Image root not found: {image_root}")

    class_names = cfg.get("class_names", None)
    mapping = None
    if build_class_index is not None:
        try:
            mapping = build_class_index(image_root, class_names)
        except Exception as e:
            console.warn(f"[reports] Could not build class index: {e}")

    base_outputs = resolver.outputs_root
    cm_csv = _anchor(args.cm_csv) or _anchor(reports_cfg.get("cm_csv"))
    cm_png = _anchor(args.cm_png) or _anchor(reports_cfg.get("cm_png"))
    subdir = reports_cfg.get("subdir")

    generate_image_reports(
        csv_path=csv_path,
        root=image_root,
        mapping=mapping,
        base_outputs=base_outputs,
        out_subdir=subdir,
        cm_csv=cm_csv,
        cm_png=cm_png,
        config=cfg,
        console=console,
        labels_csv=labels_csv,
    )


if __name__ == "__main__":
    main()
