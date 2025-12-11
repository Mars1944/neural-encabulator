"""
One-shot benchmark + analysis to recommend batch_size and num_workers.

Runs a single epoch for each combo, caches the dataset to cut overhead,
saves CSV and plot, and prints the best settings for your config.

Usage:
  python bench_optimize.py
  python bench_optimize.py --grid "16:2,32:4" --debug-train-fraction 0.01
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt  # type: ignore
import torch

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

from common import ConfigManager
from main import ModelManager
from trainer import Trainer

# Editable defaults: cross-product of these vectors is used when no CLI overrides.
BATCH_VECTOR: List[int] = [16, 32, 48]
WORKER_VECTOR: List[int] = [2, 4, 6]
LIMIT_SAMPLES_RATIO: float = 0.01  # fraction of train data per run for speed


def parse_int_list(raw: str) -> List[int]:
    return [int(x) for x in raw.replace(";", ",").split(",") if x.strip()]


def parse_grid(raw: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for chunk in raw.replace(";", ",").split(","):
        if ":" not in chunk:
            continue
        bs_s, w_s = chunk.split(":", 1)
        try:
            pairs.append((int(bs_s), int(w_s)))
        except Exception:
            continue
    return pairs


def run_one_epoch(trainer: Trainer, train_loader: torch.utils.data.DataLoader) -> Tuple[float, float, Dict[str, float]]:
    n_images = len(train_loader.dataset) if hasattr(train_loader, "dataset") else None
    proc = psutil.Process() if psutil else None
    if trainer.device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(trainer.device)
        except Exception:
            pass
    if proc:
        proc.cpu_percent(None)  # prime
    start = time.time()
    trainer._train_one_epoch(train_loader, epoch=1)  # type: ignore[attr-defined]
    duration = time.time() - start
    cpu_pct = proc.cpu_percent(interval=None) if proc else None
    ram_mb = proc.memory_info().rss / (1024.0 * 1024.0) if proc else None
    cuda_peak = None
    if trainer.device.type == "cuda":
        try:
            cuda_peak = torch.cuda.max_memory_allocated(trainer.device) / (1024.0 * 1024.0)
        except Exception:
            cuda_peak = None
    ips = (n_images / duration) if (n_images and duration > 0) else 0.0
    return duration, ips, {"ram_mb": ram_mb, "cpu_percent": cpu_pct, "cuda_peak_mb": cuda_peak}


def plot_results(rows: List[Dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    plt.figure(figsize=(7, 5))
    markers = ["o", "s", "^", "D", "x", "+"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    workers = sorted({int(r["num_workers"]) for r in rows})
    for idx, w in enumerate(workers):
        xs = [r["batch_size"] for r in rows if int(r["num_workers"]) == w]
        ys = [r["imgs_per_sec"] for r in rows if int(r["num_workers"]) == w]
        plt.plot(xs, ys, marker=markers[idx % len(markers)], color=colors[idx % len(colors)], label=f"workers={w}")
    plt.xlabel("batch_size")
    plt.ylabel("images/sec")
    plt.title("Throughput vs batch_size/num_workers")
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[bench-opt] Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark and recommend batch_size/num_workers.")
    default_cfg = Path(__file__).resolve().with_name("config.json")
    default_csv = Path(__file__).resolve().with_name("bench_epoch_results.csv")
    parser.add_argument("--config", type=str, default=str(default_cfg), help="Path to config.json")
    parser.add_argument("--csv", type=str, default=str(default_csv), help="Output CSV path")
    parser.add_argument("--out", type=str, default="bench_analysis.png", help="Output plot path")
    parser.add_argument("--batch-sizes", type=str, default="", help="Override batch sizes (comma/semicolon). Leave empty to use BATCH_VECTOR.")
    parser.add_argument("--num-workers", type=str, default="", help="Override worker counts (comma/semicolon). Leave empty to use WORKER_VECTOR.")
    parser.add_argument("--grid", type=str, default="", help="Optional paired grid, e.g., '16:2,32:4' for batch:num_workers combinations.")
    parser.add_argument("--debug-train-fraction", type=float, default=None, help="Fraction of training data to use; overrides LIMIT_SAMPLES_RATIO if set.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sibling = Path(__file__).resolve().with_name("config.json")
        if sibling.exists():
            cfg_path = sibling
        else:
            raise FileNotFoundError(f"Config file not found: {args.config}")

    base_cfg = ConfigManager(cfg_path)
    if args.grid.strip():
        combos = parse_grid(args.grid)
    elif args.batch_sizes.strip() or args.num_workers.strip():
        batch_sizes = parse_int_list(args.batch_sizes) if args.batch_sizes.strip() else list(BATCH_VECTOR)
        worker_counts = parse_int_list(args.num_workers) if args.num_workers.strip() else list(WORKER_VECTOR)
        combos = [(bs, w) for w in worker_counts for bs in batch_sizes]
    else:
        combos = [(bs, w) for w in WORKER_VECTOR for bs in BATCH_VECTOR]
    limit_frac = float(args.debug_train_fraction) if args.debug_train_fraction is not None else float(LIMIT_SAMPLES_RATIO)
    unique_workers = sorted({w for _, w in combos})

    print(f"[bench-opt] Using config: {cfg_path}")
    print(f"[bench-opt] Combos (batch_size, num_workers): {combos}")
    print(f"[bench-opt] debug_limit_training=True, debug_train_fraction={limit_frac}")

    manager = ModelManager(base_cfg.config, base_cfg.device)
    trainer = Trainer(base_cfg.config, base_cfg.device, manager.model)
    cached_ds = None
    collate_fn = None
    pin_memory_default = bool(base_cfg.config.get("pin_memory", False))

    results: List[Dict[str, float]] = []
    for bs, w in combos:
        trainer.config["batch_size"] = bs
        trainer.config["num_workers"] = w
        trainer.config["debug_limit_training"] = True
        trainer.config["debug_train_fraction"] = limit_frac

        if cached_ds is None:
            init_loader, _ = trainer._build_image_loaders()  # type: ignore[attr-defined]
            cached_ds = init_loader.dataset
            pin_memory_default = bool(trainer.config.get("pin_memory", False))
            collate_fn = Trainer._flow_collate if trainer.is_multitask else None
            print(f"[bench-opt] Cached dataset size: {len(cached_ds) if hasattr(cached_ds, '__len__') else 'unknown'}")

        train_loader = torch.utils.data.DataLoader(
            cached_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=w,
            pin_memory=pin_memory_default,
            collate_fn=collate_fn,
        )

        print(f"[bench-opt] batch_size={bs} num_workers={w} -> running one epoch...")
        dur, ips, metrics = run_one_epoch(trainer, train_loader)
        row = {
            "batch_size": bs,
            "num_workers": w,
            "seconds": dur,
            "imgs_per_sec": ips,
            "ram_mb": metrics.get("ram_mb", ""),
            "cpu_percent": metrics.get("cpu_percent", ""),
            "cuda_peak_mb": metrics.get("cuda_peak_mb", ""),
        }
        results.append(row)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Save CSV
    import csv

    csv_path = Path(args.csv)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["batch_size", "num_workers", "seconds", "imgs_per_sec", "ram_mb", "cpu_percent", "cuda_peak_mb"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[bench-opt] Saved results CSV to {csv_path}")

    # Plot
    plot_results(results, Path(args.out))

    # Recommend best
    if results:
        best = max(results, key=lambda r: r["imgs_per_sec"])
        print(
            f"[bench-opt] Best: batch_size={int(best['batch_size'])}, num_workers={int(best['num_workers'])}, "
            f"imgs/sec={best['imgs_per_sec']:.1f}, seconds={best['seconds']:.2f}"
        )
        print(
            "To apply, set in your config/training block:\n"
            f'  "batch_size": {int(best["batch_size"])},\n'
            f'  "num_workers": {int(best["num_workers"])}\n'
            "and rerun training."
        )


if __name__ == "__main__":
    main()
