"""
Quick benchmark runner: steps through combinations of batch_size and num_workers, running a single training epoch for each.
It reports wall-clock seconds and images/sec per combo, and produces a comparison plot.

Usage:
  python bench_epoch.py --config config.json --batch-sizes 16,32,48 --num-workers 2,4,6
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt  # type: ignore
import torch
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

from common import ConfigManager
from main import ModelManager
from trainer import Trainer

# Editable vectors for batch sizes and worker counts.
# You can edit these directly, and they will form the full cross-product when --use-matrix is passed.
BATCH_VECTOR: List[int] = [4, 8, 12, 16, 20, 28, 32, 48, 64]
WORKER_VECTOR: List[int] = [2, 4, 6, 8, 10, 12, 14, 16, 18]
# Limiting samples ratio for quick runs (applied via debug_limit_training/debug_train_fraction)
LIMIT_SAMPLES_RATIO: float = 0.15
# Default epochs per combo for quick sweeps
DEFAULT_EPOCHS: int = 3
# RAM safety watermark (fraction of total RAM used before skipping a run)
RAM_WATERMARK: float = 0.95


def parse_int_list(raw: str) -> List[int]:
    return [int(x) for x in raw.replace(";", ",").split(",") if x.strip()]


def parse_grid(raw: str) -> List[Tuple[int, int]]:
    """Parse a string like '16:2,32:4' into [(16,2),(32,4)]."""
    pairs = []
    for chunk in raw.replace(";", ",").split(","):
        if ":" not in chunk:
            continue
        bs_s, w_s = chunk.split(":", 1)
        try:
            pairs.append((int(bs_s), int(w_s)))
        except Exception:
            continue
    return pairs


def run_one_epoch(
    trainer: Trainer, train_loader: torch.utils.data.DataLoader, epoch_num: int
) -> Tuple[float, float, dict]:
    """
    Run a single training epoch with the provided loader.
    Returns (epoch_seconds, images_per_sec, metrics dict).
    """
    n_images = len(train_loader.dataset) if hasattr(train_loader, "dataset") else None

    proc = psutil.Process() if psutil else None
    mem_before = proc.memory_info() if proc else None
    vm_before = psutil.virtual_memory() if psutil else None
    if trainer.device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(trainer.device)
        except Exception:
            pass

    start = time.time()
    trainer._train_one_epoch(train_loader, epoch=epoch_num)  # type: ignore[attr-defined]
    duration = time.time() - start
    mem_after = proc.memory_info() if proc else None
    vm_after = psutil.virtual_memory() if psutil else None
    if trainer.device.type == "cuda":
        try:
            cuda_alloc = torch.cuda.memory_allocated(trainer.device) / (1024.0 * 1024.0)
            cuda_reserved = torch.cuda.memory_reserved(trainer.device) / (1024.0 * 1024.0)
            cuda_peak = torch.cuda.max_memory_allocated(trainer.device) / (1024.0 * 1024.0)
        except Exception:
            cuda_alloc = cuda_reserved = cuda_peak = None
    else:
        cuda_alloc = cuda_reserved = cuda_peak = None

    imgs_per_sec = (n_images / duration) if (n_images and duration > 0) else 0.0
    metrics: dict = {
        "ram_mb": (mem_after.rss / (1024.0 * 1024.0)) if mem_after else None,
        "vms_mb": (mem_after.vms / (1024.0 * 1024.0)) if mem_after else None,
        "sys_ram_used_mb": (vm_after.used / (1024.0 * 1024.0)) if vm_after else None,
        "sys_ram_avail_mb": (vm_after.available / (1024.0 * 1024.0)) if vm_after else None,
        "cpu_percent": proc.cpu_percent(interval=None) if proc else None,
        "cuda_alloc_mb": cuda_alloc,
        "cuda_reserved_mb": cuda_reserved,
        "cuda_peak_mb": cuda_peak,
    }
    return duration, imgs_per_sec, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Bench one-epoch throughput over batch_size/num_workers grid.")
    default_cfg = Path(__file__).resolve().with_name("config.json")
    parser.add_argument("--config", type=str, default=str(default_cfg), help="Path to config.json")
    parser.add_argument("--batch-sizes", type=str, default="", help="Override batch sizes (comma/semicolon). Leave empty to use BATCH_VECTOR.")
    parser.add_argument("--num-workers", type=str, default="", help="Override worker counts (comma/semicolon). Leave empty to use WORKER_VECTOR.")
    parser.add_argument("--grid", type=str, default="", help="Optional paired grid, e.g., '16:2,32:4' for batch:num_workers combinations.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs to run per combo.")
    parser.add_argument("--debug-train-fraction", type=float, default=None, help="Fraction of training data to use (debug_limit_training). Overrides LIMIT_SAMPLES_RATIO if set.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        # Try sibling config.json next to this script
        sibling = Path(__file__).resolve().with_name("config.json")
        if sibling.exists():
            cfg_path = sibling
        else:
            raise FileNotFoundError(f"Config file not found: {args.config}")

    base_cfg = ConfigManager(cfg_path)
    # Build model/trainer once and cache dataset to minimize overhead between runs
    manager = ModelManager(base_cfg.config, base_cfg.device)
    trainer = Trainer(base_cfg.config, base_cfg.device, manager.model)
    cached_ds = None
    collate_fn = None
    pin_memory_default = bool(base_cfg.config.get("pin_memory", False))
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

    results: List[dict] = []
    total_runs = len(combos)
    bar = tqdm(total=total_runs, desc="bench", unit="run", dynamic_ncols=True, ascii=True) if tqdm is not None else None

    print(f"[bench] Using config: {cfg_path}")
    print(f"[bench] Combos (batch_size, num_workers): {combos}")
    print(f"[bench] debug_limit_training=True, debug_train_fraction={limit_frac}")

    for idx, (bs, w) in enumerate(combos):
        # RAM safeguard: skip combo if system RAM is already too high
        ram_warning = ""
        if psutil:
            vm = psutil.virtual_memory()
            if vm.percent >= RAM_WATERMARK * 100:
                ram_warning = f"skip: RAM {vm.percent:.1f}% >= {RAM_WATERMARK*100:.0f}%"
                print(f"[bench][warn] {ram_warning} (bs={bs}, workers={w})")
                row = {
                    "batch_size": bs,
                    "num_workers": w,
                    "seconds": None,
                    "imgs_per_sec": None,
                    "ram_mb": None,
                    "vms_mb": None,
                    "sys_ram_used_mb": vm.used / (1024.0 * 1024.0),
                    "sys_ram_avail_mb": vm.available / (1024.0 * 1024.0),
                    "cpu_percent": None,
                    "cuda_alloc_mb": None,
                    "cuda_reserved_mb": None,
                    "cuda_peak_mb": None,
                    "warning": ram_warning,
                }
                results.append(row)
                if bar is not None:
                    bar.update(1)
                continue

        # Adjust trainer config for current combo
        trainer.config["batch_size"] = bs
        trainer.config["num_workers"] = w
        trainer.config["debug_limit_training"] = True
        trainer.config["debug_train_fraction"] = limit_frac
        trainer.config["max_epochs"] = int(args.epochs)

        # Cache datasets once (first run)
        if cached_ds is None:
            train_loader_init, _ = trainer._build_image_loaders()  # type: ignore[attr-defined]
            cached_ds = train_loader_init.dataset
            pin_memory_default = bool(trainer.config.get("pin_memory", False))
            collate_fn = Trainer._flow_collate if trainer.is_multitask else None
            print(f"[bench] Cached dataset size: {len(cached_ds) if hasattr(cached_ds, '__len__') else 'unknown'}")

        # Rebuild loader using cached dataset
        train_loader = torch.utils.data.DataLoader(
            cached_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=w,
            pin_memory=pin_memory_default,
            collate_fn=collate_fn,
        )

        if bar is not None:
            bar.set_postfix({"bs": bs, "workers": w})
        else:
            print(f"[bench] batch_size={bs} num_workers={w} -> running {args.epochs} epoch(s)...")
        total_dur = 0.0
        total_ips = 0.0
        metrics = {}
        num_epochs = max(int(args.epochs), 1)
        for e_idx in range(num_epochs):
            if bar is None:
                print(f"[bench] (bs={bs}, workers={w}) epoch {e_idx+1}/{num_epochs} ...")
            dur_one, ips_one, metrics = run_one_epoch(trainer, train_loader, epoch_num=e_idx + 1)
            total_dur += dur_one
            total_ips += ips_one
            if num_epochs > 1:
                print(f"[bench] (bs={bs}, workers={w}) epoch {e_idx+1}/{num_epochs}: duration={dur_one:.2f}s imgs/sec={ips_one:.1f}")
        dur = total_dur / num_epochs
        ips = total_ips / num_epochs
        ram_str = f" | ram_mb={metrics.get('ram_mb', ''):.1f}" if metrics.get("ram_mb") is not None else ""
        cuda_str = ""
        if metrics.get("cuda_peak_mb") is not None:
            cuda_str = f" | cuda_peak_mb={metrics['cuda_peak_mb']:.1f}"
        # Post-run RAM watermark warning
        if psutil and metrics.get("sys_ram_used_mb"):
            vm_used_pct = metrics["sys_ram_used_mb"] * 1024 * 1024 / psutil.virtual_memory().total
            if vm_used_pct >= RAM_WATERMARK:
                ram_warning = f"post-run RAM {vm_used_pct*100:.1f}% >= {RAM_WATERMARK*100:.0f}%"
        print(f"[bench] batch_size={bs} num_workers={w} duration={dur:.2f}s imgs/sec={ips:.1f} (avg over {num_epochs} epoch(s)){ram_str}{cuda_str}")
        row = {
            "batch_size": bs,
            "num_workers": w,
            "seconds": dur,
            "imgs_per_sec": ips,
            "ram_mb": metrics.get("ram_mb", ""),
            "vms_mb": metrics.get("vms_mb", ""),
            "sys_ram_used_mb": metrics.get("sys_ram_used_mb", ""),
            "sys_ram_avail_mb": metrics.get("sys_ram_avail_mb", ""),
            "cpu_percent": metrics.get("cpu_percent", ""),
            "cuda_alloc_mb": metrics.get("cuda_alloc_mb", ""),
            "cuda_reserved_mb": metrics.get("cuda_reserved_mb", ""),
            "cuda_peak_mb": metrics.get("cuda_peak_mb", ""),
            "warning": ram_warning,
        }
        results.append(row)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()

    # Plot results
    if results:
        plt.figure(figsize=(7, 5))
        markers = ["o", "s", "^", "D", "x", "+"]
        for idx, w in enumerate(unique_workers):
            xs = []
            ys = []
            for r in results:
                if r["num_workers"] != w:
                    continue
                if r["imgs_per_sec"] is None or r["imgs_per_sec"] == "":
                    continue
                xs.append(r["batch_size"])
                ys.append(r["imgs_per_sec"])
            if xs:
                plt.plot(xs, ys, marker=markers[idx % len(markers)], label=f"workers={w}")
        plt.xlabel("batch_size")
        plt.ylabel("images/sec")
        plt.title("Throughput vs batch_size/num_workers (1 epoch)")
        plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        plt.legend()
        out_path = Path(args.config).with_name("bench_epoch_throughput.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"[bench] Saved plot to {out_path}")

        # Print tabular summary
        print("\n[bench] Summary:")
        for r in results:
            sec = r["seconds"]
            ips = r["imgs_per_sec"]
            sec_str = f"{sec:.2f}" if isinstance(sec, (int, float)) else f"{sec}"
            ips_str = f"{ips:.1f}" if isinstance(ips, (int, float)) else f"{ips}"
            warn_str = f" | warning={r['warning']}" if r.get("warning") else ""
            print(
                f"batch_size={r['batch_size']:>3} | workers={r['num_workers']:>2} | "
                f"seconds={sec_str} | imgs/sec={ips_str}{warn_str}"
            )

        # Save CSV
        import csv

        csv_path = Path(args.config).with_name("bench_epoch_results.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "batch_size",
                    "num_workers",
                    "seconds",
                    "imgs_per_sec",
                    "ram_mb",
                    "vms_mb",
                    "sys_ram_used_mb",
                    "sys_ram_avail_mb",
                    "cpu_percent",
                    "cuda_alloc_mb",
                    "cuda_reserved_mb",
                    "cuda_peak_mb",
                    "warning",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"[bench] Saved results CSV to {csv_path}")


if __name__ == "__main__":
    main()
