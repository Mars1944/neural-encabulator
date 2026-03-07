"""
Aggregate multiple bench_epoch_results.csv files and find the best worker/batch settings.

Usage:
  python aggregate_bench.py --csvs file1.csv file2.csv ... --out agg_plot.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore


def _to_float(v: str) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def load_rows(paths: List[Path]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for path in paths:
        if not path.exists():
            print(f"[aggregate] skipping missing CSV: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    rows.append(
                        {
                            "batch_size": float(r.get("batch_size", 0)),
                            "num_workers": float(r.get("num_workers", 0)),
                            "seconds": _to_float(r.get("seconds", "")) or 0.0,
                            "imgs_per_sec": _to_float(r.get("imgs_per_sec", "")) or 0.0,
                            "sys_ram_used_mb": _to_float(r.get("sys_ram_used_mb", "")) or None,
                            "vms_mb": _to_float(r.get("vms_mb", "")) or None,
                            "source": path.name,
                        }
                    )
                except Exception:
                    continue
    return rows


def find_best(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return max(rows, key=lambda r: r["imgs_per_sec"]) if rows else {}


def plot(rows: List[Dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    plt.figure(figsize=(16, 10))
    workers = sorted({int(r["num_workers"]) for r in rows})
    markers = ["o", "s", "^", "D", "x", "+"]
    cmap = plt.get_cmap("tab20", max(len(workers), 1))
    for idx, w in enumerate(workers):
        xs = [r["batch_size"] for r in rows if int(r["num_workers"]) == w]
        ys = [r["imgs_per_sec"] for r in rows if int(r["num_workers"]) == w]
        color = cmap(idx)
        plt.scatter(xs, ys, marker=markers[idx % len(markers)], color=color, label=f"workers={w}")

    # Highlight best point
    best = max(rows, key=lambda r: r["imgs_per_sec"])
    plt.scatter(
        [best["batch_size"]],
        [best["imgs_per_sec"]],
        color="gold",
        edgecolors="black",
        s=120,
        label=f"best bs={int(best['batch_size'])}, workers={int(best['num_workers'])}",
        zorder=6,
    )
    plt.axhline(y=best["imgs_per_sec"], color="gold", linestyle="--", linewidth=1.2, alpha=0.8, label=f"best imgs/sec={best['imgs_per_sec']:.1f}")
    plt.annotate(
        f"bs={int(best['batch_size'])}, workers={int(best['num_workers'])}\n{best['imgs_per_sec']:.1f} img/s",
        (best["batch_size"], best["imgs_per_sec"]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
    )
    plt.xlabel("batch_size")
    plt.ylabel("images/sec")
    try:
        ticks = sorted({int(r["batch_size"]) for r in rows})
        plt.xticks(ticks)
    except Exception:
        pass
    plt.title("Aggregated throughput vs batch_size/num_workers")
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[aggregate] Saved plot to {out_path}")


def plot_ram(rows: List[Dict[str, float]], out_path: Path) -> None:
    ram_rows = [r for r in rows if "sys_ram_used_mb" in r and r["sys_ram_used_mb"]]
    if not ram_rows:
        print("[aggregate] No RAM data to plot.")
        return
    plt.figure(figsize=(16, 10))
    workers = sorted({int(r["num_workers"]) for r in ram_rows})
    markers = ["o", "s", "^", "D", "x", "+"]
    cmap = plt.get_cmap("tab20", max(len(workers), 1))
    for idx, w in enumerate(workers):
        xs = [r["batch_size"] for r in ram_rows if int(r["num_workers"]) == w]
        ys = [r["sys_ram_used_mb"] for r in ram_rows if int(r["num_workers"]) == w]
        color = cmap(idx)
        plt.scatter(xs, ys, marker=markers[idx % len(markers)], color=color, label=f"workers={w}")
    best = max(ram_rows, key=lambda r: r["sys_ram_used_mb"])
    plt.scatter(
        [best["batch_size"]],
        [best["sys_ram_used_mb"]],
        color="gold",
        edgecolors="black",
        s=120,
        label=f"max ram bs={int(best['batch_size'])}, workers={int(best['num_workers'])}",
        zorder=6,
    )
    plt.annotate(
        f"bs={int(best['batch_size'])}, workers={int(best['num_workers'])}\nRAM={best['sys_ram_used_mb']:.1f} MB",
        (best["batch_size"], best["sys_ram_used_mb"]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
    )
    plt.xlabel("batch_size")
    plt.ylabel("RAM (MB)")
    try:
        ticks = sorted({int(r["batch_size"]) for r in ram_rows})
        plt.xticks(ticks)
    except Exception:
        pass
    plt.title("RAM usage vs batch_size/num_workers")
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[aggregate] Saved RAM plot to {out_path}")


def plot_vms(rows: List[Dict[str, float]], out_path: Path) -> None:
    vms_rows = [r for r in rows if "vms_mb" in r and r["vms_mb"]]
    if not vms_rows:
        print("[aggregate] No VMS data to plot.")
        return
    plt.figure(figsize=(16, 10))
    workers = sorted({int(r["num_workers"]) for r in vms_rows})
    markers = ["o", "s", "^", "D", "x", "+"]
    cmap = plt.get_cmap("tab20", max(len(workers), 1))
    for idx, w in enumerate(workers):
        xs = [r["batch_size"] for r in vms_rows if int(r["num_workers"]) == w]
        ys = [r["vms_mb"] for r in vms_rows if int(r["num_workers"]) == w]
        color = cmap(idx)
        plt.scatter(xs, ys, marker=markers[idx % len(markers)], color=color, label=f"workers={w}")
    best = max(vms_rows, key=lambda r: r["vms_mb"])
    plt.scatter(
        [best["batch_size"]],
        [best["vms_mb"]],
        color="gold",
        edgecolors="black",
        s=120,
        label=f"max VMS bs={int(best['batch_size'])}, workers={int(best['num_workers'])}",
        zorder=6,
    )
    plt.annotate(
        f"bs={int(best['batch_size'])}, workers={int(best['num_workers'])}\nVMS={best['vms_mb']:.1f} MB",
        (best["batch_size"], best["vms_mb"]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
    )
    plt.xlabel("batch_size")
    plt.ylabel("VMS (MB)")
    try:
        ticks = sorted({int(r["batch_size"]) for r in vms_rows})
        plt.xticks(ticks)
    except Exception:
        pass
    plt.title("Virtual memory (VMS) vs batch_size/num_workers")
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[aggregate] Saved VMS plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate bench_epoch CSVs and find best settings.")
    # Default CSV list you can edit here; used when --csv is omitted.
    default_csvs = [
        "set_zero_bench_epoch_results.csv",
        "set_one_bench_epoch_results.csv",
        "set_two_bench_epoch_results.csv",
        "set_three_bench_epoch_results.csv",
        "set_four_bench_epoch_results.csv",
        "set_five_bench_epoch_results.csv",
        "set_six_bench_epoch_results.csv",
        "set_seven_bench_epoch_results.csv",
        "set_eight_bench_epoch_results.csv",
    ]
    # Base directory to resolve default CSVs
    default_base = Path(r"C:\Users\Owner\Documents\Grad School\Fall 2025\ME 521 - CPTS 534 Project\neural-encabulator\prototype two")
    parser.add_argument(
        "--csv",
        nargs="+",
        default=None,
        help="List of bench_epoch_results.csv files. If omitted, uses the built-in default list above.",
    )
    parser.add_argument("--out", type=str, default=None, help="Output plot path (defaults to aggregate_bench.png under the CSV base directory)")
    args = parser.parse_args()

    csv_args = args.csv if args.csv is not None else default_csvs
    paths = [Path(p) if Path(p).is_absolute() else (default_base / p) for p in csv_args]
    rows = load_rows(paths)
    print(f"[aggregate] Loaded {len(rows)} rows from {len(paths)} file(s)")

    best = find_best(rows)
    if best:
        print(
            f"[best overall] bs={int(best['batch_size'])} workers={int(best['num_workers'])} "
            f"imgs/sec={best['imgs_per_sec']:.1f} seconds={best['seconds']:.2f} (source={best.get('source','')})"
        )
    else:
        print("[aggregate] No rows to analyze.")

    out_path = Path(args.out) if args.out is not None else (default_base / "aggregate_bench.png")
    plot(rows, out_path)
    ram_path = out_path.with_name(out_path.stem + "_ram" + out_path.suffix)
    plot_ram(rows, ram_path)
    vms_path = out_path.with_name(out_path.stem + "_vms" + out_path.suffix)
    plot_vms(rows, vms_path)


if __name__ == "__main__":
    main()
