"""
Analyze one or more bench_epoch_results.csv files and summarize throughput.

Usage:
  python analyze_bench.py --csv bench_epoch_results.csv --out bench_analysis.png
  python analyze_bench.py --csv file1.csv file2.csv file3.csv --out bench_analysis.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt  # type: ignore


def _to_float(v: str) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def load_rows(paths: List[Path]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for path in paths:
        rows.extend(load_one(path))
    return rows


def load_one(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if not path.exists():
        print(f"[analyze] skipping missing CSV: {path}")
        return rows
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
                        "ram_mb": _to_float(r.get("ram_mb", "")) or 0.0,
                        "cuda_peak_mb": _to_float(r.get("cuda_peak_mb", "")) or 0.0,
                        "source": path.name,
                    }
                )
            except Exception:
                continue
    return rows


def summarize(rows: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    if not rows:
        return {}, {}, {}
    best = max(rows, key=lambda r: r["imgs_per_sec"])
    by_worker: Dict[int, Dict[str, float]] = {}
    by_batch: Dict[int, Dict[str, float]] = {}
    # Best per worker
    grouped: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for r in rows:
        grouped[int(r["num_workers"])].append(r)
    for w, grp in grouped.items():
        top = max(grp, key=lambda r: r["imgs_per_sec"])
        by_worker[w] = top
    # Best per batch
    grouped_b: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for r in rows:
        grouped_b[int(r["batch_size"])].append(r)
    for b, grp in grouped_b.items():
        top = max(grp, key=lambda r: r["imgs_per_sec"])
        by_batch[b] = top
    return best, by_worker, by_batch


def plot(rows: List[Dict[str, float]], out_path: Path) -> None:
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
    try:
        ticks = sorted({int(r["batch_size"]) for r in rows})
        plt.xticks(ticks)
    except Exception:
        pass
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[analyze] Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze bench_epoch_results.csv throughput.")
    default_list = [
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
    default_csv = Path(__file__).resolve().with_name("bench_epoch_results.csv")
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        default=[str(default_csv)] + default_list,
        help="Path(s) to bench_epoch_results.csv (one or more). If omitted, uses built-in defaults.",
    )
    parser.add_argument("--out", type=str, default="bench_analysis.png", help="Output plot path")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_paths = []
    for p in args.csv:
        path = Path(p)
        if not path.is_absolute():
            path = base_dir / path
        csv_paths.append(path)

    rows = load_rows(csv_paths)
    if not rows:
        raise FileNotFoundError(f"No valid CSV rows found in: {', '.join(args.csv)}")
    best, by_worker, by_batch = summarize(rows)

    print(f"[analyze] Rows: {len(rows)}")
    if best:
        print(
            f"[best overall] bs={int(best['batch_size'])} workers={int(best['num_workers'])} "
            f"imgs/sec={best['imgs_per_sec']:.1f} seconds={best['seconds']:.2f}"
        )
    if by_worker:
        print("[best per worker]:")
        for w in sorted(by_worker):
            r = by_worker[w]
            print(f"  workers={w}: bs={int(r['batch_size'])} imgs/sec={r['imgs_per_sec']:.1f} seconds={r['seconds']:.2f}")
    if by_batch:
        print("[best per batch size]:")
        for b in sorted(by_batch):
            r = by_batch[b]
            print(f"  bs={b}: workers={int(r['num_workers'])} imgs/sec={r['imgs_per_sec']:.1f} seconds={r['seconds']:.2f}")

    # Combined plot
    plot(rows, Path(args.out))

    # Per-file plots
    for csv_path in csv_paths:
        single_rows = load_one(csv_path)
        if not single_rows:
            continue
        single_out = csv_path.with_name(f"{csv_path.stem}_analysis.png")
        plot(single_rows, single_out)


if __name__ == "__main__":
    main()
