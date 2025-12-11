"""
Master control program to run train -> infer -> reports in sequence.

Usage:
  python master_control.py --config config.json

The config file should include an "mcp" section with booleans:
{
  "mcp": {
    "run_train": true,
    "run_infer": true,
    "run_reports": true
  }
}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run_step(label: str, cmd: list[str]) -> None:
    print(f"[mcp] starting {label}: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with code {result.returncode}")
    print(f"[mcp] finished {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Master control for train -> infer -> reports")
    default_cfg = Path(__file__).resolve().with_name("config.json")
    parser.add_argument("--config", type=str, default=str(default_cfg), help="Path to config.json")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    mcp_cfg = cfg.get("mcp", {})
    run_train = bool(mcp_cfg.get("run_train", True))
    run_infer = bool(mcp_cfg.get("run_infer", True))
    run_reports = bool(mcp_cfg.get("run_reports", True))
    run_bench = bool(mcp_cfg.get("run_bench_optimize", False))

    py = sys.executable
    base_dir = Path(__file__).resolve().parent

    if run_train:
        _run_step("train", [py, str(base_dir / "train.py"), "--config", str(cfg_path)])
    else:
        print("[mcp] skipping train (run_train=false)")

    if run_infer:
        _run_step("infer", [py, str(base_dir / "infer.py"), "--config", str(cfg_path)])
    else:
        print("[mcp] skipping infer (run_infer=false)")

    if run_reports:
        _run_step("reports", [py, str(base_dir / "reports.py"), "--config", str(cfg_path)])
    else:
        print("[mcp] skipping reports (run_reports=false)")

    if run_bench:
        _run_step("bench_optimize", [py, str(base_dir / "bench_optimize.py"), "--config", str(cfg_path)])
    else:
        print("[mcp] skipping bench_optimize (run_bench_optimize=false)")

    print("[mcp] pipeline complete")


if __name__ == "__main__":
    main()
