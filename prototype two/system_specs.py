"""
System specification reporter for the prototype two project.

Usage:
  python system_specs.py

Outputs CPU, RAM, GPU, and storage details, with optional speeds when available.
Relies on standard libraries plus optional psutil, torch, and nvidia-smi/PowerShell queries.
"""

from __future__ import annotations

import csv
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _run_cmd(cmd: List[str]) -> tuple[bool, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        return True, out.strip()
    except Exception as e:
        return False, str(e)


def cpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.system(),
        "release": platform.release(),
        "processor": platform.processor() or platform.machine(),
        "os_bits": platform.architecture()[0],
    }
    try:
        import psutil  # type: ignore

        cpu_freq = psutil.cpu_freq()
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True)
        if cpu_freq:
            info["cpu_freq_current_mhz"] = cpu_freq.current
            info["cpu_freq_max_mhz"] = cpu_freq.max
    except Exception:
        pass
    # Windows: use WMI for base/boost clock details if available
    if platform.system().lower().startswith("win"):
        ok, out = _run_cmd(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Processor | Select-Object Name,MaxClockSpeed,CurrentClockSpeed | ConvertTo-Json",
            ]
        )
        if ok:
            try:
                data = json.loads(out)
                if isinstance(data, dict):
                    data = [data]
                if data:
                    info["wmi_cpu"] = data
                    # Populate base/max if not already set
                    if "cpu_freq_current_mhz" not in info and data[0].get("CurrentClockSpeed") is not None:
                        info["cpu_freq_current_mhz"] = data[0].get("CurrentClockSpeed")
                    if "cpu_freq_max_mhz" not in info and data[0].get("MaxClockSpeed") is not None:
                        info["cpu_freq_max_mhz"] = data[0].get("MaxClockSpeed")
            except Exception:
                pass
    return info


def ram_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["total_gb"] = round(vm.total / (1024**3), 2)
    except Exception:
        pass

    # Windows-only: query per-DIMM details via PowerShell/WMI as structured JSON
    if platform.system().lower().startswith("win"):
        ok, out = _run_cmd(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_PhysicalMemory | Select-Object Manufacturer,Speed,ConfiguredClockSpeed,Capacity | ConvertTo-Json",
            ]
        )
        if ok:
            try:
                data = json.loads(out)
                if isinstance(data, dict):
                    data = [data]
                modules = []
                for m in data:
                    cap_bytes = float(m.get("Capacity", 0.0) or 0.0)
                    modules.append(
                        {
                            "manufacturer": m.get("Manufacturer"),
                            "rated_speed_mhz": m.get("Speed"),
                            "configured_speed_mhz": m.get("ConfiguredClockSpeed"),
                            "capacity_gb": round(cap_bytes / (1024**3), 2),
                        }
                    )
                info["modules"] = modules
            except Exception as e:
                info["module_speeds_error"] = f"parse error: {e}"
        else:
            info["module_speeds_error"] = out
    return info


def gpu_info() -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []

    # First, try PyTorch (gives reliable memory totals)
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    {
                        "source": "torch",
                        "index": i,
                        "name": props.name,
                        "total_mem_gb": round(props.total_memory / (1024**3), 2),
                        "cuda_cc": f"{props.major}.{props.minor}",
                    }
                )
    except Exception:
        pass

    # Next, try nvidia-smi for memory clocks (if NVIDIA GPU)
    ok, out = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,clocks.mem,clocks.gr,clocks.max.memory,clocks.max.graphics",
            "--format=csv,noheader,nounits",
        ]
    )
    if ok:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                try:
                    idx = int(parts[0])
                except Exception:
                    idx = None
                gpus.append(
                    {
                        "source": "nvidia-smi",
                        "index": idx,
                        "name": parts[1],
                        "total_mem_mb": parts[2],
                        "free_mem_mb": parts[3],
                        "mem_clock_current_mhz": parts[4],
                        "graphics_clock_current_mhz": parts[5],
                        "mem_clock_max_mhz": parts[6],
                        "graphics_clock_max_mhz": parts[7] if len(parts) > 7 else None,
                        "descriptions": {
                            "total_mem_mb": "Total VRAM (MB)",
                            "free_mem_mb": "Free VRAM (MB)",
                            "mem_clock_current_mhz": "Memory clock current (MHz)",
                            "graphics_clock_current_mhz": "Graphics clock current (MHz)",
                            "mem_clock_max_mhz": "Memory clock max (MHz)",
                            "graphics_clock_max_mhz": "Graphics clock max (MHz)",
                        },
                    }
                )

    # Windows WMI fallback (adapter name and memory)
    if not gpus and platform.system().lower().startswith("win"):
        ok, out = _run_cmd(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | Format-Table -AutoSize",
            ]
        )
        if ok:
            gpus.append({"source": "powershell", "raw": out})
    return gpus


def disk_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        total, used, free = shutil.disk_usage(".")
        info["disk_total_gb"] = round(total / (1024**3), 2)
        info["disk_free_gb"] = round(free / (1024**3), 2)
    except Exception:
        pass
    return info


def pretty_print(title: str, obj: Any) -> None:
    print(f"\n== {title} ==")
    # Special pretty layout for RAM modules
    if title.upper().startswith("RAM") and isinstance(obj, dict):
        total = obj.get("total_gb", None)
        if total is not None:
            print(f"Total RAM (GB): {total}")
        modules = obj.get("modules", [])
        if modules and isinstance(modules, list):
            print("\nmodule_speeds:")
            headers = ["manufacturer", "rated_speed_mhz", "configured_speed_mhz", "capacity_gb"]
            header_labels = ["Manufacturer", "Speed (MHz)", "Configured (MHz)", "Capacity (GB)"]
            col_widths = {h: len(h) for h in headers}
            for m in modules:
                if not isinstance(m, dict):
                    continue
                for h in headers:
                    val = "" if m.get(h) is None else str(m.get(h))
                    col_widths[h] = max(col_widths[h], len(val))
            spacer = "    "
            header_line = "  " + spacer.join([label.ljust(col_widths[h]) for label, h in zip(header_labels, headers)])
            print(header_line)
            print("  " + spacer.join(["-" * col_widths[h] for h in headers]))
            for m in modules:
                if not isinstance(m, dict):
                    continue
                row = []
                for h in headers:
                    val = "" if m.get(h) is None else str(m.get(h))
                    row.append(val.ljust(col_widths[h]))
                print("  " + spacer.join(row))
        # Print any remaining fields not covered above
        for k, v in obj.items():
            if k in ("modules", "total_gb"):
                continue
            print(f"{k}: {v}")
        return
    # Special pretty layout for GPU entries
    if title.upper().startswith("GPU") and isinstance(obj, list):
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                print(f"[{i}] {item}")
                continue
            print(f"[GPU {i}]")
            name = item.get("name")
            src = item.get("source")
            idx = item.get("index", None)
            if name:
                print(f"  Name: {name}")
            if idx is not None:
                print(f"  Device index: {idx}")
            if src:
                print(f"  Source: {src}")
            # Prefer human labels if provided
            labels = item.get("descriptions", {})
            for key, label in [
                ("total_mem_mb", "Total VRAM (MB)"),
                ("free_mem_mb", "Free VRAM (MB)"),
                ("total_mem_gb", "Total VRAM (GB)"),
                ("mem_clock_current_mhz", "Memory clock current (MHz)"),
                ("graphics_clock_current_mhz", "Graphics clock current (MHz)"),
                ("mem_clock_max_mhz", "Memory clock max (MHz)"),
                ("graphics_clock_max_mhz", "Graphics clock max (MHz)"),
                ("cuda_cc", "CUDA compute capability"),
            ]:
                if key in item:
                    lbl = labels.get(key, label) if isinstance(labels, dict) else label
                    print(f"  {lbl}: {item.get(key)}")
            # Print any leftover keys
            for k, v in item.items():
                if k in {"name", "source", "index", "descriptions", "total_mem_mb", "free_mem_mb", "total_mem_gb", "mem_clock_current_mhz", "graphics_clock_current_mhz", "mem_clock_max_mhz", "graphics_clock_max_mhz", "cuda_cc"}:
                    continue
                print(f"  {k}: {v}")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{k}: {v}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            print(f"[{i}]")
            if isinstance(item, dict):
                for k, v in item.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {item}")
    else:
        print(obj)


def main() -> None:
    print("[system_specs] Python:", sys.version.split()[0])
    cpu = cpu_info()
    ram = ram_info()
    gpu = gpu_info()
    disk = disk_info()
    pretty_print("CPU Specs", cpu)
    pretty_print("RAM Specs", ram)
    pretty_print("GPU Specs", gpu)
    pretty_print("Disk (current drive)", disk)

    # Save a flat CSV snapshot of all sections
    specs = {
        "CPU Specs": cpu,
        "RAM Specs": ram,
        "GPU Specs": gpu,
        "Disk": disk,
        "Python": {"version": sys.version},
    }
    rows: List[Dict[str, str]] = []

    def _flatten(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_prefix = f"{prefix}.{k}" if prefix else str(k)
                _flatten(new_prefix, v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_prefix = f"{prefix}[{i}]"
                _flatten(new_prefix, v)
        else:
            rows.append({"section": prefix, "value": str(obj)})

    for section, content in specs.items():
        _flatten(section, content)

    csv_path = Path(__file__).with_name("system_specs.csv")
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["section", "value"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"[system_specs] CSV saved to: {csv_path}")
    except Exception as e:
        print(f"[system_specs] Failed to write CSV: {e}")


if __name__ == "__main__":
    main()
