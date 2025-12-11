"""
Quick environment checker: verifies key Python packages and reports GPU visibility for PyTorch.

Usage:
  python check_env.py
"""

from __future__ import annotations

import importlib
import sys


def check_pkg(name: str) -> tuple[bool, str | None]:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", None)
        return True, ver
    except Exception as e:
        return False, str(e)


def main() -> None:
    packages = ["torch", "numpy", "opencv", "PIL", "matplotlib", "tqdm", "psutil", "scipy", "graphviz"]
    print("[check_env] Package versions:")
    for pkg in packages:
        ok, ver = check_pkg(pkg)
        if ok:
            print(f"  {pkg}: {ver or 'installed'}")
        else:
            print(f"  {pkg}: NOT FOUND ({ver})")

    try:
        import torch

        print("\n[check_env] PyTorch build:", torch.__version__)
        print("  CUDA available:", torch.cuda.is_available())
        print("  torch.version.cuda:", torch.version.cuda)
        if torch.cuda.is_available():
            print("  GPU count:", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                print(f"    cuda:{i} -> {torch.cuda.get_device_name(i)}")
            try:
                print("  Current device:", torch.cuda.current_device())
            except Exception:
                pass
    except Exception as e:
        print(f"[check_env] PyTorch not usable: {e}")

    # Graphviz binary check (for flowchart generation)
    try:
        import shutil

        dot_path = shutil.which("dot")
        if dot_path:
            print(f"\n[check_env] Graphviz dot found: {dot_path}")
        else:
            print("\n[check_env] Graphviz dot not found. Install Graphviz and ensure 'dot' is on PATH.")
    except Exception:
        pass

    print("\n[check_env] Python:", sys.version)
    if not sys.version.startswith("3.12"):
        print("[check_env][WARN] Python version is not 3.12; current:", sys.version.split()[0])

    print("\n[check_env] Suggested pip installs (CPU build):")
    print('  pip install torch torchvision torchaudio numpy opencv-python pillow matplotlib tqdm psutil scipy')
    print("Optional for flowchart generation:")
    print('  pip install graphviz  # requires Graphviz binaries on PATH (dot)')
    print("For CUDA builds, use the PyTorch index URL for your CUDA version, e.g.:")
    print('  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio')


if __name__ == "__main__":
    main()
