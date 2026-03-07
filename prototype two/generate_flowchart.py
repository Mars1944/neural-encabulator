"""
Generate a PNG flowchart of the project code structure.

Usage:
  python generate_flowchart.py [--out flowchart.png]

Requires the `graphviz` Python package and Graphviz binaries on PATH.
If graphviz is not installed, the script will print a helpful message.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict


def _ensure_dot(dot_path: str | None = None) -> bool:
    """Ensure Graphviz 'dot' is reachable. Optionally prepend a provided path or known default installs."""
    # If explicitly provided
    if dot_path:
        p = Path(dot_path)
        if p.exists():
            os.environ["PATH"] = f"{p.parent}{os.pathsep}{os.environ.get('PATH', '')}"
            os.environ["GRAPHVIZ_DOT"] = str(p)
    # If GRAPHVIZ_DOT is set, respect it
    env_dot = os.environ.get("GRAPHVIZ_DOT", None)
    if env_dot and Path(env_dot).exists():
        os.environ["PATH"] = f"{Path(env_dot).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    # Probe common Windows install locations if still not found
    if shutil.which("dot") is None:
        common_paths = [
            Path(r"C:\Program Files\Graphviz\bin\dot.exe"),
            Path(r"C:\Program Files (x86)\Graphviz\bin\dot.exe"),
        ]
        for cp in common_paths:
            if cp.exists():
                os.environ["PATH"] = f"{cp.parent}{os.pathsep}{os.environ.get('PATH', '')}"
                os.environ["GRAPHVIZ_DOT"] = str(cp)
                break
    return shutil.which("dot") is not None


def build_graph() -> "graphviz.Digraph":
    import graphviz  # type: ignore

    g = graphviz.Digraph("neural_encabulator", format="png")
    g.graph_attr["dpi"] = "600"  # Increase PNG resolution
    g.attr(rankdir="LR", fontsize="10", labelloc="t", label="Prototype Two: Code Structure")

    # Clusters for readability
    with g.subgraph(name="cluster_config") as c:
        c.attr(label="Config + Setup", style="dashed")
        c.node("config.json", shape="note")
        c.node("common.ConfigManager", label="common.py\nConfigManager", shape="box")
        c.node("common.PathResolver", label="common.py\nPathResolver", shape="box")
        c.node("setup_folders", label="setup_folders.py", shape="box")
        c.edge("config.json", "common.ConfigManager")
        c.edge("common.ConfigManager", "common.PathResolver")
        c.edge("common.PathResolver", "setup_folders")

    with g.subgraph(name="cluster_model") as c:
        c.attr(label="Model + Optim", style="dashed")
        c.node("cnn_model", label="cnn_model.py\n(SimpleCNN / MultiTaskCNN)", shape="box")
        c.node("cnn_optim", label="cnn_optim.py\n(CnnOptim)", shape="box")
        c.edge("cnn_model", "cnn_optim", style="dotted")

    with g.subgraph(name="cluster_data") as c:
        c.attr(label="Data", style="dashed")
        c.node("data_loader", label="data_loader.py\n(ImageFolderDataset /\nPairedFlowDataset)", shape="box")
        c.node("vector_field_data", label="vector_field_data.py", shape="box")

    with g.subgraph(name="cluster_train") as c:
        c.attr(label="Training", style="dashed")
        c.node("train.py", shape="box")
        c.node("main.ModelManager", label="main.py\nModelManager", shape="box")
        c.node("trainer.Trainer", label="trainer.py\nTrainer", shape="box")
        c.edge("train.py", "main.ModelManager")
        c.edge("main.ModelManager", "cnn_model")
        c.edge("main.ModelManager", "trainer.Trainer")
        c.edge("trainer.Trainer", "cnn_optim")
        c.edge("trainer.Trainer", "data_loader")

    with g.subgraph(name="cluster_infer") as c:
        c.attr(label="Inference + Reports", style="dashed")
        c.node("infer.py", shape="box")
        c.node("reports.py", shape="box")
        c.node("heatmap_viewer.py", shape="box")
        c.node("postprocess_plots.py", shape="box")
        c.edge("infer.py", "reports.py", style="dotted")
        c.edge("infer.py", "heatmap_viewer.py", style="dotted")
        c.edge("reports.py", "postprocess_plots.py", style="dotted")

    with g.subgraph(name="cluster_pipeline") as c:
        c.attr(label="Pipeline Orchestration", style="dashed")
        c.node("master_control.py", shape="box")
        c.edge("master_control.py", "train.py")
        c.edge("master_control.py", "infer.py")
        c.edge("master_control.py", "reports.py")

    # Cross-cluster edges
    g.edge("common.ConfigManager", "main.ModelManager")
    g.edge("common.ConfigManager", "train.py")
    g.edge("common.ConfigManager", "infer.py")
    g.edge("common.ConfigManager", "reports.py")
    g.edge("common.PathResolver", "data_loader", style="dotted")
    g.edge("common.PathResolver", "vector_field_data", style="dotted")
    g.edge("trainer.Trainer", "infer.py", style="dotted", label="checkpoint")

    return g


def build_detailed_graphs() -> Dict[str, "graphviz.Digraph"]:
    import graphviz  # type: ignore

    graphs: Dict[str, "graphviz.Digraph"] = {}

    # Config/Setup
    cfg = graphviz.Digraph("config_setup", format="png")
    cfg.graph_attr["dpi"] = "600"
    cfg.attr(rankdir="TB", fontsize="10", labelloc="t", label="Config + Setup")
    cfg.node("config.json", shape="note")
    cfg.node("ConfigManager", label="common.ConfigManager", shape="box")
    cfg.node("PathResolver", label="common.PathResolver", shape="box")
    cfg.node("setup_folders", label="setup_folders.py", shape="box")
    cfg.edge("config.json", "ConfigManager")
    cfg.edge("ConfigManager", "PathResolver")
    cfg.edge("PathResolver", "setup_folders")
    graphs["config_setup"] = cfg

    # Training
    tr = graphviz.Digraph("training", format="png")
    tr.graph_attr["dpi"] = "600"
    tr.attr(rankdir="LR", fontsize="10", labelloc="t", label="Training")
    tr.node("train.py", shape="box")
    tr.node("ModelManager", label="main.ModelManager\n(builds model)", shape="box")
    tr.node("Trainer", label="trainer.Trainer\n(train/val loop)", shape="box")
    tr.node("cnn_model", label="cnn_model.py\nSimpleCNN/MultiTaskCNN", shape="box")
    tr.node("cnn_optim", label="cnn_optim.py\nCnnOptim\n(opt + scheduler)", shape="box")
    tr.node("data_loader", label="data_loader.py\nImageFolderDataset/\nPairedFlowDataset", shape="box")
    tr.node("PairedFlowDataset", label="PairedFlowDataset\n(normalize targets,\nlabels/flow CSV)", shape="box", style="rounded")
    tr.node("ImageFolderDataset", label="ImageFolderDataset\n(class folders)", shape="box", style="rounded")
    tr.node("checkpoints", label="runs/checkpoints", shape="folder")
    tr.edge("train.py", "ModelManager", label="cfg/model")
    tr.edge("ModelManager", "cnn_model", label="constructs")
    tr.edge("ModelManager", "Trainer", label="model + device")
    tr.edge("Trainer", "cnn_optim", label="build opt/sched")
    tr.edge("Trainer", "data_loader", label="build loaders")
    tr.edge("data_loader", "PairedFlowDataset", style="dotted")
    tr.edge("data_loader", "ImageFolderDataset", style="dotted")
    tr.edge("Trainer", "checkpoints", label="torch.save best/final")
    tr.edge("Trainer", "cnn_model", style="dotted", label="forward/backward")
    tr.edge("Trainer", "PairedFlowDataset", style="dotted", label="multitask data")
    graphs["training"] = tr

    # Inference/Reports
    inf = graphviz.Digraph("inference", format="png")
    inf.graph_attr["dpi"] = "600"
    inf.attr(rankdir="LR", fontsize="10", labelloc="t", label="Inference + Reports")
    inf.node("infer.py", shape="box")
    inf.node("InferenceRunner", label="infer.py\nInferenceRunner", shape="box", style="rounded")
    inf.node("heatmap_viewer.py", shape="box")
    inf.node("postprocess_plots.py", shape="box")
    inf.node("reports.py", shape="box")
    inf.node("preds", label="predictions/heatmaps", shape="folder")
    inf.edge("infer.py", "InferenceRunner", label="build model/load ckpt")
    inf.edge("InferenceRunner", "preds", label="logits/probs/reg")
    inf.edge("preds", "postprocess_plots.py", style="dotted", label="probs/heatmaps")
    inf.edge("preds", "heatmap_viewer.py", style="dotted", label="stitched view")
    inf.edge("infer.py", "reports.py", style="dotted", label="summaries/CM")
    graphs["inference"] = inf

    # Data
    data = graphviz.Digraph("data", format="png")
    data.graph_attr["dpi"] = "600"
    data.attr(rankdir="LR", fontsize="10", labelloc="t", label="Data")
    data.node("data_loader", label="data_loader.py\n(datasets/transforms)", shape="box")
    data.node("PairedFlowDataset_d", label="PairedFlowDataset\nCSV+image pairing,\nimpute, normalize targets", shape="box", style="rounded")
    data.node("ImageFolderDataset_d", label="ImageFolderDataset\nclass dirs, augment", shape="box", style="rounded")
    data.node("vector_field_data", label="vector_field_data.py\n(load tiles CHW)", shape="box")
    data.node("PathResolver", label="common.PathResolver\n(anchor paths)", shape="box")
    data.edge("PathResolver", "data_loader", style="dotted", label="anchor roots")
    data.edge("PathResolver", "vector_field_data", style="dotted")
    data.edge("data_loader", "PairedFlowDataset_d", style="dotted")
    data.edge("data_loader", "ImageFolderDataset_d", style="dotted")
    graphs["data"] = data

    return graphs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PNG flowchart of the project code structure.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.json")),
        help="Config path (used to anchor default output location)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path. If omitted, saves alongside the config as flowchart.png. Relative paths are anchored to the config directory.",
    )
    parser.add_argument(
        "--dot-path",
        type=str,
        default=None,
        help="Optional path to dot.exe (e.g., C:\\Program Files\\Graphviz\\bin\\dot.exe) if not on PATH",
    )
    parser.add_argument(
        "--no-detailed",
        dest="detailed",
        action="store_false",
        help="Skip detailed per-section flowcharts (default: generate them).",
    )
    parser.set_defaults(detailed=True)
    args = parser.parse_args()

    if not _ensure_dot(args.dot_path):
        print(
            "[flowchart] Graphviz 'dot' not found. Install Graphviz and ensure dot is on PATH, "
            "or pass --dot-path to dot.exe."
        )
        return

    cfg_path = Path(args.config)
    base_dir = cfg_path.parent if cfg_path.exists() else Path(__file__).parent
    if args.out is None:
        out_path = base_dir / "flowchart.png"
    else:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = base_dir / out_path

    try:
        g = build_graph()
    except ModuleNotFoundError:
        print("[flowchart] graphviz is required. Install with: pip install graphviz (and ensure Graphviz binaries are on PATH).")
        return
    except Exception as e:
        print(f"[flowchart] Failed to build graph: {e}")
        return

    try:
        g.render(str(out_path.with_suffix("")), cleanup=True)
        print(f"[flowchart] Saved flowchart to: {out_path}")
    except Exception as e:
        print(f"[flowchart] Failed to render flowchart: {e}")
        return

    if args.detailed:
        try:
            details = build_detailed_graphs()
            for name, dg in details.items():
                detail_path = out_path.with_name(f"{out_path.stem}-{name}{out_path.suffix}")
                dg.render(str(detail_path.with_suffix("")), cleanup=True)
                print(f"[flowchart] Saved detailed flowchart to: {detail_path}")
        except Exception as e:
            print(f"[flowchart] Failed to render detailed flowcharts: {e}")


if __name__ == "__main__":
    main()
