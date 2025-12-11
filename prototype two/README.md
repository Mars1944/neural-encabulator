# Prototype Two - Usage Guide

## Setup
- Python 3.12+; install deps:  
  `pip install torch torchvision torchaudio numpy opencv-python pillow matplotlib tqdm psutil scipy graphviz`
- Install Graphviz system binaries (for flowcharts) and ensure `dot` is on PATH.

## Train
- Edit `config.json` (data paths, model/training settings). Multitask uses `train_flow_csv` + images.
- Run: `python train.py --config config.json`
- Checkpoints/history: `outputs_root/<mode>/training/` (best/final .pth, CSV, plot).

## Inference
- Run: `python infer.py --config config.json`
- Outputs: predictions/heatmaps/CM under `outputs_root`.
- Optional post-processing: `python postprocess_plots.py --config config.json --probs_npy <path> --field_path <field>`

## Pipeline Orchestration
- Run all steps: `python master_control.py --config config.json` (honors `mcp` flags for train/infer/reports).

## Benchmarks
- Sweep batch/workers: `python bench_epoch.py --config config.json --use-matrix`
- Aggregate plots: `python aggregate_bench.py --csvs <csvs> --out agg_plot.png`

## Flowcharts
- Generate overview + detailed charts: `python generate_flowchart.py`  
  Outputs next to `config.json` (use `--no-detailed` to skip extras).

## Config Reference (config.json)
- Paths/Meta: `_doc` (notes), `paths.outputs_root` (all outputs), `seed`, `device`, `model_type`.
- Data (image path is default): `train_image_dir`, `val_image_dir`, `test_image_dir`, `image_size`, `image_channels`, `image_mean/std`, `augment`; legacy vector keys: `train_field_path`, `test_field_path`, `tile_size/stride`, `add_magnitude`, `normalize`, `limit_tiles`.
- Model: `num_classes`, `conv_channels`, `kernel_size`, `pool_every`, `dropout`, `use_batchnorm`, `dilations`.
- Training: `task` (multitask/classification), `train_flow_csv`/`flow_csv` (paired CSV), `batch_size`, `num_workers`, `pin_memory`, `val_split`, `learning_rate`, `weight_decay`, `optimizer`, `scheduler` (+ `scheduler_params`), `max_epochs`, `loss_weight_cls`, `loss_weight_reg`, `debug_limit_training`/`debug_train_fraction`, `normalize_regression_targets`, `train_source_labels`/`val_source_labels`.
- Inference: `use_test`, `file_labels`, `file_summary`, `inference_tile_size/stride`, `recurse`, `generate_heatmap`, `generate_confusion_matrix`, `show_heatmap`, `heatmap_class`, `cm_csv/png`, `heatmaps_dir`, `stitched_heatmap`, `class_names`, `image_infer_batch_size`, `mixed_precision`, `channels_last`, `prefetch_workers`, `torch_compile`, `use_inference_mode`, `allow_tf32`, `use_opencv_loader`, `plot_top_n`, `image_include_classes`.
- Reports/Viewer: `predictions_csv`, `image_root`, `subdir`, `viewer_values_csv`, `viewer_title`, `viewer_cmap`, `viewer_no_show`, `viewer_save`, `viewer_rows`, `viewer_cols`.
- Notes: paths anchor to the config directory; `PathResolver`/`ensure_outputs_ready` create output folders. PairedFlowDataset can impute missing numeric fields and will log a CSV if many imputations occur. Training artifacts (history/plot/settings) live under `outputs_root/<mode>/training/`.
