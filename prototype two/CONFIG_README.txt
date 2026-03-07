Prototype Two Config Guide (config.json)

This file explains each key in config.json, what it controls, and which code paths consume it.
Paths are resolved relative to the directory that contains config.json unless already absolute.
Folder creation is centralized via setup_folders/ensure_outputs_ready so entrypoints can create missing dirs automatically.

Top-level
- _doc: Optional notes for humans. Not used by code.
- paths.outputs_root: Base folder for generated artifacts (heatmaps, stitched images, confusion matrices, checkpoints, training history/plots). Used by common.PathResolver and ensure_outputs_ready/setup_folders across infer.py, trainer.py, reports.py.
- seed: Global RNG seed for Python, NumPy, and PyTorch. Set by common.ConfigManager.
- device: Preferred device: auto|cpu|cuda[:index]|mps. Selected by common.DeviceSelector. Printed and used in train.py and infer.py.
- model_type: Model family string. Current code uses "cnn" via cnn_model.SimpleCNN in main.ModelManager.

Section: data
- train_field_path: Vector-field training sources. String, list, or directory/glob. Used by trainer.Trainer to tile vector fields.
- test_field_path: Vector-field test/validation sources. String, list, or directory/glob. Used by infer.InferenceRunner (and Trainer as validation when present).
- train_data_kind: Data type for training: "vector" or "image". Controls Trainer loading path.
- test_data_kind: Data type for inference: "vector" or "image". Controls InferenceRunner path.
- train_image_dir: Root directory for image training. Expects subfolders per class, or a pointer .txt/.csv containing the real path. Used by image_data.ImageFolderDataset (Trainer) and PairedFlowDataset.
- val_image_dir: Optional root directory for image validation. If empty, Trainer splits train_image_dir by val_split.
- test_image_dir: Root directory for image inference (folder-per-class, but ground truth not required). Accepts pointer .txt/.csv containing the real path. Used by infer.run_image_inference.
- image_size: [H, W] resize for images. Used by image_data and image inference.
- image_channels: 1 or 3; converts images to grayscale or RGB and sets model input channels. Used by image_data, infer, and main.ModelManager.
- tile_size: [H, W] for vector tiling. Used by vector_field_data and consumed by trainer/infer.
- tile_stride: [SH, SW] stride for vector tiling. Default = tile_size when omitted. Used by trainer/infer.
- add_magnitude: When true, append magnitude channel to vector fields (from first 2–3 channels). Used by vector_field_data.
- normalize: Per-channel z-score normalization for vector tiles. Used by vector_field_data.
- limit_tiles: Optional integer cap on number of vector tiles loaded per source/split. Used by trainer/infer.

Section: model
- num_classes: Number of output classes. Used by cnn_model.SimpleCNN classifier and confusion matrices.
- conv_channels: List of feature map sizes for each conv block. Used by cnn_model.SimpleCNN.
- kernel_size: Convolution kernel size. Used by cnn_model.SimpleCNN/ConvBlock.
- pool_every: Insert MaxPool2d after this many conv blocks. Used by cnn_model.SimpleCNN.
- dropout: Dropout2d probability in conv blocks. Used by cnn_model.SimpleCNN.
- use_batchnorm: Enable BatchNorm2d after conv. Used by cnn_model.SimpleCNN.
- dilations: Dilation per block (int or list[int]) for receptive field control. Used by cnn_model.SimpleCNN.

Section: training
- batch_size: DataLoader batch size. Used by trainer.Trainer.
- num_workers: DataLoader workers. Used by trainer.Trainer.
- pin_memory: Pin GPU memory in DataLoader. Used by trainer.Trainer.
- val_split: Fraction of training set reserved for validation when a dedicated val set is not provided. Used by trainer.Trainer.
- learning_rate: Optimizer LR. Used by cnn_optim.CnnOptim.
- weight_decay: Optimizer weight decay. Used by cnn_optim.CnnOptim.
- optimizer: Optimizer name (adam, adamw, sgd, etc.). Used by cnn_optim.CnnOptim.
- scheduler: LR scheduler name (none, cosine, step, plateau, etc.). Used by cnn_optim.CnnOptim.
- max_epochs: Number of training epochs. Used by trainer.Trainer.
- train_source_labels: Optional per-source labels when providing multiple vector-field inputs. Used by trainer.Trainer to assign labels when embedded labels are absent.
- val_source_labels: Same as above for validation sources.
- training_history_csv: Optional override path for training history CSV; defaults to outputs_root/<subdir>/training/training_history.csv. Used by trainer.Trainer and reports plotting helper.
- training_settings: Not a config key, but trainer writes a snapshot CSV of config keys to outputs_root/<subdir>/training/training_settings.csv at start.
- Timing fields: training history/log CSVs include epoch_time_sec and total_time_sec per epoch.

Section: inference
- use_test: When true, inference chooses test inputs by default; train overrides this to False for building models. Read by main.ModelManager and train.py.
- file_labels: CSV mapping file_name,label for confusion matrix when vector test CSVs lack embedded labels. Used by infer.InferenceRunner.
- file_summary: Output CSV for per-file predictions. Used by infer (both vector and image paths).
- inference_tile_size: [H, W] override for tiling during inference only. Used by infer.run_inference.
- inference_tile_stride: [SH, SW] override for stride during inference only. Used by infer.run_inference.
- recurse: If true, expand directories and globs recursively for inputs. Used by infer.InferenceRunner.
- generate_heatmap: Enable saving per-tile heatmaps and stitched heatmap in vector mode. Used by infer.InferenceRunner.
- generate_confusion_matrix: Enable confusion matrix outputs. Used by infer.InferenceRunner.
- show_heatmap: Display stitched heatmap window after inference (requires matplotlib). Used by infer.run_inference.
- heatmap_class: Class index to visualize in heatmaps; default uses the predicted class. Used by infer.run_inference.
- cm_csv: Path to save confusion matrix CSV. Used by infer.InferenceRunner/save_confusion_outputs.
- cm_png: Path to save confusion matrix image. Used by infer.InferenceRunner/save_confusion_outputs.
- heatmaps_dir: Directory to save per-tile heatmap images. Used by infer.InferenceRunner and HeatmapGenerator.
- stitched_heatmap: Path to save the stitched tile grid image/NPY. Used by infer.InferenceRunner and HeatmapGenerator.
- labels_npy: Optional labels array for viewer or other tools (not required). Recognized by some utilities.
- class_names: Ordered list of class display names; also sets num_classes for image mode if provided. Used by main.ModelManager, infer, and confusion outputs.

Section: viewer
- viewer_values_csv: Values file (CSV) for heatmap_viewer.py when plotting a precomputed grid.
- viewer_title: Plot title for stitched heatmap viewer. Used by heatmap_viewer.py.
- viewer_cmap: Colormap (e.g., viridis). Used by heatmap_viewer.py and HeatmapGenerator.
- viewer_no_show: When true, do not open a window; just save. Used by heatmap_viewer.py.
- viewer_save: Output image path to save viewer result. Used by heatmap_viewer.py.
- viewer_rows: Grid rows for the viewer layout. Used by heatmap_viewer.py.
- viewer_cols: Grid cols for the viewer layout. Used by heatmap_viewer.py.

Image-mode settings (extra details)
- image_mean, image_std: Per-channel normalization (values in [0,1] scale). If one value is given, it is broadcast to all channels.
- augment: Optional augmentation flags for image training, e.g., {"flip": true, "rotate": true}. Used by image_data.ImageFolderDataset.
- Imputation logging: PairedFlowDataset will impute missing digits/velocity/Re using last seen values; when more than 10 imputations occur, details are saved to <metadata_stem>_impute_warnings.csv (and a single warning is printed).

Notes
- Path resolution: common.ConfigManager anchors relative paths to the config file directory so you can keep portable configs.
- Channel inference: In vector mode, input channels are inferred from the vector field (and add_magnitude). In image mode, input channels come from image_channels.
- Class order: For image datasets, class order comes from class_names if provided, else alphabetical subfolder names under train_image_dir/test_image_dir.
- Training progress artifacts: trainer writes training_history.csv and training_progress.png under outputs_root/<subdir>/training; reports.py will reuse these for plots when present.

Entry points
- Training (vector or image depending on *data_kind*): train.py (uses common.ConfigManager, main.ModelManager, trainer.Trainer).
- Inference (vector tiles): infer.py -> run_inference.
- Inference (images): infer.py -> run_image_inference (when test_data_kind == "image").
- Heatmap viewer: heatmap_viewer.py (optional, for visualization).
- Folder setup: setup_folders.py (or common.ensure_outputs_ready) creates expected directories before runs.
