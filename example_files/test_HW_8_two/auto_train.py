import os
import json
import subprocess
import itertools
import csv
from datetime import datetime
from tqdm import tqdm
import re
from plot_training import plot_training_results  # drop-in plotting function

# === Paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.json")
TRAIN_SCRIPT = os.path.join(script_dir, "model_setup_and_traning.py")
LOG_FILE = os.path.join(script_dir, "auto_train_results.csv")
PLOT_DIR = os.path.join(script_dir, "auto_train_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# === Load base config ===
with open(config_path, "r") as f:
    base_config = json.load(f)
print("-- Base config loaded --")

# === Hyperparameter grid ===
param_grid = {
    "learning_rate": [0.001, 0.0005, 0.0001],
    "hidden_size": [32, 64, 128],
    "depth": [2, 3, 4],
    "dropout": [0.2, 0.3]
}

param_names = list(param_grid.keys())

# Prepare log file
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(param_names + ["accuracy", "timestamp"])

# Generate all hyperparameter combinations
param_combinations = list(itertools.product(*param_grid.values()))

# === Loop through configurations ===
for combo in tqdm(param_combinations, desc="Auto-training", unit="config"):
    # Update config
    config = base_config.copy()
    for name, value in zip(param_names, combo):
        config[name] = value

    # Save config temporarily
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Run training script
    result = subprocess.run(
        ["python", TRAIN_SCRIPT],
        capture_output=True,
        text=True
    )

    # --- Extract accuracy robustly ---
    accuracy = None
    for line in result.stdout.splitlines():
        match = re.search(r"Final Accuracy[:\s]+([0-9.]+)", line)
        if match:
            try:
                accuracy = float(match.group(1))
            except ValueError:
                accuracy = 0.0

    if accuracy is None:
        print("⚠ Warning: Accuracy not found. Setting to 0.0")
        accuracy = 0.0

    # --- Log results ---
    with open(LOG_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(combo) + [accuracy, datetime.now().isoformat()])

    tqdm.write(f"✔ Config {combo} completed. Accuracy = {accuracy:.4f}")

    # --- Generate and save plot ---
    hyperparams = {name: value for name, value in zip(param_names, combo)}
    # Create descriptive filename
    plot_filename = "_".join([f"{k}{v}" for k, v in hyperparams.items()]) + ".png"
    plot_path = os.path.join(PLOT_DIR, plot_filename)

    # Call the drop-in plot function
    # Here we pass empty lists for loss/accuracy if you don't have them per epoch
    plot_training_results(loss_history=[], accuracy_history=[accuracy], hyperparams=hyperparams, output_file=plot_path)

print(f"\n=== Auto-training complete! Results saved to: {LOG_FILE} and plots in {PLOT_DIR} ===")
