# ==========================================================
# auto_train.py
# Automates training across hyperparameter sweeps.
# Saves config, runs training, captures accuracy, and logs results.
# ==========================================================

import json
import subprocess
import itertools
import csv
import os
from datetime import datetime

# === Base paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.json")
TRAIN_SCRIPT = os.path.join(script_dir, "model_setup_and_traning.py")
LOG_FILE = os.path.join(script_dir, "auto_train_results.csv")

# === Load base config ===
with open(config_path, "r") as f:
    base_config = json.load(f)
print("-- Base config loaded --")

# === Define hyperparameter grid ===
param_grid = {
    "learning_rate": [0.001, 0.0005, 0.0001],
    "hidden_size": [32, 64, 128],
    "depth": [2, 3, 4],
    "dropout": [0.2, 0.3]
}

# === Create combinations ===
param_combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

# === Prepare log file ===
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = param_names + ["accuracy", "plot_path", "timestamp"]
        writer.writerow(header)

# === Run all parameter combinations ===
for combo in param_combinations:
    # Update config with current combo
    config = base_config.copy()
    for name, value in zip(param_names, combo):
        config[name] = value

    # Save updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n🚀 Running config:")
    for name, val in zip(param_names, combo):
        print(f"  {name}: {val}")

    # Run training script
    result = subprocess.run(["python", TRAIN_SCRIPT], capture_output=True, text=True)

    # Parse accuracy
    accuracy = None
    for line in result.stdout.splitlines():
        if "Final Accuracy" in line:
            try:
                accuracy = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    # Parse plot path (if training script prints it)
    plot_path = None
    for line in result.stdout.splitlines():
        if "Training plot saved:" in line:
            plot_path = line.split(":", 1)[-1].strip()

    if accuracy is None:
        accuracy = 0.0
    if plot_path is None:
        plot_path = "N/A"

    # Log results
    with open(LOG_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        row = list(combo) + [accuracy, plot_path, datetime.now().isoformat()]
        writer.writerow(row)

    print(f"✔ Completed config. Accuracy = {accuracy:.3f}")
    print(f"📊 Plot: {plot_path}")

print("\n=== Auto-training complete! Results saved to:", LOG_FILE, "===")
