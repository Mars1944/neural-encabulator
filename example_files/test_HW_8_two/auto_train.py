import json
import subprocess
import itertools
import csv
import os
from datetime import datetime

# === Base config file path ===
CONFIG_FILE = "config.json"
TRAIN_SCRIPT = "example_files/test_HW_8_two/model_setup_and_traning.py"
LOG_FILE = "auto_train_results.csv"

# === Load the base config ===
with open(CONFIG_FILE, "r") as f:
    base_config = json.load(f)

print("-- Base config loaded --")

# === Define hyperparameters to vary ===
# You can hold some values constant by using a list with a single value
param_grid = {
    "learning_rate": [0.001, 0.0005, 0.0001],
    "hidden_size": [32, 64, 128],
    "depth": [2, 3, 4],
    "dropout": [0.2, 0.3]
}

# === Create all combinations of parameters ===
param_combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

# === Prepare log file ===
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = param_names + ["accuracy", "timestamp"]
        writer.writerow(header)

# === Run each configuration ===
for combo in param_combinations:
    # Update config with this combination
    config = base_config.copy()
    for name, value in zip(param_names, combo):
        config[name] = value

    # Save temporary config
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n=== Running config: {config} ===")

    # Run the training script and capture output
    result = subprocess.run(
        ["python", TRAIN_SCRIPT],
        capture_output=True,
        text=True
    )

    # Extract accuracy from training output
    # (Assuming your training script prints: "Final Accuracy: X.XXX")
    accuracy = None
    for line in result.stdout.splitlines():
        if "Final Accuracy" in line:
            try:
                accuracy = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    if accuracy is None:
        accuracy = 0.0  # fallback

    # Log results
    with open(LOG_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        row = list(combo) + [accuracy, datetime.now().isoformat()]
        writer.writerow(row)

    print(f"✔ Completed config. Accuracy = {accuracy:.3f}")

print("\n=== Auto-training complete! Results saved to:", LOG_FILE, "===")
