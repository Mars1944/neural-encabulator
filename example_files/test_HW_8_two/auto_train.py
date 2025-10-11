import os
import json
import subprocess
import itertools
import csv
from datetime import datetime
from tqdm import tqdm
import re
from plot_training import plot_training_results

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

# === Hyperparameter grid (loaded from config.json) ===
auto_train_cfg = base_config.get("auto_train", {})
# Build grid from list-valued entries (exclude helpers like 'hold_constant')
param_grid = {
    k: v for k, v in auto_train_cfg.items()
    if isinstance(v, list) and k not in {"hold_constant"}
}
if not param_grid:
    raise ValueError("No hyperparameter grid found in config.json under 'auto_train'.")
param_names = list(param_grid.keys())

# Prepare log file
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(param_names + ["accuracy", "timestamp"])

# Generate all hyperparameter combinations
param_combinations = list(itertools.product(*param_grid.values()))

# Track best configuration across runs
best_accuracy = None
best_hyperparams = None
best_full_config = None

# === Loop through configurations ===
try:
    for combo in tqdm(param_combinations, desc="Auto-training", unit="config"):
        # Update config
        config = base_config.copy()
        for name, value in zip(param_names, combo):
            config[name] = value

        # Save config temporarily
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        # Run training script with UTF-8 environment to avoid Windows cp1252 issues
        env = dict(os.environ)
        env.update({
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        })
        result = subprocess.run(
            ["python", TRAIN_SCRIPT],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )

        if result.returncode != 0:
            tqdm.write(
                f"Training failed for config {combo} (exit code {result.returncode})."
            )
            if result.stderr:
                tqdm.write(result.stderr.strip())
            accuracy = None
            loss_history = []
            accuracy_history = []
        else:
            # Extract metrics from training output
            accuracy = None
            loss_history = []
            accuracy_history = []
            epoch_pattern = re.compile(
                r"Epoch \[(\d+)/(\d+)\] - Loss: ([0-9.]+), Accuracy: ([0-9.]+)"
            )

            for line in result.stdout.splitlines():
                match = re.search(r"Final Accuracy[:\s]+([0-9.]+)", line)
                if match and accuracy is None:
                    try:
                        accuracy = float(match.group(1))
                    except ValueError:
                        accuracy = None

                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    try:
                        loss_history.append(float(epoch_match.group(3)))
                        accuracy_history.append(float(epoch_match.group(4)))
                    except ValueError:
                        pass

            if accuracy is None:
                tqdm.write(
                    f"Accuracy not reported for config {combo}; marking as unknown."
                )

        # Log results
        with open(LOG_FILE, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                list(combo)
                + ["FAILED" if accuracy is None else f"{accuracy:.6f}", datetime.now().isoformat()]
            )

        if accuracy is not None:
            tqdm.write(f"Config {combo} completed. Accuracy = {accuracy:.4f}")
        else:
            tqdm.write(f"Config {combo} recorded without a valid accuracy.")

        # Generate and save plot
        hyperparams = {name: value for name, value in zip(param_names, combo)}
        if accuracy is not None:
            # Track best configuration so far
            if best_accuracy is None or accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hyperparams = hyperparams
                best_full_config = base_config.copy()
                for k, v in hyperparams.items():
                    best_full_config[k] = v

            if loss_history and accuracy_history:
                plot_filename = "_".join([f"{k}{v}" for k, v in hyperparams.items()]) + ".png"
                plot_path = os.path.join(PLOT_DIR, plot_filename)
                plot_training_results(
                    loss_history=loss_history,
                    accuracy_history=accuracy_history,
                    hyperparams=hyperparams,
                    output_file=plot_path,
                )
            else:
                tqdm.write(
                    "! No epoch-wise metrics captured; skipping auto-train plot for this run."
                )

finally:
    with open(config_path, "w") as f:
        json.dump(base_config, f, indent=4)
    print("\n-- Base config restored after auto-training --")

# === Save optimal configuration ===
op_config_path = os.path.join(script_dir, "op_config.json")
if best_full_config is not None:
    summary = {
        "best_accuracy": best_accuracy,
        "selected_hyperparameters": best_hyperparams,
        "timestamp": datetime.now().isoformat(),
        "source": "auto_train.py",
    }
    best_full_config["auto_train_summary"] = summary
    with open(op_config_path, "w") as f:
        json.dump(best_full_config, f, indent=4)
    print(f"-- Optimal config saved to: {op_config_path} (acc={best_accuracy:.4f})")
else:
    print("-- No valid accuracy found; op_config.json not written.")

print(
    f"\n=== Auto-training complete! Results saved to: {LOG_FILE} and plots in {PLOT_DIR} ==="
)

