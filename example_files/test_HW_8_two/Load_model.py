import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("++++ Program Start ++++")

# ===============================================================
# === Load config ===
# ===============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.json")


def _resolve_path(raw_path, default_dir):
    """Mirror the trainer's path resolution logic for cross-platform runs."""

    if not raw_path:
        return default_dir

    expanded = os.path.expanduser(os.path.expandvars(raw_path))
    if os.path.exists(expanded):
        return os.path.abspath(expanded)

    fallback = os.path.join(default_dir, raw_path)
    if os.path.exists(fallback):
        return os.path.abspath(fallback)

    basename_fallback = os.path.join(default_dir, os.path.basename(raw_path))
    return os.path.abspath(basename_fallback)

with open(config_path, "r") as f:
    config = json.load(f)
print("-- config file loaded --")

# Paths
DATA_FILE = _resolve_path(config["data_file"], script_dir)
OUTPUT_DIR = _resolve_path(config.get("output_dir", os.path.join(script_dir, "results")), script_dir)
MODEL_PATH = os.path.abspath(os.path.join(OUTPUT_DIR, "mlp_model.pth"))
ENCODER_PATH = os.path.join(OUTPUT_DIR, "label_encoder.pkl")

# ===============================================================
# === Device setup ===
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-- device setup --")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ===============================================================
# === Load dataset ===
# ===============================================================
print(f"-- Loading dataset from: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
print(f"Dataset loaded successfully with shape: {df.shape}")

# Split features and labels (assuming last column = target)
X = df.iloc[:, :-1].values.astype(np.float32)
y_raw = df.iloc[:, -1].values

# ===============================================================
# === Load Label Encoder ===
# ===============================================================
label_encoder = joblib.load(ENCODER_PATH)
print(f"-- Label encoder loaded from: {ENCODER_PATH}")
print(f"Classes: {list(label_encoder.classes_)}")

# Convert string labels to numeric for evaluation
y = label_encoder.transform(y_raw)

# ===============================================================
# === Define MLP Model (same architecture as training) ===
# ===============================================================
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class FlexibleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, depth, num_classes, dropout=0.2, verbose=False):
        super(FlexibleMLP, self).__init__()

        layers = []
        layer_info = []
        in_features = input_size
        current_size = hidden_size

        for i in range(depth):
            layers.append(nn.Linear(in_features, current_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layer_info.append(f"Hidden Layer {i+1}: {in_features} → {current_size}, Dropout({dropout})")
            in_features = current_size
            current_size = max(current_size // 2, num_classes)

        layers.append(nn.Linear(in_features, num_classes))
        layer_info.append(f"Output Layer: {in_features} → {num_classes}")

        self.network = nn.Sequential(*layers)
        self.layer_info = layer_info

        if verbose:
            self.print_architecture()

    def forward(self, x):
        return self.network(x)

    def print_architecture(self):
        print("\n🧩 FlexibleMLP Architecture:")
        print("────────────────────────────")
        for layer in self.layer_info:
            print(layer)
        print("────────────────────────────\n")


input_size = X.shape[1]
hidden_size = config.get("hidden_size", 64)
depth = config.get("depth", 3)
dropout = config.get("dropout", 0.3)
model_set = config.get("model_set", 0)
num_classes = config.get("num_classes", len(label_encoder.classes_))

if model_set == 0:
    model = MLP(input_size, hidden_size, num_classes).to(device)
    print(f"Standard MLP selected: hidden_size={hidden_size}")
else:
    model = FlexibleMLP(
        input_size,
        hidden_size,
        depth,
        num_classes,
        dropout=dropout,
        verbose=False,
    ).to(device)
    print(
        "Flexible MLP selected: "
        f"hidden_size={hidden_size}, depth={depth}, dropout={dropout}"
    )

# ===============================================================
# === Load trained model ===
# ===============================================================
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"-- Model loaded successfully from: {MODEL_PATH}")

# ===============================================================
# === Run predictions ===
# ===============================================================
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

with torch.no_grad():
    outputs = model(X_tensor)
    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

# Decode predictions back to original labels
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_true_labels = label_encoder.inverse_transform(y)



# ===============================================================
# === Evaluation ===
# ===============================================================
acc = accuracy_score(y_true_labels, y_pred_labels)
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_)

print("\n=== Evaluation Results ===")
print(f"Accuracy: {acc * 100:.2f}%")
print("Confusion Matrix (row specific):\n")
Confusion_Matrix_row_specific  = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
print(Confusion_Matrix_row_specific)


print("Confusion Matrix (column specific):")
Confusion_Matrix_column_specific  = pd.DataFrame(cm.T, index=label_encoder.classes_, columns=label_encoder.classes_)
print(Confusion_Matrix_column_specific)
print('total number of samples in confusion matrix',Confusion_Matrix_column_specific.to_numpy().sum())

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))

# ===============================================================
# === Example: Predict New Data (optional) ===
# ===============================================================
# Example input (replace with your new sample)
# new_sample = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
# new_tensor = torch.tensor(new_sample).to(device)
# with torch.no_grad():
#     pred = torch.argmax(model(new_tensor), dim=1).cpu().numpy()
#     decoded = label_encoder.inverse_transform(pred)
#     print(f"Predicted class for new sample: {decoded[0]}")

print("\n++++ Program End ++++")
