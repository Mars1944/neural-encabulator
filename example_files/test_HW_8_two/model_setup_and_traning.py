import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from datetime import datetime

print("++++ Program Start ++++")

# ===============================================================
# === Load Config ===
# ===============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)
print("-- config file loaded --")

# Paths
DATA_FILE = os.path.abspath(config["data_file"])
OUTPUT_DIR = os.path.abspath(config.get("output_dir", os.path.join(script_dir, "results")))
MODEL_PATH = os.path.abspath(os.path.join(OUTPUT_DIR, "mlp_model.pth"))
PLOT_DIR = os.path.join(script_dir, "plots")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print(f"-- Model output directory: {OUTPUT_DIR}")

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

# Assume last column is target
X = df.iloc[:, :-1].values.astype(np.float32)
y_raw = df.iloc[:, -1].values

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
encoder_path = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
joblib.dump(label_encoder, encoder_path)
print(f"-- Label encoder saved to: {encoder_path}")
print(f"Classes: {list(label_encoder.classes_)}")

# ===============================================================
# === Train/Test Split ===
# ===============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# ===============================================================
# === Model Definitions ===
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
    def __init__(self, input_size, hidden_size, depth, num_classes, dropout=0.2, verbose=True):
        super(FlexibleMLP, self).__init__()

        layers = []
        layer_info = []
        in_features = input_size
        current_size = hidden_size

        # Build hidden layers with decay
        for i in range(depth):
            layers.append(nn.Linear(in_features, current_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layer_info.append(f"Hidden Layer {i+1}: {in_features} → {current_size}, Dropout({dropout})")
            in_features = current_size
            current_size = max(current_size // 2, num_classes)

        # Output layer
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

# ===============================================================
# === Initialize Model ===
# ===============================================================
input_size = X.shape[1]
hidden_size = config.get("hidden_size", 64)
depth = config.get("depth", 3)
dropout = config.get("dropout", 0.3)
learning_rate = config.get("learning_rate", 0.001)
epochs = config.get("epochs", 50)
batch_size = config.get("batch_size", 16)
model_set = config.get("model_set", 1)
num_classes = len(np.unique(y))

if model_set == 0:
    model = MLP(input_size, hidden_size, num_classes).to(device)
    print(f"Standard MLP selected: hidden_size={hidden_size}")
else:
    model = FlexibleMLP(input_size, hidden_size, depth, num_classes, dropout=dropout).to(device)
    print(f"Flexible MLP selected: hidden_size={hidden_size}, depth={depth}, dropout={dropout}")

# ===============================================================
# === Loss, Optimizer, and Training ===
# ===============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_history = []
accuracy_history = []

print(f"-- Training model for {epochs} epochs --")

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0.0

    for i in range(0, X_train_tensor.size(0), batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(X_train_tensor)
    loss_history.append(avg_loss)

    # Validation accuracy each epoch
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        y_pred = torch.argmax(preds, dim=1).cpu().numpy()
        epoch_acc = accuracy_score(y_test, y_pred)
        accuracy_history.append(epoch_acc)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# ===============================================================
# === Evaluation ===
# ===============================================================
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor)
    y_pred = torch.argmax(preds, dim=1).cpu().numpy()
    y_true = y_test_tensor.cpu().numpy()

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n==============================")
print("✅ Evaluation Complete!")
print(f"Overall Accuracy: {acc*100:.2f}%")
print("Confusion Matrix:\n", cm)
print("==============================\n")

# ===============================================================
# === Save Model & Plot Training ===
# ===============================================================
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

# Plot training metrics
from plot_training import plot_training_results

hyperparams = {
    "hidden_size": hidden_size,
    "lr": learning_rate,
    "depth": depth,
    "dropout": dropout
}

plot_path = plot_training_results(loss_history, accuracy_history, hyperparams, output_dir=PLOT_DIR)
print(f"Training plots saved to: {plot_path}")

# ===============================================================
# === Final Output for Auto-Trainer ===
# ===============================================================
print("==============================")
print("✅ Training complete!")
print(f"Final Accuracy: {acc:.4f}")  # <-- Auto-trainer reads this
print("==============================")
print("++++ Program End ++++")
