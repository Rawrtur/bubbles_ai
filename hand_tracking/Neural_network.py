from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Einstellungen
# --------------------------------------------------

CSV_PATH = Path("hand_tracking/examples/hand_data.csv")
MODEL_PATH = Path("hand_tracking/handgesten.pt")

BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 0.001

torch.manual_seed(42)

# --------------------------------------------------
# 2. CSV laden und überprüfen
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

if "label" not in df.columns:
    raise ValueError("Die CSV-Datei enthält keine Spalte 'label'.")

X = df.drop(columns=["label"]).to_numpy(dtype="float32")
y = df["label"].to_numpy(dtype="int64")

if X.shape[1] != 63:
    raise ValueError(
        f"Erwartet wurden 63 Koordinaten, gefunden: {X.shape[1]}"
    )

print("Anzahl Beispiele:", len(df))
print("Gestenverteilung:")
print(df["label"].value_counts().sort_index())

# --------------------------------------------------
# 3. Trainings- und Testdaten aufteilen
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# --------------------------------------------------
# 4. Neuronales Netz definieren
# --------------------------------------------------

class HandGestureNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


num_classes = len(set(y.tolist()))

model = HandGestureNet(num_classes=num_classes)

# --------------------------------------------------
# 5. Loss-Funktion und Optimierer
# --------------------------------------------------

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# --------------------------------------------------
# 6. Modell trainieren
# --------------------------------------------------

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoche {epoch + 1:02d}/{EPOCHS} "
        f"| Loss: {average_loss:.4f}"
    )

# --------------------------------------------------
# 7. Genauigkeit auf Testdaten messen
# --------------------------------------------------

model.eval()

with torch.no_grad():
    logits = model(X_test)
    predictions = logits.argmax(dim=1)

    correct = (predictions == y_test).sum().item()
    accuracy = correct / len(y_test) * 100

print(f"\nTestgenauigkeit: {accuracy:.2f}%")

# --------------------------------------------------
# 8. Modell speichern
# --------------------------------------------------

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
    },
    MODEL_PATH
)

print(f"Modell gespeichert unter: {MODEL_PATH}")
