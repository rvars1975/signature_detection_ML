import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR   = "dataset"
TRAIN_CSV  = os.path.join(BASE_DIR, "train_data.csv")
TEST_CSV   = os.path.join(BASE_DIR, "test_data.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "sign_data", "train")
TEST_IMG_DIR  = os.path.join(BASE_DIR, "sign_data", "test")
IMG_SIZE   = 64 
BATCH_SIZE = 64   
EPOCHS     = 5

# ─── DATASET CLASS ────────────────────────────────────────────────────────────
class SignatureDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        df_raw = pd.read_csv(csv_path, header=None)
        self.img_dir   = img_dir
        self.transform = transform
        self.samples   = []

        for _, row in df_raw.iterrows():
            label = int(row[2])
            # use both images from each pair
            for col in [0, 1]:
                rel_path = str(row[col]).replace("/", os.sep)
                full_path = os.path.join(img_dir, rel_path)
                if os.path.exists(full_path):
                    self.samples.append((full_path, label))
            self.samples = self.samples[:8000]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label

# ─── TRANSFORMS ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
train_dataset = SignatureDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform)
test_dataset  = SignatureDataset(TEST_CSV,  TEST_IMG_DIR,  transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

# ─── CNN MODEL ────────────────────────────────────────────────────────────────
class SignatureCNN(nn.Module):
    def __init__(self):
        super(SignatureCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.fc_block(self.conv_block(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SignatureCNN().to(device)
print(f"✅ Using device: {device}")

# ─── TRAINING ─────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses, train_accs = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    acc = correct / total
    train_losses.append(total_loss / len(train_loader))
    train_accs.append(acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_losses[-1]:.4f} | Accuracy: {acc:.4f}")

# ─── EVALUATION ───────────────────────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Real", "Forged"]))

# ─── PLOTS ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, marker='o', color='tomato')
axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")

axes[1].plot(train_accs, marker='o', color='steelblue')
axes[1].set_title("Training Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Real","Forged"], yticklabels=["Real","Forged"])
plt.title("Confusion Matrix"); plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.savefig("confusion_matrix.png")
plt.show()

print("✅ Done! Charts saved.")