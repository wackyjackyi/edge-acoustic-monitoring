import os
import random
import numpy as np
import pandas as pd
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# Reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Dataset
class NPYSpectrogramDataset(Dataset):
    def __init__(self, csv_path: str, split: str, label_to_id: dict):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        feat_path = row["feature_path"]
        x = np.load(feat_path).astype(np.float32)  # shape: [n_mels, time]
        y = self.label_to_id[row["target_label"]]

        # Add channel dim: [1, n_mels, time]
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


# BC-ResNet-like lightweight model (baseline)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: Tuple[int, int], stride: Tuple[int, int] = (1, 1)):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        # Depthwise conv
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)

        # Pointwise conv
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: Tuple[int, int] = (3, 3)):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(ch, ch, kernel_size)
        self.conv2 = DepthwiseSeparableConv(ch, ch, kernel_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


class TinyBCResNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Downsample along time more than frequency (common for audio)
        self.down1 = DepthwiseSeparableConv(16, 32, kernel_size=(3, 3), stride=(1, 2))
        self.res1 = ResidualBlock(32)

        self.down2 = DepthwiseSeparableConv(32, 64, kernel_size=(3, 3), stride=(2, 2))
        self.res2 = ResidualBlock(64)

        self.down3 = DepthwiseSeparableConv(64, 96, kernel_size=(3, 3), stride=(2, 2))
        self.res3 = ResidualBlock(96)

        # Global pooling + classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.res1(x)
        x = self.down2(x)
        x = self.res2(x)
        x = self.down3(x)
        x = self.res3(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x

# Train / Eval
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)

    return running_loss / max(len(loader.dataset), 1)


def main():
    import time  # keep import here to avoid changing global imports if you prefer

    set_seed(42)

    csv_path = os.path.join("data", "metadata", "segments_esc50_features.csv")

    # Label mapping (keep consistent everywhere)
    labels = ["Fire", "Thunder", "Water", "Negative"]
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    # Hyperparameters
    batch_size = 32
    lr = 1e-3
    epochs = 12

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = NPYSpectrogramDataset(csv_path, split="train", label_to_id=label_to_id)
    val_ds = NPYSpectrogramDataset(csv_path, split="val", label_to_id=label_to_id)
    test_ds = NPYSpectrogramDataset(csv_path, split="test", label_to_id=label_to_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TinyBCResNet(num_classes=len(labels)).to(device)

    # Class weighting (optional): helps because Negative has more segments
    class_counts = train_ds.df["target_label"].value_counts()
    weights = []
    for lab in labels:
        weights.append(1.0 / max(class_counts.get(lab, 1), 1))
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Output paths
    results_dir = os.path.join("data", "results")
    os.makedirs(results_dir, exist_ok=True)

    best_path = os.path.join("data", "metadata", "bcresnet_best.pt")
    log_csv_path = os.path.join(results_dir, "training_log.csv")
    config_path = os.path.join(results_dir, "run_config.txt")

    # Save run config
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"device={device}\n")
        f.write(f"torch_version={torch.__version__}\n")
        f.write(f"csv_path={csv_path}\n")
        f.write(f"labels={labels}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"lr={lr}\n")
        f.write(f"epochs={epochs}\n")
        f.write(f"class_weights={weights.detach().cpu().numpy().tolist()}\n")

    best_val = -1.0
    logs = []

    for ep in range(1, epochs + 1):
        t0 = time.time()

        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)

        improved = False
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)
            improved = True

        elapsed = time.time() - t0

        logs.append({
            "epoch": ep,
            "train_loss": float(loss),
            "val_acc": float(val_acc),
            "best_val_acc": float(best_val),
            "improved": int(improved),
            "seconds": float(elapsed),
        })

        print(f"Epoch {ep:02d} | loss={loss:.4f} | val_acc={val_acc:.4f} | best_val={best_val:.4f}")

    # Save training log CSV
    pd.DataFrame(logs).to_csv(log_csv_path, index=False)
    print(f"\nSaved training log: {log_csv_path}")

    # Load best and evaluate on test
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc = evaluate(model, test_loader, device)

    # Append final test result to config file
    with open(config_path, "a", encoding="utf-8") as f:
        f.write(f"best_checkpoint={best_path}\n")
        f.write(f"test_acc={test_acc:.6f}\n")

    print(f"\nBest checkpoint: {best_path}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Saved run config: {config_path}")


if __name__ == "__main__":
    main()