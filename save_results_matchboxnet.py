import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class NPYSpectrogramDataset(Dataset):
    def __init__(self, csv_path: str, split: str, label_to_id: dict):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = np.load(row["feature_path"]).astype(np.float32)
        y = self.label_to_id[row["target_label"]]
        x = torch.from_numpy(x)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.dw = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class MatchboxBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, repeats: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(*[
            DepthwiseSeparableConv1d(channels, kernel_size, dropout=dropout)
            for _ in range(repeats)
        ])

    def forward(self, x):
        return self.net(x) + x


class MatchboxNetLite(nn.Module):
    def __init__(self, n_mels: int, num_classes: int, channels: int = 128, kernel_size: int = 11):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.block1 = MatchboxBlock(channels, kernel_size=kernel_size, repeats=3, dropout=0.1)
        self.block2 = MatchboxBlock(channels, kernel_size=kernel_size, repeats=3, dropout=0.1)
        self.block3 = MatchboxBlock(channels, kernel_size=kernel_size, repeats=3, dropout=0.1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_prf(cm):
    n = cm.shape[0]
    out = []
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1 = 2 * p * r / (p + r + 1e-12)
        out.append((p, r, f1))
    return out


@torch.no_grad()
def main():
    labels = ["Fire", "Thunder", "Water", "Negative"]
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    csv_path = os.path.join("data", "metadata", "segments_esc50_features.csv")
    ckpt_path = os.path.join("data", "metadata", "matchboxnet_best.pt")

    results_dir = os.path.join("data", "results")
    os.makedirs(results_dir, exist_ok=True)

    txt_path = os.path.join(results_dir, "matchboxnet_metrics.txt")
    png_path = os.path.join(results_dir, "matchboxnet_confusion_matrix.png")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = NPYSpectrogramDataset(csv_path, split="test", label_to_id=label_to_id)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    x0, _ = ds[0]
    n_mels = int(x0.shape[0])

    model = MatchboxNetLite(n_mels=n_mels, num_classes=len(labels), channels=128, kernel_size=11).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(y.numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, len(labels))
    prf = per_class_prf(cm)
    acc = (np.array(y_true) == np.array(y_pred)).mean()

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Confusion Matrix (rows=true, cols=pred)\n")
        f.write("          " + "  ".join([f"{lab:>8}" for lab in labels]) + "\n")
        for i, lab in enumerate(labels):
            row = "  ".join([f"{cm[i, j]:8d}" for j in range(len(labels))])
            f.write(f"{lab:>8}  {row}\n")
        f.write("\nPer-class Precision / Recall / F1\n")
        for i, (p, r, f1) in enumerate(prf):
            f.write(f"{labels[i]:>8}: P={p:.3f}  R={r:.3f}  F1={f1:.3f}\n")
        f.write(f"\nTest accuracy: {acc:.4f}\n")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("MatchboxNet Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print("Saved:", txt_path)
    print("Saved:", png_path)


if __name__ == "__main__":
    main()