import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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
        x = np.load(row["feature_path"]).astype(np.float32)  # [n_mels, time]
        y = self.label_to_id[row["target_label"]]
        x = torch.from_numpy(x).unsqueeze(0)  # [1, n_mels, time]
        y = torch.tensor(y, dtype=torch.long)
        return x, y


# Model (same as training)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=(1, 1)):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel_size=(3, 3)):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(ch, ch, kernel_size)
        self.conv2 = DepthwiseSeparableConv(ch, ch, kernel_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


class TinyBCResNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.down1 = DepthwiseSeparableConv(16, 32, kernel_size=(3, 3), stride=(1, 2))
        self.res1 = ResidualBlock(32)
        self.down2 = DepthwiseSeparableConv(32, 64, kernel_size=(3, 3), stride=(2, 2))
        self.res2 = ResidualBlock(64)
        self.down3 = DepthwiseSeparableConv(64, 96, kernel_size=(3, 3), stride=(2, 2))
        self.res3 = ResidualBlock(96)
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
        return self.fc(x)


# Metrics
def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_prf(cm):
    num_classes = cm.shape[0]
    metrics = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        metrics.append((precision, recall, f1))
    return metrics


@torch.no_grad()
def main():
    labels = ["Fire", "Thunder", "Water", "Negative"]
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}

    csv_path = os.path.join("data", "metadata", "segments_esc50_features.csv")
    ckpt_path = os.path.join("data", "metadata", "bcresnet_best.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = NPYSpectrogramDataset(csv_path, split="test", label_to_id=label_to_id)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model = TinyBCResNet(num_classes=len(labels)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    y_true = []
    y_pred = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(y.numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, num_classes=len(labels))
    metrics = per_class_prf(cm)

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("       " + "  ".join([f"{lab:>8}" for lab in labels]))
    for i in range(len(labels)):
        row = "  ".join([f"{cm[i, j]:8d}" for j in range(len(labels))])
        print(f"{labels[i]:>8}  {row}")

    print("\nPer-class Precision / Recall / F1:")
    for i, (p, r, f1) in enumerate(metrics):
        print(f"{labels[i]:>8}: P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

    acc = (np.array(y_true) == np.array(y_pred)).mean()
    print(f"\nTest accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()