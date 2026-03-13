import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class NPYSpectrogramDataset1D(Dataset):
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
        x = torch.from_numpy(x)  # [n_mels, time]
        y = torch.tensor(y, dtype=torch.long)
        return x, y, row["segment_path"], row["feature_path"], row["target_label"]


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
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
        self.block1 = MatchboxBlock(channels, kernel_size, repeats=3, dropout=0.1)
        self.block2 = MatchboxBlock(channels, kernel_size, repeats=3, dropout=0.1)
        self.block3 = MatchboxBlock(channels, kernel_size, repeats=3, dropout=0.1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def compute_cm(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def save_cm(cm, labels, out_path, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
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

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_sample_panel(rows, out_path, sr=16000):
    fig, axes = plt.subplots(nrows=len(rows), ncols=2, figsize=(12, 3.2 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = np.array([axes])

    for i, r in enumerate(rows):
        seg_path = r["segment_path"]
        feat_path = r["feature_path"]
        title = r["title"]

        y, _ = librosa.load(seg_path, sr=sr, mono=True)
        t = np.linspace(0, len(y) / sr, num=len(y), endpoint=False)

        axes[i, 0].plot(t, y, linewidth=0.8)
        axes[i, 0].set_title(title + " | waveform")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_ylabel("Amplitude")

        logmel = np.load(feat_path).astype(np.float32)
        img = librosa.display.specshow(logmel, sr=sr, x_axis="time", y_axis="mel", ax=axes[i, 1])
        axes[i, 1].set_title(title + " | log-mel")
        fig.colorbar(img, ax=axes[i, 1], format="%+2.0f")

    plt.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def main():
    random.seed(42)

    labels = ["Fire", "Thunder", "Water", "Negative"]
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}

    csv_path = os.path.join("data", "metadata", "segments_esc50_features.csv")
    ckpt_path = os.path.join("data", "metadata", "matchboxnet_best.pt")

    out_dir = os.path.join("data", "results", "viz_matchboxnet")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = NPYSpectrogramDataset1D(csv_path, split="test", label_to_id=label_to_id)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    x0, _, *_ = ds[0]
    n_mels = int(x0.shape[0])

    model = MatchboxNetLite(n_mels=n_mels, num_classes=len(labels), channels=128, kernel_size=11).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    records = []
    y_true, y_pred = [], []

    for x, y, seg_path, feat_path, true_label in dl:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()

        y_pred.extend(pred.tolist())
        y_true.extend(y.numpy().tolist())

        for i in range(len(pred)):
            records.append({
                "segment_path": seg_path[i],
                "feature_path": feat_path[i],
                "true": true_label[i],
                "pred": id_to_label[int(pred[i])]
            })

    cm = compute_cm(y_true, y_pred, len(labels))
    save_cm(cm, labels, os.path.join(out_dir, "confusion_matrix.png"), "MatchboxNet Confusion Matrix")

    dfp = pd.DataFrame(records)
    dfp.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    for cls in labels:
        df_correct = dfp[(dfp["true"] == cls) & (dfp["pred"] == cls)]
        if len(df_correct) > 0:
            picks = df_correct.sample(n=min(3, len(df_correct)), random_state=42)
            rows = []
            for _, r in picks.iterrows():
                rows.append({
                    "segment_path": r["segment_path"],
                    "feature_path": r["feature_path"],
                    "title": f"true={cls}, pred={cls}"
                })
            plot_sample_panel(rows, os.path.join(out_dir, f"samples_correct_{cls}.png"))

    df_mist = dfp[dfp["true"] != dfp["pred"]]
    if len(df_mist) > 0:
        pair = df_mist.groupby(["true", "pred"]).size().sort_values(ascending=False).index[0]
        df_pair = df_mist[(df_mist["true"] == pair[0]) & (df_mist["pred"] == pair[1])]
        picks = df_pair.sample(n=min(3, len(df_pair)), random_state=42)
        rows = []
        for _, r in picks.iterrows():
            rows.append({
                "segment_path": r["segment_path"],
                "feature_path": r["feature_path"],
                "title": f"true={pair[0]}, pred={pair[1]}"
            })
        plot_sample_panel(rows, os.path.join(out_dir, f"samples_mistake_{pair[0]}_as_{pair[1]}.png"))

    print("Saved visualizations to:", out_dir)


if __name__ == "__main__":
    main()