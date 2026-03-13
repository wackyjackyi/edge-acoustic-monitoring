import os
import random
import numpy as np
import pandas as pd
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import librosa
except ImportError as e:
    raise ImportError("Please install librosa: pip install librosa soundfile") from e


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CSV_PATH = r"X:\NEU\capstone\data\metadata\master_audio_dataset_no_negative.csv"
RESULTS_DIR = os.path.join("data", "results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "bcresnet_best.pt")
SPLIT_CACHE_PATH = os.path.join("data", "metadata", "master_audio_dataset_no_negative_split.csv")

SR = 16000
DURATION = 5
MAX_LEN = SR * DURATION
N_MELS = 64
SEED = 42

LABELS = ["Aircraft", "Fire", "Thunder", "Train", "Water", "Wind"]
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 12
NUM_WORKERS = 0


def stratified_split_dataframe(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    train_parts, val_parts, test_parts = [], [], []
    for label, group in df.groupby("label"):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(group)
        n_train = max(1, int(n * 0.8))
        n_val = max(1, int(n * 0.1))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1

        train_parts.append(group.iloc[:n_train].assign(split="train"))
        val_parts.append(group.iloc[n_train:n_train + n_val].assign(split="val"))
        test_parts.append(group.iloc[n_train + n_val:].assign(split="test"))

    out = pd.concat(train_parts + val_parts + test_parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_or_create_split(csv_path: str, split_cache_path: str) -> pd.DataFrame:
    if os.path.exists(split_cache_path):
        df = pd.read_csv(split_cache_path)
        if "split" in df.columns:
            return df

    df = pd.read_csv(csv_path)
    df = df[df["label"].isin(LABELS)].copy()
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    df = stratified_split_dataframe(df, seed=SEED)
    os.makedirs(os.path.dirname(split_cache_path), exist_ok=True)
    df.to_csv(split_cache_path, index=False)
    print(f"Saved split metadata: {split_cache_path}")
    return df


class FixedLengthAudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str, label_to_id: dict):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, path: str) -> np.ndarray:
        y, _ = librosa.load(path, sr=SR, mono=True)
        if len(y) > MAX_LEN:
            y = y[:MAX_LEN]
        elif len(y) < MAX_LEN:
            y = np.pad(y, (0, MAX_LEN - len(y)))
        return y.astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y = self._load_audio(row["path"])
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        x = torch.from_numpy(log_mel).unsqueeze(0)  # [1, n_mels, time]
        target = torch.tensor(self.label_to_id[row["label"]], dtype=torch.long)
        return x, target


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: Tuple[int, int], stride: Tuple[int, int] = (1, 1)):
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
    def __init__(self, ch: int, kernel_size: Tuple[int, int] = (3, 3)):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(ch, ch, kernel_size)
        self.conv2 = DepthwiseSeparableConv(ch, ch, kernel_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


class TinyBCResNet(nn.Module):
    def __init__(self, num_classes: int = 6):
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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = torch.argmax(model(x), dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)
    return running_loss / max(len(loader.dataset), 1)


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    full_df = load_or_create_split(CSV_PATH, SPLIT_CACHE_PATH)
    train_ds = FixedLengthAudioDataset(full_df, split="train", label_to_id=LABEL_TO_ID)
    val_ds = FixedLengthAudioDataset(full_df, split="val", label_to_id=LABEL_TO_ID)
    test_ds = FixedLengthAudioDataset(full_df, split="test", label_to_id=LABEL_TO_ID)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = TinyBCResNet(num_classes=len(LABELS)).to(device)

    class_counts = train_ds.df["label"].value_counts()
    weights = torch.tensor([1.0 / max(class_counts.get(lab, 1), 1) for lab in LABELS],
                           dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = -1.0
    log_rows = []
    for ep in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        improved = False
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            improved = True
        log_rows.append({
            "epoch": ep,
            "train_loss": float(loss),
            "val_acc": float(val_acc),
            "best_val_acc": float(best_val),
            "improved": int(improved),
        })
        print(f"Epoch {ep:02d} | loss={loss:.4f} | val_acc={val_acc:.4f} | best_val={best_val:.4f}")

    pd.DataFrame(log_rows).to_csv(os.path.join(RESULTS_DIR, "bcresnet_training_log.csv"), index=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f"\nBest checkpoint: {CHECKPOINT_PATH}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
