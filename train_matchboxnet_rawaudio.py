import os
import random
import numpy as np
import pandas as pd

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
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "matchboxnet_best.pt")
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
        log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)  # [n_mels, time]
        x = torch.from_numpy(log_mel)
        target = torch.tensor(self.label_to_id[row["label"]], dtype=torch.long)
        return x, target


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.dw = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding,
                            dilation=dilation, groups=channels, bias=False)
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


@torch.no_grad()
def evaluate_acc(model, loader, device):
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
    running = 0.0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * y.size(0)
    return running / max(len(loader.dataset), 1)


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

    x0, _ = train_ds[0]
    n_mels = int(x0.shape[0])
    model = MatchboxNetLite(n_mels=n_mels, num_classes=len(LABELS), channels=128, kernel_size=11).to(device)

    class_counts = train_ds.df["label"].value_counts()
    weights = torch.tensor([1.0 / max(class_counts.get(lab, 1), 1) for lab in LABELS],
                           dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = -1.0
    for ep in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate_acc(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"Epoch {ep:02d} | loss={loss:.4f} | val_acc={val_acc:.4f} | best_val={best_val:.4f}")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    test_acc = evaluate_acc(model, test_loader, device)
    print(f"\nBest checkpoint: {CHECKPOINT_PATH}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
