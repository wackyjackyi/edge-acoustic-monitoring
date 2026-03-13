import os
import json
import shutil
import random
from datetime import datetime
from typing import Tuple

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

try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError as e:
    raise ImportError("Please install scikit-learn: pip install scikit-learn") from e

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("Please install matplotlib: pip install matplotlib") from e


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ====== USER SETTINGS ======
CSV_PATH = r"X:\NEU\capstone\data\metadata\master_audio_dataset_no_negative.csv"
RESULTS_ROOT = os.path.join("data", "results", "bcresnet_runs")
SPLIT_CACHE_PATH = os.path.join("data", "metadata", "master_audio_dataset_no_negative_split.csv")
RUN_NAME = datetime.now().strftime("run_%Y%m%d_%H%M%S")

SR = 16000
DURATION = 5
MAX_LEN = SR * DURATION
N_MELS = 64
SEED = 42

LABELS = ["Aircraft", "Fire", "Thunder", "Train", "Water", "Wind"]
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}
ID_TO_LABEL = {i: lab for lab, i in LABEL_TO_ID.items()}

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 12
NUM_WORKERS = 0
# ===========================


class RunPaths:
    def __init__(self, results_root: str, run_name: str):
        self.run_dir = os.path.join(results_root, run_name)
        self.checkpoint = os.path.join(self.run_dir, "bcresnet_best.pt")
        self.training_log = os.path.join(self.run_dir, "training_log.csv")
        self.summary_txt = os.path.join(self.run_dir, "summary.txt")
        self.config_json = os.path.join(self.run_dir, "config.json")
        self.class_report_csv = os.path.join(self.run_dir, "classification_report.csv")
        self.class_report_txt = os.path.join(self.run_dir, "classification_report.txt")
        self.confusion_csv = os.path.join(self.run_dir, "confusion_matrix.csv")
        self.confusion_png = os.path.join(self.run_dir, "confusion_matrix.png")
        self.test_preds_csv = os.path.join(self.run_dir, "test_predictions.csv")
        self.split_copy = os.path.join(self.run_dir, "split_metadata_used.csv")

    def make_dirs(self):
        os.makedirs(self.run_dir, exist_ok=True)


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
        x = torch.from_numpy(log_mel).unsqueeze(0)
        target = torch.tensor(self.label_to_id[row["label"]], dtype=torch.long)
        return x, target, row["path"], row["label"]


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
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        pred = torch.argmax(model(x), dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def predict_with_details(model, loader, device):
    model.eval()
    all_true_ids = []
    all_pred_ids = []
    all_paths = []
    all_true_labels = []
    all_pred_labels = []

    for x, y, paths, true_labels in loader:
        x = x.to(device)
        logits = model(x)
        pred_ids = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        true_ids = y.numpy().tolist()

        all_true_ids.extend(true_ids)
        all_pred_ids.extend(pred_ids)
        all_paths.extend(list(paths))
        all_true_labels.extend(list(true_labels))
        all_pred_labels.extend([ID_TO_LABEL[i] for i in pred_ids])

    pred_df = pd.DataFrame({
        "path": all_paths,
        "true_label": all_true_labels,
        "pred_label": all_pred_labels,
        "true_id": all_true_ids,
        "pred_id": all_pred_ids,
    })
    return all_true_ids, all_pred_ids, pred_df


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)
    return running_loss / max(len(loader.dataset), 1)


def save_confusion_matrix(cm: np.ndarray, labels: list, out_png: str):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = RunPaths(RESULTS_ROOT, RUN_NAME)
    paths.make_dirs()

    config = {
        "csv_path": CSV_PATH,
        "results_root": RESULTS_ROOT,
        "run_name": RUN_NAME,
        "split_cache_path": SPLIT_CACHE_PATH,
        "sample_rate": SR,
        "duration_sec": DURATION,
        "max_len": MAX_LEN,
        "n_mels": N_MELS,
        "seed": SEED,
        "labels": LABELS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "num_workers": NUM_WORKERS,
        "device": device,
    }
    with open(paths.config_json, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    full_df = load_or_create_split(CSV_PATH, SPLIT_CACHE_PATH)
    shutil.copy2(SPLIT_CACHE_PATH, paths.split_copy)

    train_ds = FixedLengthAudioDataset(full_df, split="train", label_to_id=LABEL_TO_ID)
    val_ds = FixedLengthAudioDataset(full_df, split="val", label_to_id=LABEL_TO_ID)
    test_ds = FixedLengthAudioDataset(full_df, split="test", label_to_id=LABEL_TO_ID)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = TinyBCResNet(num_classes=len(LABELS)).to(device)

    class_counts = train_ds.df["label"].value_counts()
    weights = torch.tensor(
        [1.0 / max(class_counts.get(lab, 1), 1) for lab in LABELS],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = -1.0
    best_epoch = -1
    log_rows = []

    print(f"Results folder: {paths.run_dir}")
    for ep in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        improved = False
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = ep
            torch.save(model.state_dict(), paths.checkpoint)
            improved = True

        log_rows.append({
            "epoch": ep,
            "train_loss": float(loss),
            "val_acc": float(val_acc),
            "best_val_acc": float(best_val),
            "best_epoch_so_far": int(best_epoch),
            "improved": int(improved),
        })
        print(f"Epoch {ep:02d} | loss={loss:.4f} | val_acc={val_acc:.4f} | best_val={best_val:.4f}")

    pd.DataFrame(log_rows).to_csv(paths.training_log, index=False)

    model.load_state_dict(torch.load(paths.checkpoint, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    true_ids, pred_ids, pred_df = predict_with_details(model, test_loader, device)
    pred_df.to_csv(paths.test_preds_csv, index=False)

    report_dict = classification_report(
        true_ids,
        pred_ids,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(paths.class_report_csv)

    report_text = classification_report(
        true_ids,
        pred_ids,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
    )
    with open(paths.class_report_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    cm = confusion_matrix(true_ids, pred_ids, labels=list(range(len(LABELS))))
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    cm_df.to_csv(paths.confusion_csv)
    save_confusion_matrix(cm, LABELS, paths.confusion_png)

    split_counts = full_df.groupby(["split", "label"]).size().unstack(fill_value=0)
    with open(paths.summary_txt, "w", encoding="utf-8") as f:
        f.write("BC-ResNet training summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Run folder: {paths.run_dir}\n")
        f.write(f"CSV used: {CSV_PATH}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best val acc: {best_val:.4f}\n")
        f.write(f"Test acc: {test_acc:.4f}\n\n")
        f.write("Labels:\n")
        for i, lab in enumerate(LABELS):
            f.write(f"  {i}: {lab}\n")
        f.write("\nTrain split class counts:\n")
        f.write(split_counts.loc["train"].to_string() + "\n")
        f.write("\nVal split class counts:\n")
        f.write(split_counts.loc["val"].to_string() + "\n")
        f.write("\nTest split class counts:\n")
        f.write(split_counts.loc["test"].to_string() + "\n")
        f.write("\nClassification report:\n")
        f.write(report_text)

    print(f"\nBest checkpoint: {paths.checkpoint}")
    print(f"Training log: {paths.training_log}")
    print(f"Summary: {paths.summary_txt}")
    print(f"Classification report: {paths.class_report_txt}")
    print(f"Confusion matrix: {paths.confusion_png}")
    print(f"Test predictions: {paths.test_preds_csv}")
    print(f"Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
