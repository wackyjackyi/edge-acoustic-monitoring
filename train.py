# ============================================================
# train.py
#
# Single training script for all experiments.
# Loads pre-extracted features, trains a model, and saves
# the best checkpoint based on validation accuracy.
#
# Usage examples:
#   python train.py --model bcresnet --features mel --duration 9.6
#   python train.py --model matchboxnet --features mel --duration 30
#   python train.py --model bcresnet --features mel_mfcc --duration 30
#   python train.py --model bcresnet --features mel --duration 5 --chunks
# ============================================================

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    BATCH_SIZE, EPOCHS, LR, RANDOM_SEED,
    TEST_SIZE, VAL_SIZE, FIRE_CAP,
    get_metadata_path, get_model_path
)
from models.bcresnet import TinyBCResNet
from models.matchboxnet import MatchboxNet


# ============================================================
# Argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train audio classification model")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["bcresnet", "matchboxnet"],
        help="Model architecture to train"
    )

    parser.add_argument(
        "--features",
        type=str,
        required=True,
        choices=["mel", "mfcc", "mel_mfcc"],
        help="Feature type to use"
    )

    parser.add_argument(
        "--duration",
        type=float,
        required=True,
        help="Feature duration in seconds (e.g. 9.6, 30, 5)"
    )

    parser.add_argument(
        "--chunks",
        action="store_true",
        help="Use chunked features instead of pad/truncate"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})"
    )

    parser.add_argument(
        "--fire_cap",
        type=int,
        default=FIRE_CAP,
        help=f"Max Fire samples before splitting (default: {FIRE_CAP})"
    )

    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class NPYDataset(Dataset):

    def __init__(self, df, split, label_to_id):
        self.df          = df[df["split"] == split].reset_index(drop=True)
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x   = np.load(row["feature_path"]).astype(np.float32)
        x   = torch.from_numpy(x).unsqueeze(0)   # (1, n_rows, n_frames)
        y   = torch.tensor(self.label_to_id[row["label"]])
        return x, y


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        pred    = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    return correct / total


# ============================================================
# Build model
# ============================================================

def build_model(model_name, num_classes, feature_shape):
    """
    feature_shape: (n_rows, n_frames) — used to set in_channels for MatchboxNet
    """
    if model_name == "bcresnet":
        return TinyBCResNet(num_classes)

    elif model_name == "matchboxnet":
        in_channels = feature_shape[0]  # frequency rows = channels for 1D model
        return MatchboxNet(num_classes, in_channels)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    set_seed()

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------
    meta_path = get_metadata_path(args.features, args.duration, args.chunks)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Metadata not found: {meta_path}\n"
            f"Run feature_extraction.py first with matching arguments."
        )

    df = pd.read_csv(meta_path)
    print(f"Dataset size: {len(df)}")

    df = df[df["feature_path"].apply(
        lambda x: os.path.exists(x) and os.path.getsize(x) > 0
    )].reset_index(drop=True)
    print(f"After removing corrupted features: {len(df)}")

    # --------------------------------------------------------
    # Cap Fire before splitting
    # --------------------------------------------------------
    fire_df     = df[df["label"] == "Fire"].sample(args.fire_cap, random_state=RANDOM_SEED)
    non_fire_df = df[df["label"] != "Fire"]
    df          = pd.concat([fire_df, non_fire_df]).reset_index(drop=True)

    print(f"\nDataset after capping Fire at {args.fire_cap}:")
    print(df["label"].value_counts())

    # --------------------------------------------------------
    # Train / Val / Test split
    # --------------------------------------------------------
    train_df, temp_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=VAL_SIZE, stratify=temp_df["label"], random_state=RANDOM_SEED
    )

    train_df["split"] = "train"
    val_df["split"]   = "val"
    test_df["split"]  = "test"

    df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    labels      = sorted(df["label"].unique())
    label_to_id = {l: i for i, l in enumerate(labels)}

    print(f"\nLabels: {labels}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------
    train_ds = NPYDataset(df, "train", label_to_id)
    val_ds   = NPYDataset(df, "val",   label_to_id)
    test_ds  = NPYDataset(df, "test",  label_to_id)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    sample_shape = np.load(df.iloc[0]["feature_path"]).shape  # (n_rows, n_frames)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice       : {device}")
    print(f"Feature shape: {sample_shape}")
    print(f"Model        : {args.model}")

    model     = build_model(args.model, len(labels), sample_shape).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    model_path = get_model_path(args.model, args.features, args.duration, args.chunks)
    best_val   = 0

    print(f"\nSaving best model to: {model_path}\n")

    for epoch in range(args.epochs):

        model.train()
        running_loss = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | loss={running_loss:.3f} | val_acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), model_path)

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------
    model.load_state_dict(torch.load(model_path))
    test_acc = evaluate(model, test_loader, device)

    print(f"\nBest val accuracy : {best_val:.4f}")
    print(f"Test accuracy     : {test_acc:.4f}")


if __name__ == "__main__":
    main()