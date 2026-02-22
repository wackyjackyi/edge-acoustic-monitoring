import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# Inputs / outputs
IN_SEG_CSV = os.path.join("data", "metadata", "segments_esc50.csv")
OUT_FEAT_DIR = os.path.join("data", "features", "esc50")
OUT_SEG_CSV = os.path.join("data", "metadata", "segments_esc50_features.csv")

# Feature settings (log-mel)
SR = 16000
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 64
FMIN = 20
FMAX = 8000  # nyquist for 16k is 8000

os.makedirs(OUT_FEAT_DIR, exist_ok=True)

df = pd.read_csv(IN_SEG_CSV)

# Pass 1: compute log-mel and collect train stats
train_feats = []
feat_cache = []  # store temporary to avoid recompute path mapping issues

for idx, r in tqdm(df.iterrows(), total=len(df), desc="Extracting log-mel (pass1)"):
    seg_path = r["segment_path"]
    y, _ = librosa.load(seg_path, sr=SR, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    feat_cache.append(logmel)

    if r["split"] == "train":
        train_feats.append(logmel)

# Compute global mean/std from TRAIN only
train_stack = np.concatenate([f.reshape(-1) for f in train_feats], axis=0)
mean = float(train_stack.mean())
std = float(train_stack.std() + 1e-8)

print(f"Train mean={mean:.4f}, std={std:.4f}")

# Save normalization stats
np.savez(os.path.join(OUT_FEAT_DIR, "norm_stats.npz"), mean=mean, std=std)

# Pass 2: normalize and save per-segment features
feature_paths = []

for i, logmel in tqdm(list(enumerate(feat_cache)), total=len(feat_cache), desc="Saving normalized features (pass2)"):
    norm = (logmel - mean) / std

    seg_path = df.loc[i, "segment_path"]
    base = os.path.splitext(os.path.basename(seg_path))[0]
    out_path = os.path.join(OUT_FEAT_DIR, base + ".npy")

    np.save(out_path, norm.astype(np.float32))
    feature_paths.append(out_path)

df["feature_path"] = feature_paths
df.to_csv(OUT_SEG_CSV, index=False)

print("Saved:", OUT_SEG_CSV)
print("Feature dir:", OUT_FEAT_DIR)