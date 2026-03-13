import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

# Update this path on your machine
CSV_PATH = r"X:\NEU\capstone\data\metadata\master_audio_dataset_no_negative.csv"

# 6-class setup after removing Negative
CLASSES = ["Aircraft", "Fire", "Thunder", "Train", "Water", "Wind"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Fixed-length audio settings
SR = 16000
DURATION = 5  # seconds
MAX_LEN = SR * DURATION

class FixedLengthAudioDataset(Dataset):
    def __init__(self, csv_path=CSV_PATH, sr=SR, duration=DURATION, n_mels=64):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.sr = sr
        self.duration = duration
        self.max_len = sr * duration
        self.n_mels = n_mels

        # Safety filter in case a Negative row still exists
        self.df = self.df[self.df["label"].isin(CLASSES)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path: str) -> np.ndarray:
        y, _ = librosa.load(path, sr=self.sr, mono=True)

        # Crop if too long
        if len(y) > self.max_len:
            y = y[:self.max_len]

        # Pad with zeros if too short
        elif len(y) < self.max_len:
            y = np.pad(y, (0, self.max_len - len(y)))

        return y.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        label = row["label"]

        y = self._load_audio(path)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=self.n_mels
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        x = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(LABEL_TO_IDX[label], dtype=torch.long)

        return x, target

if __name__ == "__main__":
    ds = FixedLengthAudioDataset()
    print("rows:", len(ds))
    x, y = ds[0]
    print("sample feature shape:", x.shape)
    print("sample label id:", y.item())
