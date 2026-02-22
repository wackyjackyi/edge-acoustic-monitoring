import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# Input / output paths
IN_CSV = os.path.join("data", "metadata", "metadata_master.csv")
OUT_DIR = os.path.join("data", "standardized_wav", "esc50")

# Standardization settings
TARGET_SR = 16000       # target sample rate
TARGET_MONO = True      # convert to mono

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_CSV)

# Only process ESC-50 for Phase 1
df_esc = df[df["source"] == "ESC-50"].copy()

std_paths = []
durations = []

for fp in tqdm(df_esc["file_path"].tolist(), desc="Standardizing"):
    # Load audio, resample, convert to mono
    y, sr = librosa.load(fp, sr=TARGET_SR, mono=TARGET_MONO)

    # Save standardized wav using original filename
    out_fp = os.path.join(OUT_DIR, os.path.basename(fp))
    sf.write(out_fp, y, TARGET_SR, subtype="PCM_16")

    std_paths.append(out_fp)
    durations.append(len(y) / TARGET_SR)

# Update master metadata with standardized path + duration
df.loc[df_esc.index, "std_path"] = std_paths
df.loc[df_esc.index, "duration_sec"] = durations

df.to_csv(IN_CSV, index=False)

print(f"Saved and updated: {IN_CSV}")
print("Example std_path:", df.loc[df_esc.index[0], "std_path"])