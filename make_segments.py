import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# Input / output
IN_CSV = os.path.join("data", "metadata", "metadata_master.csv")
OUT_DIR = os.path.join("data", "segments", "esc50")

# Segmentation settings
SR = 16000
SEG_LEN_SEC = 2.0
STRIDE_SEC = 1.0

seg_len = int(SEG_LEN_SEC * SR)
stride = int(STRIDE_SEC * SR)

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_CSV)
df = df[df["source"] == "ESC-50"].copy()

rows = []

for _, r in tqdm(df.iterrows(), total=len(df), desc="Segmenting"):
    std_path = r["std_path"]
    if not isinstance(std_path, str) or not os.path.exists(std_path):
        raise FileNotFoundError(f"Missing std_path: {std_path}")

    y, _ = librosa.load(std_path, sr=SR, mono=True)
    n = len(y)

    # Drop last partial segment for a simple baseline
    num_segs = 1 + (n - seg_len) // stride if n >= seg_len else 0

    base = os.path.splitext(os.path.basename(std_path))[0]

    for i in range(num_segs):
        start = i * stride
        end = start + seg_len
        seg = y[start:end]

        seg_name = f"{base}_seg{i:02d}.wav"
        seg_path = os.path.join(OUT_DIR, seg_name)

        sf.write(seg_path, seg, SR, subtype="PCM_16")

        rows.append({
            "segment_path": seg_path,
            "target_label": r["target_label"],
            "source": r["source"],
            "original_file": std_path,
            "split": r["split"],
            "start_sec": start / SR,
            "end_sec": end / SR,
        })

segments = pd.DataFrame(rows)
out_csv = os.path.join("data", "metadata", "segments_esc50.csv")
segments.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print("\nSegments by label:")
print(segments["target_label"].value_counts())
print("\nSegments by split:")
print(segments["split"].value_counts())