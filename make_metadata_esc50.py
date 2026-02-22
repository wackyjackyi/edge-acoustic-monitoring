import os
import pandas as pd

# ESC-50
ESC_ROOT = os.path.join("data", "raw", "esc50", "ESC-50-master")
META_CSV = os.path.join(ESC_ROOT, "meta", "esc50.csv")
AUDIO_DIR = os.path.join(ESC_ROOT, "audio")

# load labels
df = pd.read_csv(META_CSV)

keep = ["crackling_fire", "thunderstorm", "rain", "helicopter", "train", "airplane", "wind"]
df = df[df["category"].isin(keep)].copy()

map_to_target = {
    "crackling_fire": "Fire",
    "thunderstorm": "Thunder",
    "rain": "Water",
    "helicopter": "Negative",
    "train": "Negative",
    "airplane": "Negative",
    "wind": "Negative",
}
df["target_label"] = df["category"].map(map_to_target)

df["file_path"] = df["filename"].apply(lambda x: os.path.join(AUDIO_DIR, x))
df["source"] = "ESC-50"
df["original_label"] = df["category"]

# train: folds 1-3, val: fold 4, test: fold 5
def fold_to_split(fold: int) -> str:
    if fold in (1, 2, 3):
        return "train"
    if fold == 4:
        return "val"
    return "test"

df["split"] = df["fold"].apply(fold_to_split)

# output metadata_master.csv
out = df[["file_path", "source", "original_label", "target_label", "fold", "split"]].reset_index(drop=True)
os.makedirs(os.path.join("data", "metadata"), exist_ok=True)
out.to_csv(os.path.join("data", "metadata", "metadata_master.csv"), index=False)

print("✅ Saved: data/metadata/metadata_master.csv")
print("\nCounts by target_label:")
print(out["target_label"].value_counts())
print("\nCounts by split:")
print(out["split"].value_counts())