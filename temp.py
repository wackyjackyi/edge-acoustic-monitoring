import pandas as pd
import os

csv_path = r"X:\NEU\capstone\data\metadata\master_audio_dataset.csv"
out_path = r"X:\NEU\capstone\data\metadata\master_audio_dataset_1.csv"

df = pd.read_csv(csv_path)
df = df[df["path"].apply(os.path.exists)].copy()

print("remaining rows:", len(df))
df.to_csv(out_path, index=False)
print("saved:", out_path)