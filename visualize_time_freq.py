import os
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

CSV_PATH = os.path.join("data", "metadata", "segments_esc50_features.csv")
OUT_DIR = os.path.join("data", "results", "viz")

SR = 16000
N_SAMPLES_PER_CLASS = 3

random.seed(42)
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

classes = ["Fire", "Thunder", "Water", "Negative"]

for cls in classes:
    df_cls = df[df["target_label"] == cls].copy()

    # Prefer test samples for visualization (optional)
    if (df_cls["split"] == "test").any():
        df_pool = df_cls[df_cls["split"] == "test"]
    else:
        df_pool = df_cls

    picks = df_pool.sample(n=min(N_SAMPLES_PER_CLASS, len(df_pool)), random_state=42)

    fig, axes = plt.subplots(
        nrows=len(picks),
        ncols=2,
        figsize=(12, 3.2 * len(picks)),
        constrained_layout=True
    )

    if len(picks) == 1:
        axes = np.array([axes])

    for i, (_, row) in enumerate(picks.iterrows()):
        seg_path = row["segment_path"]
        feat_path = row["feature_path"]

        # Time domain: waveform
        y, _ = librosa.load(seg_path, sr=SR, mono=True)
        t = np.linspace(0, len(y) / SR, num=len(y), endpoint=False)

        ax_wave = axes[i, 0]
        ax_wave.plot(t, y, linewidth=0.8)
        ax_wave.set_title(f"{cls} | waveform | {os.path.basename(seg_path)}")
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude")

        # Frequency/time domain: log-mel spectrogram
        logmel = np.load(feat_path).astype(np.float32)  # [n_mels, time]
        ax_mel = axes[i, 1]
        img = librosa.display.specshow(
            logmel,
            sr=SR,
            x_axis="time",
            y_axis="mel",
            ax=ax_mel
        )
        ax_mel.set_title(f"{cls} | log-mel spectrogram")
        fig.colorbar(img, ax=ax_mel, format="%+2.0f")

    out_path = os.path.join(OUT_DIR, f"{cls}_time_freq.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

print(f"Saved visualizations to: {OUT_DIR}")