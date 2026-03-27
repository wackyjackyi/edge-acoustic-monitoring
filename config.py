# ============================================================
# config.py
#
# Central configuration file for all experiments.
# All paths, hyperparameters, and feature settings live here.
# ============================================================

import os

# ============================================================
# Paths
# ============================================================

BASE_DIR      = "/content/drive/MyDrive/Master's Project"
CODE_DIR      = os.path.join(BASE_DIR, "code")
DATASETS_DIR  = os.path.join(BASE_DIR, "Datasets")

METADATA_CSV  = os.path.join(DATASETS_DIR, "metadata_all_data.csv")
FEATURES_DIR  = os.path.join(DATASETS_DIR, "features")
MODELS_DIR    = os.path.join(BASE_DIR, "saved_models")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,   exist_ok=True)


# ============================================================
# Audio parameters
# ============================================================

SR         = 16000   # sample rate (Hz)
HOP_LENGTH = 512     # frames step size (librosa default)
N_MELS     = 64      # mel frequency bands
N_MFCC     = 40      # MFCC coefficients


# ============================================================
# Duration options (seconds)
# ============================================================

# number of frames = (duration * SR) / HOP_LENGTH
DURATION_OPTIONS = {
    9.6: 300,    # (300 * 512) / 16000 = 9.6s
    30.0: 938,   # (938 * 512) / 16000 ≈ 30s
}


# ============================================================
# Chunking
# ============================================================

CHUNK_SEC     = 5                          # chunk size in seconds
CHUNK_SAMPLES = SR * CHUNK_SEC             # 80,000 samples
CHUNK_FRAMES  = CHUNK_SAMPLES // HOP_LENGTH  # 156 frames


# ============================================================
# Training hyperparameters
# ============================================================

BATCH_SIZE  = 32
EPOCHS      = 16
LR          = 1e-3
RANDOM_SEED = 42
TEST_SIZE   = 0.2
VAL_SIZE    = 0.5    # fraction of temp_df used for val


# ============================================================
# Balancing
# ============================================================

FIRE_CAP = 300       # max Fire samples before splitting


# ============================================================
# Feature folder naming convention
# ============================================================

def get_feature_folder(features, duration, chunks=False):
    """
    Returns the feature folder name based on settings.

    Examples:
        get_feature_folder("mel", 9.6)           → features/mel_9s
        get_feature_folder("mel", 30)             → features/mel_30s
        get_feature_folder("mel_mfcc", 30)        → features/mel_mfcc_30s
        get_feature_folder("mel", 5, chunks=True) → features/mel_chunks_5s
    """
    if chunks:
        name = f"{features}_chunks_{int(duration)}s"
    else:
        dur_str = f"{int(duration)}s" if duration == int(duration) else f"{duration}s"
        name = f"{features}_{dur_str}"

    return os.path.join(FEATURES_DIR, name)


def get_metadata_path(features, duration, chunks=False):
    """
    Returns the metadata CSV path based on settings.

    Examples:
        get_metadata_path("mel", 9.6)           → metadata_mel_9s.csv
        get_metadata_path("mel_mfcc", 30)        → metadata_mel_mfcc_30s.csv
        get_metadata_path("mel", 5, chunks=True) → metadata_mel_chunks_5s.csv
    """
    if chunks:
        name = f"metadata_with_features_{features}_chunks_{int(duration)}s.csv"
    else:
        dur_str = f"{int(duration)}s" if duration == int(duration) else f"{duration}s"
        name = f"metadata_with_features_{features}_{dur_str}.csv"

    return os.path.join(DATASETS_DIR, name)


def get_model_path(model_name, features, duration, chunks=False):
    """
    Returns the saved model path based on settings.

    Examples:
        get_model_path("bcresnet", "mel", 9.6)    → saved_models/bcresnet_mel_9s.pt
        get_model_path("matchboxnet", "mel", 30)   → saved_models/matchboxnet_mel_30s.pt
    """
    if chunks:
        name = f"{model_name}_{features}_chunks_{int(duration)}s.pt"
    else:
        dur_str = f"{int(duration)}s" if duration == int(duration) else f"{duration}s"
        name = f"{model_name}_{features}_{dur_str}.pt"

    return os.path.join(MODELS_DIR, name)