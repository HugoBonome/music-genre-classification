"""Single source of truth for paths and hyperparameters.

Every notebook and module reads parameters from here so that preprocessing,
training and inference stay consistent. Do not hardcode these values elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Repo root = parent of the src/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

# GTZAN lives OUTSIDE the repo (never committed). Override with GTZAN_DIR env var.
GTZAN_DIR = Path(
    os.environ.get("GTZAN_DIR", r"C:\Users\User\Downloads\archive\gtzan-dataset")
)
AUDIO_DIR = GTZAN_DIR / "genres_original"      # <genre>/<genre>.000NN.wav
IMAGE_DIR = GTZAN_DIR / "images_original"      # <genre>/<genre>000NN.png
FEATURES_30S_CSV = GTZAN_DIR / "features_30_sec.csv"
FEATURES_3S_CSV = GTZAN_DIR / "features_3_sec.csv"

# Repo-local outputs.
DATA_DIR = REPO_ROOT / "data"                  # processed arrays / cached spectrograms
MODELS_DIR = REPO_ROOT / "models"              # saved .keras + history
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
NUM_CLASSES = len(GENRES)
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
IDX_TO_GENRE = {i: g for g, i in GENRE_TO_IDX.items()}

# Known-corrupt file in GTZAN (its PNG is also missing). Exclude everywhere.
CORRUPT_FILES = {"jazz.00054.wav"}

# --------------------------------------------------------------------------- #
# Audio / Mel-spectrogram parameters
# --------------------------------------------------------------------------- #
SAMPLE_RATE = 22050          # GTZAN native sample rate
TRACK_DURATION = 30.0        # seconds per original clip
SEGMENT_DURATION = 3.0       # seconds per training segment
SEGMENTS_PER_TRACK = int(TRACK_DURATION // SEGMENT_DURATION)  # 10
SAMPLES_PER_SEGMENT = int(SAMPLE_RATE * SEGMENT_DURATION)

N_MELS = 128                 # Mel bands -> image height
N_FFT = 2048
HOP_LENGTH = 512             # ~130 frames for a 3s segment -> image width
FMIN = 0
FMAX = SAMPLE_RATE // 2

# Number of time frames a 3s segment produces (librosa, center=True).
MEL_FRAMES = 1 + SAMPLES_PER_SEGMENT // HOP_LENGTH  # 130
# CNN input shape for the librosa Mel-spectrogram path (single channel).
INPUT_SHAPE = (N_MELS, MEL_FRAMES, 1)

# Provided-PNG path: images are resized to this RGB size for the image models.
IMAGE_SIZE = (128, 128)

# Where cached/processed arrays are written (kept out of git).
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# --------------------------------------------------------------------------- #
# Train / validation / test split (split BY TRACK to avoid segment leakage)
# --------------------------------------------------------------------------- #
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --------------------------------------------------------------------------- #
# Training defaults
# --------------------------------------------------------------------------- #
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
