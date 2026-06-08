"""Generate and cache the model-ready datasets.

Two input sources are produced (the project compares both):

* **librosa path** — 3s Mel spectrograms computed from the ``.wav`` audio,
  shape ``(N, n_mels, frames, 1)``, normalized to [0, 1], stored as float16.
* **image path** — the provided ``images_original`` PNGs resized to
  ``IMAGE_SIZE`` RGB, shape ``(N, H, W, 3)`` in [0, 1].

Splitting is inherited from the *track* split (see :mod:`src.data`), so 3s
segments never leak across train/val/test. Arrays are cached under
``data/processed/`` and skipped on subsequent runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config as C
from . import audio

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# librosa Mel-spectrogram path
# --------------------------------------------------------------------------- #
def _fix_width(spec: np.ndarray, width: int = C.MEL_FRAMES) -> np.ndarray:
    """Pad or truncate a spectrogram to a fixed number of time frames."""
    if spec.shape[1] == width:
        return spec
    if spec.shape[1] > width:
        return spec[:, :width]
    pad = width - spec.shape[1]
    return np.pad(spec, ((0, 0), (0, pad)), mode="constant", constant_values=spec.min())


def build_mel_dataset(split_index: pd.DataFrame, verbose: bool = True) -> dict:
    """Compute 3s Mel spectrograms for every track, grouped by split.

    Returns ``{split: (X, y)}`` with ``X`` float16 ``(N, n_mels, frames, 1)``
    and ``y`` int ``(N,)``.
    """
    out = {}
    for split in SPLITS:
        rows = split_index[split_index["split"] == split]
        # Pre-allocate the worst case (every track yields SEGMENTS_PER_TRACK).
        cap = len(rows) * C.SEGMENTS_PER_TRACK
        X = np.empty((cap, C.N_MELS, C.MEL_FRAMES, 1), dtype=np.float16)
        y = np.empty(cap, dtype=np.int16)
        n = 0
        for i, (_, r) in enumerate(rows.iterrows()):
            wav = audio.load_audio(r["path"])
            for seg in audio.segment_waveform(wav):
                spec = audio.mel_spectrogram_db(seg)
                spec = _fix_width(spec)
                X[n, :, :, 0] = audio.normalize_minmax(spec)
                y[n] = r["label"]
                n += 1
            if verbose and (i + 1) % 100 == 0:
                print(f"  [{split}] {i + 1}/{len(rows)} tracks -> {n} segments")
        out[split] = (X[:n], y[:n])
        if verbose:
            print(f"  [{split}] done: X={out[split][0].shape}, y={out[split][1].shape}")
    return out


# --------------------------------------------------------------------------- #
# Provided-PNG image path
# --------------------------------------------------------------------------- #
def build_image_dataset(
    split_index: pd.DataFrame, size: tuple[int, int] = C.IMAGE_SIZE, verbose: bool = True
) -> dict:
    """Load and resize the provided spectrogram PNGs, grouped by split."""
    out = {}
    for split in SPLITS:
        rows = split_index[(split_index["split"] == split) & split_index["has_image"]]
        X = np.empty((len(rows), size[1], size[0], 3), dtype=np.float32)
        y = np.empty(len(rows), dtype=np.int16)
        for n, (_, r) in enumerate(rows.iterrows()):
            im = Image.open(r["image_path"]).convert("RGB").resize(size)
            X[n] = np.asarray(im, dtype=np.float32) / 255.0
            y[n] = r["label"]
        out[split] = (X, y)
        if verbose:
            print(f"  [{split}] images: X={X.shape}, y={y.shape}")
    return out


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def save_dataset(dataset: dict, prefix: str, out_dir: Path = C.PROCESSED_DIR) -> None:
    """Persist ``{split: (X, y)}`` as ``<prefix>_<split>_X/y.npy``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, (X, y) in dataset.items():
        np.save(out_dir / f"{prefix}_{split}_X.npy", X)
        np.save(out_dir / f"{prefix}_{split}_y.npy", y)


def load_dataset(prefix: str, out_dir: Path = C.PROCESSED_DIR) -> dict:
    """Load a cached dataset previously written by :func:`save_dataset`."""
    out = {}
    for split in SPLITS:
        X = np.load(out_dir / f"{prefix}_{split}_X.npy")
        y = np.load(out_dir / f"{prefix}_{split}_y.npy")
        out[split] = (X, y)
    return out


def is_cached(prefix: str, out_dir: Path = C.PROCESSED_DIR) -> bool:
    return all(
        (out_dir / f"{prefix}_{split}_{a}.npy").exists()
        for split in SPLITS for a in ("X", "y")
    )
