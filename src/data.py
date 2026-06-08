"""Dataset indexing and leakage-free train/val/test splitting.

The index is at the *track* level (one row per 30s clip). Splitting happens on
tracks, then segments inherit their track's split — this prevents 3s segments
from the same song leaking across train/val/test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def build_track_index() -> pd.DataFrame:
    """Scan ``genres_original`` and return one row per audio track.

    Columns: ``path``, ``filename``, ``genre``, ``label`` (int), ``has_image``.
    The known-corrupt file (``jazz.00054.wav``) is excluded.
    """
    rows = []
    for genre in C.GENRES:
        for wav in sorted((C.AUDIO_DIR / genre).glob("*.wav")):
            if wav.name in C.CORRUPT_FILES:
                continue
            # PNG naming drops the dot: jazz.00054.wav -> jazz00054.png
            png_name = wav.name.replace(".wav", ".png").replace(".", "", 1)
            png = C.IMAGE_DIR / genre / png_name
            rows.append(
                {
                    "path": str(wav),
                    "filename": wav.name,
                    "genre": genre,
                    "label": C.GENRE_TO_IDX[genre],
                    "image_path": str(png) if png.exists() else None,
                    "has_image": png.exists(),
                }
            )
    return pd.DataFrame(rows)


def split_tracks(
    index: pd.DataFrame,
    seed: int = C.SEED,
    train_ratio: float = C.TRAIN_RATIO,
    val_ratio: float = C.VAL_RATIO,
) -> pd.DataFrame:
    """Add a ``split`` column ('train'/'val'/'test'), stratified per genre.

    Splitting is done within each genre so class balance is preserved across
    the three splits.
    """
    rng = np.random.default_rng(seed)
    index = index.copy()
    index["split"] = ""
    for genre in C.GENRES:
        idx = index.index[index["genre"] == genre].to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        index.loc[idx[:n_train], "split"] = "train"
        index.loc[idx[n_train:n_train + n_val], "split"] = "val"
        index.loc[idx[n_train + n_val:], "split"] = "test"
    return index
