"""Audio loading, segmentation and Mel-spectrogram computation.

All functions read defaults from :mod:`src.config` so the spectrogram
parameters are identical across EDA, preprocessing, training and inference.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from . import config as C


def load_audio(path: str | Path, sr: int = C.SAMPLE_RATE) -> np.ndarray:
    """Load a mono waveform at the target sample rate."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def segment_waveform(
    y: np.ndarray,
    samples_per_segment: int = C.SAMPLES_PER_SEGMENT,
    n_segments: int = C.SEGMENTS_PER_TRACK,
) -> list[np.ndarray]:
    """Split a 30s waveform into ``n_segments`` non-overlapping chunks.

    Segments shorter than ``samples_per_segment`` (e.g. the tail of a clip that
    is slightly under 30s) are dropped so every segment has the same shape.
    """
    segments = []
    for i in range(n_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        chunk = y[start:end]
        if len(chunk) == samples_per_segment:
            segments.append(chunk)
    return segments


def mel_spectrogram_db(
    y: np.ndarray,
    sr: int = C.SAMPLE_RATE,
    n_mels: int = C.N_MELS,
    n_fft: int = C.N_FFT,
    hop_length: int = C.HOP_LENGTH,
    fmin: int = C.FMIN,
    fmax: int = C.FMAX,
) -> np.ndarray:
    """Compute a log-scaled (dB) Mel spectrogram, shape ``(n_mels, frames)``."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmin=fmin, fmax=fmax,
    )
    return librosa.power_to_db(mel, ref=np.max)


def normalize_minmax(spec: np.ndarray) -> np.ndarray:
    """Scale a spectrogram to [0, 1] for use as a CNN image input."""
    lo, hi = spec.min(), spec.max()
    if hi - lo < 1e-8:
        return np.zeros_like(spec, dtype=np.float32)
    return ((spec - lo) / (hi - lo)).astype(np.float32)
