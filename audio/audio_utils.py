"""
Audio utility helpers shared across the audio package.
"""

import numpy as np


SAMPLE_RATE = 16_000   # Hz — preferred by Whisper
CHANNELS = 1           # Mono
DTYPE = np.float32


def pcm_to_float(data: bytes, dtype=DTYPE) -> np.ndarray:
    """Convert raw PCM bytes to a normalised float32 numpy array."""
    array = np.frombuffer(data, dtype=np.int16).astype(dtype)
    return array / 32768.0


def float_to_pcm(data: np.ndarray) -> bytes:
    """Convert a float32 numpy array back to int16 PCM bytes."""
    clipped = np.clip(data, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def rms_level(data: np.ndarray) -> float:
    """Return the root-mean-square level of an audio buffer."""
    return float(np.sqrt(np.mean(data ** 2)))
