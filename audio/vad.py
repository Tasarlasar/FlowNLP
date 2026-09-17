"""
Voice Activity Detection — Silero VAD wrapper.

Downloads the Silero VAD model via torch.hub on first use and caches it
locally so subsequent launches are instant.
"""

import logging
from typing import List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
VAD_THRESHOLD = 0.5


class VADProcessor:
    """
    Wraps Silero VAD and exposes a simple API for detecting speech
    segments within a float32 audio buffer.

    Parameters
    ----------
    threshold : float
        Probability threshold above which a frame is considered speech.
    sample_rate : int
        Audio sample rate — must match the model's expected rate.
    """

    def __init__(
        self,
        threshold: float = VAD_THRESHOLD,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._model = None
        self._utils = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading Silero VAD model …")

        # Try the pip-installed silero-vad package first (no internet needed)
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks  # type: ignore
            self._model = load_silero_vad()
            # Wrap utils to match torch.hub tuple interface
            self._utils = (get_speech_timestamps, collect_chunks)
            logger.info("Silero VAD model loaded (pip package).")
            return
        except Exception as exc:
            logger.debug("silero-vad pip package unavailable (%s), trying torch.hub …", exc)

        # Fallback: torch.hub (requires internet on first use)
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        logger.info("Silero VAD model loaded (torch.hub).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def contains_speech(self, audio: np.ndarray) -> bool:
        """Return True if the buffer contains at least one speech segment."""
        segments = self.get_speech_segments(audio)
        return len(segments) > 0

    def get_speech_segments(
        self, audio: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Return a list of (start_sec, end_sec) tuples for speech regions.

        Parameters
        ----------
        audio : np.ndarray
            Float32 mono audio at `self._sample_rate` Hz.
        """
        self._load()
        get_speech_timestamps = self._utils[0]

        tensor = torch.from_numpy(audio).float()
        timestamps = get_speech_timestamps(
            tensor,
            self._model,
            threshold=self._threshold,
            sampling_rate=self._sample_rate,
        )
        return [
            (
                t["start"] / self._sample_rate,
                t["end"] / self._sample_rate,
            )
            for t in timestamps
        ]

    def filter_speech(self, audio: np.ndarray) -> np.ndarray:
        """Return only the speech portions of the audio buffer concatenated."""
        self._load()
        collect_chunks = self._utils[1]
        get_speech_timestamps = self._utils[0]

        tensor = torch.from_numpy(audio).float()
        timestamps = get_speech_timestamps(
            tensor,
            self._model,
            threshold=self._threshold,
            sampling_rate=self._sample_rate,
        )
        if not timestamps:
            return audio
        speech_tensor = collect_chunks(timestamps, tensor)
        return speech_tensor.numpy()
