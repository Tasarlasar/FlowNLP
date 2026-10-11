"""
Voice Activity Detection — Silero VAD + RMS Energy Gate

Two-stage filtering:
  Stage 1 — RMS energy gate: instantly rejects frames that are too quiet
             to be speech without loading the neural model at all.
  Stage 2 — Silero VAD: neural speech/noise classifier for frames that
             passed the energy check.

This combination catches:
  - Absolute silence (energy gate)
  - Low-level ambient noise / hum (energy gate)
  - Non-speech sounds that are loud enough to pass energy check
    but are classified as non-speech by Silero (fan noise, keyboard, etc.)
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

# ── Tunable thresholds ─────────────────────────────────────────────────────────
# RMS gate: audio below this level is dropped before Silero even runs.
# 0.008 ≈ -42 dBFS — catches silence and very low background hum.
ENERGY_GATE_RMS   = 0.008

# Silero probability threshold: above this = speech.
# 0.30 is intentionally permissive — the energy gate handles the heavy lifting.
VAD_THRESHOLD     = 0.30


class VADProcessor:
    """
    Two-stage VAD: RMS energy gate → Silero neural VAD.

    Parameters
    ----------
    threshold : float
        Silero speech probability threshold (0–1). Lower = more sensitive.
    energy_gate : float
        RMS level below which audio is rejected without running Silero.
    sample_rate : int
        Must be 16 000 Hz for Silero.
    """

    def __init__(
        self,
        threshold:   float = VAD_THRESHOLD,
        energy_gate: float = ENERGY_GATE_RMS,
        sample_rate: int   = SAMPLE_RATE,
    ) -> None:
        self._threshold   = threshold
        self._energy_gate = energy_gate
        self._sample_rate = sample_rate
        self._model       = None
        self._utils       = None

    # ── Lazy model loading ─────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return

        logger.info("Loading Silero VAD model …")

        # Prefer the pip-installed silero-vad package (offline, no torch.hub)
        try:
            from silero_vad import (  # type: ignore
                load_silero_vad,
                get_speech_timestamps,
                collect_chunks,
            )
            self._model = load_silero_vad()
            self._utils = (get_speech_timestamps, collect_chunks)
            logger.info("Silero VAD loaded (pip package).")
            return
        except Exception as exc:
            logger.debug("silero-vad pip unavailable (%s) — trying torch.hub.", exc)

        # Fallback: torch.hub (needs internet on first use)
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        logger.info("Silero VAD loaded (torch.hub).")

    # ── Public API ─────────────────────────────────────────────────────────────

    def contains_speech(self, audio: np.ndarray) -> bool:
        """Return True if the buffer passes both the energy gate and Silero VAD."""
        if not self._passes_energy_gate(audio):
            return False
        return len(self.get_speech_segments(audio)) > 0

    def get_speech_segments(
        self, audio: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Return (start_sec, end_sec) pairs for speech regions.
        Returns [] if the audio fails the energy gate.
        """
        if not self._passes_energy_gate(audio):
            return []

        self._load()
        get_speech_timestamps = self._utils[0]
        tensor = torch.from_numpy(audio).float()

        timestamps = get_speech_timestamps(
            tensor,
            self._model,
            threshold    = self._threshold,
            sampling_rate= self._sample_rate,
            min_speech_duration_ms  = 150,
            min_silence_duration_ms = 100,
        )
        return [
            (t["start"] / self._sample_rate, t["end"] / self._sample_rate)
            for t in timestamps
        ]

    def filter_speech(self, audio: np.ndarray) -> np.ndarray:
        """
        Return only the speech portions concatenated.
        Falls back to the full audio if VAD finds nothing (avoids dropping
        speech that just narrowly missed the threshold).
        """
        if not self._passes_energy_gate(audio):
            return np.array([], dtype=np.float32)

        self._load()
        get_speech_timestamps = self._utils[0]
        collect_chunks        = self._utils[1]

        tensor     = torch.from_numpy(audio).float()
        timestamps = get_speech_timestamps(
            tensor,
            self._model,
            threshold    = self._threshold,
            sampling_rate= self._sample_rate,
            min_speech_duration_ms  = 150,
            min_silence_duration_ms = 100,
        )

        if not timestamps:
            # Nothing detected — return original audio rather than empty
            # (let Whisper's own VAD make the final call)
            return audio

        speech_tensor = collect_chunks(timestamps, tensor)
        return speech_tensor.numpy()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _passes_energy_gate(self, audio: np.ndarray) -> bool:
        """Fast RMS check — runs without loading any model."""
        if audio is None or len(audio) == 0:
            return False
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self._energy_gate:
            logger.debug("Energy gate blocked audio (rms=%.4f < %.4f)", rms, self._energy_gate)
            return False
        return True
