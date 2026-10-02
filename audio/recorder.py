"""Capture a complete dictation session and return it when recording stops."""

import logging
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from audio.audio_utils import SAMPLE_RATE, CHANNELS, DTYPE

logger = logging.getLogger(__name__)

BLOCK_SIZE    = int(SAMPLE_RATE * 0.03)   # 30 ms sounddevice callback block


class SessionRecorder:
    """
    Records until stop() and returns the complete mono audio session.

    Parameters
    ----------
    device : int | None
        sounddevice device index (None = system default).
    use_vad : bool
        Whether the captured audio should be filtered through Silero VAD.
    vad_threshold : float
        Silero VAD speech probability threshold (lower = more sensitive).
    """

    def __init__(
        self,
        device: Optional[int] = None,
        use_vad: bool = True,
        vad_threshold: float = 0.30,      # lowered from 0.5 — picks up more speech
    ) -> None:
        self._device     = device
        self._use_vad    = use_vad
        self._vad_threshold = vad_threshold

        self._stream: Optional[sd.InputStream] = None
        self._frames: list[np.ndarray] = []
        self._frames_lock = threading.Lock()

        # Lazy-loaded VAD
        self._vad = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._stream is not None:
            return
        with self._frames_lock:
            self._frames.clear()

        stream = sd.InputStream(
            device=self._device,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=self._audio_callback,
        )
        self._stream = stream
        try:
            stream.start()
        except Exception:
            self._stream = None
            stream.close()
            raise
        logger.info("SessionRecorder started (device=%s, vad=%s, threshold=%.2f)",
                    self._device, self._use_vad, self._vad_threshold)

    def stop(self) -> Optional[np.ndarray]:
        """
        Stop recording and return all captured samples in their original order.
        """
        stream = self._stream
        if stream is None:
            return None

        try:
            stream.stop()
        finally:
            stream.close()
            self._stream = None

        with self._frames_lock:
            frames = self._frames
            self._frames = []
        if not frames:
            return None

        audio = np.concatenate(frames, axis=0).flatten()
        if len(audio) <= SAMPLE_RATE * 0.3:
            return None
        return audio

    def is_recording(self) -> bool:
        return self._stream is not None

    # ------------------------------------------------------------------
    # Internal — audio capture callback (sounddevice thread)
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.debug("sounddevice status: %s", status)
        with self._frames_lock:
            self._frames.append(indata.copy())

    def extract_speech(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Filter a completed session to speech, unless VAD is disabled."""
        if not self._use_vad:
            return audio
        return self._vad_filter(audio)

    def warmup_vad(self) -> None:
        """Load this recorder's VAD model before the first dictation session."""
        if self._use_vad:
            self._vad_filter(np.zeros(SAMPLE_RATE, dtype=np.float32))

    # ------------------------------------------------------------------
    # VAD (lazy load)
    # ------------------------------------------------------------------

    def _vad_filter(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Return speech-only portions of audio, or None if no speech."""
        try:
            if self._vad is None:
                from audio.vad import VADProcessor
                self._vad = VADProcessor(threshold=self._vad_threshold)
            segments = self._vad.get_speech_segments(audio)
            if not segments:
                return None
            return self._vad.filter_speech(audio)
        except Exception as exc:   # noqa: BLE001
            logger.debug("VAD error (passing audio through): %s", exc)
            return audio   # VAD failed — pass through rather than drop
