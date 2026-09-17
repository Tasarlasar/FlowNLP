"""
Streaming Audio Recorder
Captures microphone input in rolling chunks and fires an on_chunk
callback every time a speech segment is ready for transcription.

Design
------
- Audio is collected into a rolling ring-buffer of CHUNK_SECONDS seconds.
- Every STEP_SECONDS the buffer is checked for speech via VAD.
- If speech is present the chunk is forwarded to the ASR callback.
- This gives live, overlapping transcription without waiting for the
  hotkey-stop event.
- When stop() is called any remaining audio is flushed.
"""

import logging
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from audio.audio_utils import SAMPLE_RATE, CHANNELS, DTYPE

logger = logging.getLogger(__name__)

CHUNK_SECONDS = 2.0       # seconds of audio per transcription chunk
STEP_SECONDS  = 1.5       # how often to fire the callback (overlap = 0.5 s)
BLOCK_SIZE    = int(SAMPLE_RATE * 0.03)   # 30 ms sounddevice callback block


class StreamingRecorder:
    """
    Fires on_chunk(audio: np.ndarray) continuously while recording.

    Parameters
    ----------
    on_chunk : Callable[[np.ndarray], None]
        Called from a background thread with each speech chunk.
    device : int | None
        sounddevice device index (None = system default).
    use_vad : bool
        Filter silent frames before forwarding.
    vad_threshold : float
        Silero VAD speech probability threshold (lower = more sensitive).
    """

    def __init__(
        self,
        on_chunk: Callable[[np.ndarray], None],
        device: Optional[int] = None,
        use_vad: bool = True,
        vad_threshold: float = 0.30,      # lowered from 0.5 — picks up more speech
    ) -> None:
        self._on_chunk   = on_chunk
        self._device     = device
        self._use_vad    = use_vad
        self._vad_threshold = vad_threshold

        self._stream: Optional[sd.InputStream] = None
        self._ring: list[np.ndarray] = []          # rolling buffer
        self._ring_lock  = threading.Lock()
        self._stop_event = threading.Event()
        self._timer_thread: Optional[threading.Thread] = None

        # Lazy-loaded VAD
        self._vad = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._stream is not None:
            return
        with self._ring_lock:
            self._ring = []
        self._stop_event.clear()

        self._stream = sd.InputStream(
            device=self._device,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=self._audio_callback,
        )
        self._stream.start()

        # Timer thread fires the chunk callback every STEP_SECONDS
        self._timer_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True
        )
        self._timer_thread.start()
        logger.info("StreamingRecorder started (device=%s, vad=%s, threshold=%.2f)",
                    self._device, self._use_vad, self._vad_threshold)

    def stop(self) -> Optional[np.ndarray]:
        """
        Stop recording. Returns the final leftover audio (may be None).
        The caller should transcribe this last chunk too.
        """
        if self._stream is None:
            return None

        self._stop_event.set()

        self._stream.stop()
        self._stream.close()
        self._stream = None

        if self._timer_thread:
            self._timer_thread.join(timeout=2)
            self._timer_thread = None

        # Flush remaining buffer
        with self._ring_lock:
            leftover = self._collect()
        if leftover is not None and len(leftover) > SAMPLE_RATE * 0.3:
            return leftover
        return None

    def is_recording(self) -> bool:
        return self._stream is not None

    # ------------------------------------------------------------------
    # Internal — audio capture callback (sounddevice thread)
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.debug("sounddevice status: %s", status)
        with self._ring_lock:
            self._ring.append(indata.copy())

    # ------------------------------------------------------------------
    # Internal — dispatch loop (timer thread)
    # ------------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        import time
        while not self._stop_event.wait(timeout=STEP_SECONDS):
            with self._ring_lock:
                audio = self._collect()
            if audio is not None:
                self._maybe_fire(audio)

    def _collect(self) -> Optional[np.ndarray]:
        """Return the last CHUNK_SECONDS of audio from the ring buffer."""
        if not self._ring:
            return None
        audio = np.concatenate(self._ring, axis=0).flatten()
        # Keep only the last CHUNK_SECONDS worth
        max_samples = int(SAMPLE_RATE * CHUNK_SECONDS)
        if len(audio) > max_samples:
            audio = audio[-max_samples:]
        return audio

    def _maybe_fire(self, audio: np.ndarray) -> None:
        """Run VAD and fire the callback if speech is detected."""
        if self._use_vad:
            filtered = self._vad_filter(audio)
            if filtered is None or len(filtered) < SAMPLE_RATE * 0.2:
                return   # less than 200 ms of speech — skip
            audio = filtered
        self._on_chunk(audio)

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
