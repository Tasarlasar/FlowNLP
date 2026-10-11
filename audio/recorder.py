"""
Streaming Audio Recorder — Silence-Based Segmentation

How it works
------------
Audio is captured continuously into a frame buffer.
After each 30 ms frame, an RMS energy check runs.

- If energy is above ENERGY_THRESHOLD  → frame is "speech", accumulated
- If energy drops below threshold for SILENCE_DURATION_S → utterance ends,
  the accumulated speech buffer is sent to the on_chunk callback and cleared

This mirrors how every production dictation system works: complete
phrases / sentences bounded by natural pauses, not arbitrary time slices.

Benefits over rolling window approach
--------------------------------------
- No duplicate transcription of the same speech
- Whisper receives complete utterances ending at natural pauses
- Vastly reduces hallucination (Whisper guessing on partial words)
- Much lower latency for short phrases
"""

import logging
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from audio.audio_utils import SAMPLE_RATE, CHANNELS, DTYPE

logger = logging.getLogger(__name__)

# ── Tunable constants ──────────────────────────────────────────────────────────

BLOCK_DURATION_MS   = 30        # sounddevice callback interval (ms)
BLOCK_SIZE          = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)

# RMS energy threshold — frames below this are treated as silence.
# 0.01 works well for most microphones in a quiet-ish room.
# Raise to 0.02-0.03 if background noise is still triggering transcription.
ENERGY_THRESHOLD    = 0.01

# How long silence must last before the utterance is considered complete (s)
SILENCE_DURATION_S  = 0.8       # 800 ms — natural sentence-end pause

# Minimum speech duration to bother transcribing (avoids single-word misfires)
MIN_SPEECH_DURATION_S = 0.4     # 400 ms

# Maximum utterance length before forcing a flush (prevents infinite buffering)
MAX_UTTERANCE_S     = 15.0


class StreamingRecorder:
    """
    Calls on_chunk(audio: np.ndarray) at the end of each detected utterance.

    Parameters
    ----------
    on_chunk : Callable[[np.ndarray], None]
        Called from a background thread with each complete speech segment.
    device : int | None
        sounddevice device index (None = system default).
    energy_threshold : float
        RMS level above which a frame is considered speech.
    silence_duration : float
        Seconds of silence that mark the end of an utterance.
    """

    def __init__(
        self,
        on_chunk: Callable[[np.ndarray], None],
        device: Optional[int] = None,
        energy_threshold: float = ENERGY_THRESHOLD,
        silence_duration: float = SILENCE_DURATION_S,
    ) -> None:
        self._on_chunk          = on_chunk
        self._device            = device
        self._energy_threshold  = energy_threshold
        self._silence_duration  = silence_duration

        self._stream: Optional[sd.InputStream] = None

        # Speech accumulation buffer
        self._speech_frames: list[np.ndarray] = []
        self._silence_frames: int = 0           # consecutive silent frames
        self._is_speaking:    bool = False

        self._silence_frame_limit = int(
            silence_duration / (BLOCK_DURATION_MS / 1000)
        )
        self._min_speech_frames = int(
            MIN_SPEECH_DURATION_S / (BLOCK_DURATION_MS / 1000)
        )
        self._max_speech_frames = int(
            MAX_UTTERANCE_S / (BLOCK_DURATION_MS / 1000)
        )

        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._stream is not None:
            return

        with self._lock:
            self._speech_frames  = []
            self._silence_frames = 0
            self._is_speaking    = False

        self._stream = sd.InputStream(
            device=self._device,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info(
            "StreamingRecorder started — device=%s, energy_threshold=%.3f, "
            "silence_duration=%.1fs",
            self._device, self._energy_threshold, self._silence_duration,
        )

    def stop(self) -> Optional[np.ndarray]:
        """
        Stop recording. Flushes any buffered speech as a final chunk.
        Returns the leftover audio array (or None).
        """
        if self._stream is None:
            return None

        self._stream.stop()
        self._stream.close()
        self._stream = None

        with self._lock:
            leftover = self._flush()

        if leftover is not None:
            logger.debug("Flushed %d samples on stop.", len(leftover))
        return leftover

    def is_recording(self) -> bool:
        return self._stream is not None

    # ── sounddevice callback (runs on audio thread) ────────────────────────────

    def _audio_callback(
        self,
        indata:    np.ndarray,
        frames:    int,
        time_info,
        status:    sd.CallbackFlags,
    ) -> None:
        if status:
            logger.debug("sounddevice: %s", status)

        frame = indata[:, 0].copy()          # mono
        rms   = float(np.sqrt(np.mean(frame ** 2)))

        with self._lock:
            if rms >= self._energy_threshold:
                # ── Active speech ─────────────────────────────────────────────
                self._is_speaking    = True
                self._silence_frames = 0
                self._speech_frames.append(frame)

                # Force-flush if utterance is getting too long
                if len(self._speech_frames) >= self._max_speech_frames:
                    logger.debug("Max utterance length reached — force flushing.")
                    audio = self._flush()
                    if audio is not None:
                        threading.Thread(
                            target=self._on_chunk, args=(audio,), daemon=True
                        ).start()

            elif self._is_speaking:
                # ── Trailing silence after speech ─────────────────────────────
                self._speech_frames.append(frame)   # include trailing silence
                self._silence_frames += 1

                if self._silence_frames >= self._silence_frame_limit:
                    # Utterance complete
                    audio = self._flush()
                    if audio is not None:
                        threading.Thread(
                            target=self._on_chunk, args=(audio,), daemon=True
                        ).start()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _flush(self) -> Optional[np.ndarray]:
        """
        Concatenate buffered frames into a single audio array, clear the
        buffer, and return the array — or None if too short to bother.
        Must be called with self._lock held.
        """
        if len(self._speech_frames) < self._min_speech_frames:
            self._speech_frames  = []
            self._silence_frames = 0
            self._is_speaking    = False
            return None

        audio = np.concatenate(self._speech_frames).astype(DTYPE)
        self._speech_frames  = []
        self._silence_frames = 0
        self._is_speaking    = False
        logger.debug("Flushed utterance: %.2f s", len(audio) / SAMPLE_RATE)
        return audio
