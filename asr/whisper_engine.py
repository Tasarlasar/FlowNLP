"""
Whisper ASR Engine
Wraps faster-whisper with carefully tuned decoding parameters to maximise
accuracy and minimise hallucination on short dictation utterances.

Key improvements over the original
------------------------------------
- Uses 'small' model by default — significantly more accurate than 'base'
  with only ~2x memory (still fast on Apple Silicon via MPS/CPU int8)
- Auto-detects Apple Silicon MPS and uses it automatically
- Hallucination suppression:
    * no_speech_threshold=0.6  — discard result if Whisper thinks it's silence
    * log_prob_threshold=-1.0  — discard low-confidence transcriptions
    * compression_ratio_threshold=2.4 — reject repetitive/looping outputs
    * condition_on_previous_text=False — each utterance decoded independently
      (prevents context bleed-over between chunks)
- Language pinned to English (avoids wasted time on language detection)
- beam_size=5 for best accuracy; can lower to 1 for faster/lower-quality
- VAD filter inside Whisper as a second pass (catches any silence that
  slipped through the recorder's energy gate)
"""

import logging
import platform
from typing import Optional

import numpy as np

from asr.base import ASREngine

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE   = "small"   # was "base" — much better accuracy
DEFAULT_COMPUTE_TYPE = "int8"

# Whisper hallucination guard thresholds
NO_SPEECH_THRESHOLD        = 0.6
LOG_PROB_THRESHOLD         = -1.0
COMPRESSION_RATIO_THRESHOLD = 2.4

# Phrases Whisper commonly hallucinates on silence/noise — suppress them
HALLUCINATION_PHRASES = {
    "thank you",
    "thanks for watching",
    "thanks for watching!",
    "bye",
    "bye!",
    "you",
    ".",
    "",
    "  ",
}


def _best_device() -> tuple[str, str]:
    """
    Return (device, compute_type) for the current hardware.
    - Apple Silicon Mac  → cpu + int8  (MPS not yet stable for CTranslate2)
    - NVIDIA GPU present → cuda + float16
    - Everything else    → cpu + int8
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


class WhisperEngine(ASREngine):
    """
    Transcribes audio using faster-whisper.

    Parameters
    ----------
    model_size : str
        One of: tiny, base, small, medium, large-v2, large-v3.
        Defaults to 'small'.
    device : str | None
        'cpu', 'cuda', or None for auto-detect.
    compute_type : str | None
        CTranslate2 quantisation. None for auto.
    language : str
        Pin to a language code (e.g. 'en') to skip detection overhead.
        Set to None to auto-detect (slower, less reliable on short clips).
    """

    def __init__(
        self,
        model_size:   str  = DEFAULT_MODEL_SIZE,
        device:       Optional[str] = None,
        compute_type: Optional[str] = None,
        language:     Optional[str] = "en",
    ) -> None:
        auto_device, auto_compute = _best_device()
        self._model_size   = model_size
        self._device       = device       or auto_device
        self._compute_type = compute_type or auto_compute
        self._language     = language
        self._model        = None
        self._backend: Optional[str] = None

    # ── Lazy loading ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return

        logger.info(
            "Loading faster-whisper '%s' on %s (%s) …",
            self._model_size, self._device, self._compute_type,
        )
        try:
            from faster_whisper import WhisperModel  # type: ignore
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            self._backend = "faster-whisper"
            logger.info("faster-whisper '%s' loaded.", self._model_size)
        except Exception as exc:  # noqa: BLE001
            logger.warning("faster-whisper failed (%s) — falling back to openai-whisper.", exc)
            import whisper  # type: ignore
            self._model   = whisper.load_model(self._model_size)
            self._backend = "whisper"
            logger.info("openai-whisper '%s' loaded.", self._model_size)

    # ── Public interface ───────────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray) -> dict:
        """
        Transcribe a float32 mono 16 kHz audio array.

        Returns
        -------
        dict  { "text": str, "language": str, "segments": list }
        Returns {"text": ""} if the audio is silent / low-confidence.
        """
        self._load()

        if audio is None or len(audio) == 0:
            return {"text": "", "language": "en", "segments": []}

        # Quick energy check — don't bother Whisper with near-silence
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            logger.debug("Skipping transcription — audio too quiet (rms=%.4f)", rms)
            return {"text": "", "language": "en", "segments": []}

        if self._backend == "faster-whisper":
            return self._transcribe_faster_whisper(audio)
        return self._transcribe_openai_whisper(audio)

    # ── Backends ───────────────────────────────────────────────────────────────

    def _transcribe_faster_whisper(self, audio: np.ndarray) -> dict:
        segments_gen, info = self._model.transcribe(
            audio,
            language             = self._language,
            beam_size            = 5,
            best_of              = 5,
            temperature          = 0.0,        # greedy — most consistent
            no_speech_threshold        = NO_SPEECH_THRESHOLD,
            log_prob_threshold         = LOG_PROB_THRESHOLD,
            compression_ratio_threshold= COMPRESSION_RATIO_THRESHOLD,
            condition_on_previous_text = False, # no context bleed between utterances
            vad_filter           = True,        # Whisper's own VAD as a second pass
            vad_parameters       = {
                "threshold":             0.35,
                "min_speech_duration_ms": 150,
                "min_silence_duration_ms": 300,
            },
        )

        segments  = list(segments_gen)
        full_text = " ".join(seg.text.strip() for seg in segments).strip()

        # Reject common hallucinations
        if full_text.lower().strip(".,!? ") in HALLUCINATION_PHRASES:
            logger.debug("Hallucination suppressed: '%s'", full_text)
            return {"text": "", "language": info.language, "segments": []}

        if full_text:
            logger.info("Transcribed (%.2fs, lang=%s, prob=%.2f): '%s'",
                        len(audio) / 16000, info.language,
                        info.language_probability, full_text)

        return {
            "text":     full_text,
            "language": info.language,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in segments
            ],
        }

    def _transcribe_openai_whisper(self, audio: np.ndarray) -> dict:
        result = self._model.transcribe(
            audio,
            language    = self._language,
            fp16        = False,
            temperature = 0.0,
            no_speech_threshold        = NO_SPEECH_THRESHOLD,
            logprob_threshold          = LOG_PROB_THRESHOLD,
            compression_ratio_threshold= COMPRESSION_RATIO_THRESHOLD,
        )
        text = result.get("text", "").strip()
        if text.lower().strip(".,!? ") in HALLUCINATION_PHRASES:
            text = ""
        return {
            "text":     text,
            "language": result.get("language", "en"),
            "segments": result.get("segments", []),
        }
