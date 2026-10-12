"""
Whisper ASR Engine
Wraps `faster-whisper` with a fallback to the original `openai-whisper`
if faster-whisper is unavailable.

Model selection can be configured via the settings database.

Supported model sizes: tiny, base, small, medium, large-v2, large-v3
"""

import logging
from typing import Optional

import numpy as np

from asr.base import ASREngine

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "base"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"


class WhisperEngine(ASREngine):
    """
    Transcribes audio using faster-whisper (preferred) or openai-whisper.

    Parameters
    ----------
    model_size : str
        One of: tiny, base, small, medium, large-v2, large-v3.
    device : str
        'cpu' or 'cuda'.
    compute_type : str
        Quantisation level for faster-whisper ('int8', 'float16', 'float32').
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._backend: Optional[str] = None  # 'faster-whisper' or 'whisper'

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return

        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel  # type: ignore

            logger.info(
                "Loading faster-whisper '%s' on %s …",
                self._model_size,
                self._device,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            self._backend = "faster-whisper"
            logger.info("faster-whisper model loaded.")
        except ImportError:
            import whisper  # type: ignore

            logger.info(
                "faster-whisper not available — loading openai-whisper '%s' …",
                self._model_size,
            )
            self._model = whisper.load_model(self._model_size)
            self._backend = "whisper"
            logger.info("openai-whisper model loaded.")

    # ------------------------------------------------------------------
    # ASREngine interface
    # ------------------------------------------------------------------

    def transcribe(
        self, audio: np.ndarray, initial_prompt: Optional[str] = None
    ) -> dict:
        """
        Transcribe a float32 mono 16 kHz audio buffer.

        Returns
        -------
        {
            "text": str,
            "language": str,
            "segments": list,
        }
        """
        self._load()

        if audio is None or len(audio) == 0:
            logger.warning("transcribe() called with empty audio — returning empty result.")
            return {"text": "", "language": "en", "segments": []}

        if self._backend == "faster-whisper":
            return self._transcribe_faster_whisper(audio, initial_prompt)
        return self._transcribe_openai_whisper(audio, initial_prompt)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _transcribe_faster_whisper(
        self, audio: np.ndarray, initial_prompt: Optional[str] = None
    ) -> dict:
        segments_gen, info = self._model.transcribe(
            audio,
            beam_size=5,
            word_timestamps=True,
            initial_prompt=initial_prompt,
        )
        segments = list(segments_gen)
        full_text = " ".join(seg.text.strip() for seg in segments)
        return {
            "text": full_text.strip(),
            "language": info.language,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "words": [
                        {
                            "start": word.start,
                            "end": word.end,
                            "text": word.word,
                            "probability": word.probability,
                        }
                        for word in (s.words or [])
                    ],
                }
                for s in segments
            ],
        }

    def _transcribe_openai_whisper(
        self, audio: np.ndarray, initial_prompt: Optional[str] = None
    ) -> dict:
        result = self._model.transcribe(
            audio, fp16=False, initial_prompt=initial_prompt
        )
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "en"),
            "segments": [
                {
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment.get("text", ""),
                    "words": segment.get("words", []),
                }
                for segment in result.get("segments", [])
            ],
        }
