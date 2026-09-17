"""
Central Controller — Streaming Mode
Wires the streaming recorder, ASR, NLP pipeline, context detector,
formatter, and text injector together.

Architecture
------------
- While RECORDING, StreamingRecorder fires on_chunk() every ~1.5 s
  whenever speech is detected.
- Each chunk is transcribed → NLP → formatted → typed immediately.
- The hotkey toggle starts/stops the recording session.
- State always returns to IDLE after a session ends.
"""

import logging
import threading
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

from app.states import AppState
from app.hotkey import HotkeyManager
from audio.audio_utils import SAMPLE_RATE
from audio.recorder import StreamingRecorder
from asr.whisper_engine import WhisperEngine
from nlp.pipeline import NLPPipeline
from context.detector import ContextDetector
from formatter.base import FormatterBase
from formatter.email import EmailFormatter
from formatter.chat import ChatFormatter
from formatter.notes import NotesFormatter
from formatter.search import SearchFormatter
from formatter.code import CodeFormatter
from output.injector import TextInjector
from storage.database import Database

logger = logging.getLogger(__name__)


class Controller(QObject):
    """
    Qt signals
    ----------
    state_changed(AppState)
    transcript_ready(str)   — raw ASR text for the latest chunk
    result_ready(str)       — NLP-processed text for the latest chunk
    error_occurred(str)
    """

    state_changed    = Signal(object)
    transcript_ready = Signal(str)
    result_ready     = Signal(str)
    error_occurred   = Signal(str)

    def __init__(self, db: Database) -> None:
        super().__init__()
        self._db    = db
        self._state = AppState.IDLE

        # Subsystems
        self._recorder  = StreamingRecorder(on_chunk=self._on_audio_chunk)
        self._asr       = WhisperEngine()
        self._nlp       = NLPPipeline()
        self._context   = ContextDetector()
        self._injector  = TextInjector()

        self._formatters: dict[str, FormatterBase] = {
            "EMAIL":  EmailFormatter(),
            "CHAT":   ChatFormatter(),
            "NOTE":   NotesFormatter(),
            "SEARCH": SearchFormatter(),
            "CODE":   CodeFormatter(),
        }

        # Serialise chunk processing — one chunk at a time to avoid
        # overlapping pyautogui calls
        self._chunk_lock = threading.Lock()

        # Cache the active app at session start so we don't query it
        # for every chunk (the user stays in the same window while dictating)
        self._active_app: str = ""
        self._user_mode:  str = ""

        self._hotkey_manager = HotkeyManager(
            on_start=self.start_recording,
            on_stop=self.stop_recording,
        )
        self._hotkey_manager.start()

        # Pre-warm all lazy-loaded models in the background so the first
        # dictation chunk isn't blocked by 30+ seconds of cold loading.
        threading.Thread(target=self._warmup, daemon=True).start()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> AppState:
        return self._state

    def _set_state(self, state: AppState) -> None:
        self._state = state
        self.state_changed.emit(state)
        logger.info("State → %s", state.name)

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        if self._state != AppState.IDLE:
            logger.warning("start_recording ignored — state=%s", self._state.name)
            return
        self._active_app = self._get_active_app()
        self._user_mode  = self._db.get_setting("output_mode", "Auto")
        self._set_state(AppState.RECORDING)
        self._recorder.start()
        logger.info("Recording started. Active app: '%s'", self._active_app)

    def stop_recording(self) -> None:
        if self._state != AppState.RECORDING:
            logger.warning("stop_recording ignored — state=%s", self._state.name)
            return
        self._set_state(AppState.PROCESSING)

        # stop() returns any leftover audio not yet dispatched
        leftover = self._recorder.stop()
        if leftover is not None and len(leftover) > 0:
            logger.debug("Processing leftover audio (%d samples).", len(leftover))
            self._process_chunk(leftover)

        self._set_state(AppState.IDLE)

    # ------------------------------------------------------------------
    # Chunk pipeline (called from StreamingRecorder background thread)
    # ------------------------------------------------------------------

    def _on_audio_chunk(self, audio: np.ndarray) -> None:
        """Callback fired by StreamingRecorder for each detected speech chunk."""
        # Only process chunks while we are still recording
        if self._state not in (AppState.RECORDING, AppState.PROCESSING):
            return
        # Serialise so chunks don't interleave keystrokes
        thread = threading.Thread(
            target=self._process_chunk, args=(audio,), daemon=True
        )
        thread.start()

    def _process_chunk(self, audio: np.ndarray) -> None:
        with self._chunk_lock:
            try:
                # 1. ASR
                asr_result = self._asr.transcribe(audio)
                raw_text: str = asr_result.get("text", "").strip()
                if not raw_text:
                    return
                self.transcript_ready.emit(raw_text)
                logger.info("ASR chunk: '%s'", raw_text)

                # 2. NLP
                nlp_result  = self._nlp.process(raw_text)
                clean_text: str = nlp_result.get("clean_text", raw_text)

                # 3. Context
                context = self._context.detect(
                    nlp_result=nlp_result,
                    active_app=self._active_app,
                    user_mode=self._user_mode if self._user_mode != "Auto" else "",
                )

                # 4. Format
                formatter: FormatterBase = self._formatters.get(context, ChatFormatter())
                formatted: str = formatter.format(clean_text, nlp_result)

                # 5. Type into active window — append a space so words
                #    don't run together across chunks
                to_type = formatted.rstrip() + " "
                self._injector.insert(to_type)
                self.result_ready.emit(formatted)
                logger.info("Typed chunk: '%s'", formatted)

            except Exception as exc:  # noqa: BLE001
                logger.exception("Chunk processing error: %s", exc)
                self.error_occurred.emit(str(exc))

    # ------------------------------------------------------------------
    # Warmup — load all models before the first hotkey press
    # ------------------------------------------------------------------

    def _warmup(self) -> None:
        """Load every lazy model so the first real chunk has no cold-start delay."""
        import numpy as np
        logger.info("Warming up models …")
        try:
            # 1. Whisper
            silence = np.zeros(SAMPLE_RATE * 1, dtype=np.float32)
            self._asr.transcribe(silence)
            logger.info("Warmup: Whisper ready.")
        except Exception as exc:
            logger.warning("Warmup Whisper failed: %s", exc)

        try:
            # 2. NLP pipeline (tokenizer, grammar, NER, intent, embeddings)
            self._nlp.process("hello world")
            logger.info("Warmup: NLP pipeline ready.")
        except Exception as exc:
            logger.warning("Warmup NLP failed: %s", exc)

        try:
            # 3. VAD (inside StreamingRecorder — trigger load on silence)
            from audio.vad import VADProcessor
            vad = VADProcessor()
            vad.contains_speech(np.zeros(SAMPLE_RATE, dtype=np.float32))
            logger.info("Warmup: VAD ready.")
        except Exception as exc:
            logger.warning("Warmup VAD failed: %s", exc)

        logger.info("All models warmed up — ready to dictate.")
        self.state_changed.emit(self._state)   # refresh UI status label

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_active_app(self) -> str:
        try:
            from context.application import get_active_application
            return get_active_application()
        except Exception:  # noqa: BLE001
            return ""

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._state == AppState.RECORDING:
            self._recorder.stop()
        self._hotkey_manager.stop()
        logger.info("Controller shut down.")
