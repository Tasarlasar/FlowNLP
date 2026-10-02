"""Coordinate complete dictation sessions from recording through insertion."""

import logging
import threading
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

from app.states import AppState
from app.hotkey import HotkeyManager
from audio.audio_utils import SAMPLE_RATE
from audio.recorder import SessionRecorder
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
        self._recorder  = SessionRecorder()
        self._asr       = WhisperEngine(
            model_size=self._db.get_setting("whisper_model", "base")
        )
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
        try:
            self._recorder.start()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Could not start microphone recording: %s", exc)
            self.error_occurred.emit(str(exc))
            self._set_state(AppState.IDLE)
            return
        logger.info("Recording started. Active app: '%s'", self._active_app)

    def stop_recording(self) -> None:
        if self._state != AppState.RECORDING:
            logger.warning("stop_recording ignored — state=%s", self._state.name)
            return
        self._set_state(AppState.PROCESSING)

        try:
            audio = self._recorder.stop()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Could not stop microphone recording: %s", exc)
            self.error_occurred.emit(str(exc))
            self._set_state(AppState.IDLE)
            return

        if audio is None or len(audio) == 0:
            self.error_occurred.emit("No speech was captured.")
            self._set_state(AppState.IDLE)
            return

        threading.Thread(
            target=self._process_session, args=(audio,), daemon=True
        ).start()

    def _process_session(self, audio: np.ndarray) -> None:
        try:
            speech = self._recorder.extract_speech(audio)
            if speech is None or len(speech) == 0:
                self.error_occurred.emit("No speech was detected.")
                return

            asr_result = self._asr.transcribe(speech)
            raw_text: str = asr_result.get("text", "").strip()
            if not raw_text:
                self.error_occurred.emit("Speech could not be transcribed.")
                return

            self.transcript_ready.emit(raw_text)
            logger.info("ASR session: '%s'", raw_text)

            personal_dictionary = set(self._db.get_dictionary_words())
            nlp_result = self._nlp.process(
                raw_text,
                personal_dictionary=personal_dictionary,
            )
            clean_text: str = nlp_result.get("clean_text", raw_text)

            context = self._context.detect(
                nlp_result=nlp_result,
                active_app=self._active_app,
                user_mode=self._user_mode if self._user_mode != "Auto" else "",
            )
            formatter: FormatterBase = self._formatters.get(context, ChatFormatter())
            formatted = formatter.format(clean_text, nlp_result).strip()

            if formatted:
                self._injector.insert(formatted)
                self.result_ready.emit(formatted)
                logger.info("Inserted completed dictation: '%s'", formatted)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Session processing error: %s", exc)
            self.error_occurred.emit(str(exc))
        finally:
            self._set_state(AppState.IDLE)

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
            # 3. VAD (warm the same instance used for session processing)
            self._recorder.warmup_vad()
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
