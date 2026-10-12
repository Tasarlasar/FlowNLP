"""
AppController
Central orchestrator for FlowNLP App. Wires recording, ASR, NLP,
context detection, formatting, rewriting, and storage together and
exposes Qt signals for the UI layer.

Signals
-------
state_changed(AppState)       — fired on every state transition
transcript_ready(str)         — raw ASR text just transcribed
result_ready(str, str)        — (record_id, formatted text) pipeline complete
rewrite_ready(str, dict)      — (record_id, rewrite result dict) rewrite done
error_occurred(str)           — human-readable error message
recordings_changed()          — library list has changed (add / confirm / delete)
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
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

APP_DATA = Path(__file__).resolve().parent.parent / "app_data"
AUDIO_DIR = APP_DATA / "audio"

ALLOWED_MODES  = {"Auto", "Chat", "Email", "Notes", "Search", "Code"}
ALLOWED_STYLES = {"faithful", "natural"}
ALLOWED_TONES  = {"Professional", "Friendly", "Concise"}


def _edit_ratio(source: str, final: str) -> float:
    src = source.casefold().split()
    tgt = final.casefold().split()
    if not src:
        return 0.0 if not tgt else 1.0
    prev = list(range(len(tgt) + 1))
    for i, sw in enumerate(src, 1):
        curr = [i]
        for j, tw in enumerate(tgt, 1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (sw != tw)))
        prev = curr
    return round(prev[-1] / len(src), 4)


def _word_error_rate(source: str, reference: str) -> float:
    src = source.casefold().split()
    ref = reference.casefold().split()
    if not ref:
        return 0.0 if not src else 1.0
    prev = list(range(len(ref) + 1))
    for i, sw in enumerate(src, 1):
        curr = [i]
        for j, rw in enumerate(ref, 1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (sw != rw)))
        prev = curr
    return round(prev[-1] / len(ref), 4)


class AppController(QObject):
    """Qt-signal-based controller for the full dictation lifecycle."""

    state_changed      = Signal(object)          # AppState
    transcript_ready   = Signal(str)             # raw ASR text
    result_ready       = Signal(str, str)        # (record_id, formatted text)
    rewrite_ready      = Signal(str, dict)       # (record_id, rewrite dict)
    error_occurred     = Signal(str)             # error message
    recordings_changed = Signal()               # library needs refresh

    def __init__(self, db: Database) -> None:
        super().__init__()
        self._db    = db
        self._state = AppState.IDLE

        AUDIO_DIR.mkdir(parents=True, exist_ok=True)

        # Subsystems (all lazy-loaded or pre-warmed in background)
        self._recorder  = SessionRecorder()
        self._asr: Optional[WhisperEngine] = None
        self._nlp: Optional[NLPPipeline]   = None
        self._natural_rewriter             = None
        self._context   = ContextDetector()
        self._injector  = TextInjector()

        self._formatters: dict[str, FormatterBase] = {
            "EMAIL":   EmailFormatter(),
            "CHAT":    ChatFormatter(),
            "NOTE":    NotesFormatter(),
            "SEARCH":  SearchFormatter(),
            "CODE":    CodeFormatter(),
        }

        # Per-session metadata
        self._active_app: str = ""
        self._user_mode:  str = ""
        self._current_id: str = ""

        # Hotkey
        hotkey_str = db.get_setting("hotkey", "<ctrl>+<shift>+<space>")
        self._hotkey_manager = HotkeyManager(
            on_start=self.start_recording,
            on_stop=self.stop_recording,
            hotkey=hotkey_str,
        )
        self._hotkey_manager.start()

        # Inference lock — one job at a time
        self._inference_lock = threading.Lock()

        # Pre-warm models in background
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
            logger.warning("start_recording ignored (state=%s)", self._state.name)
            return
        self._active_app = self._get_active_app()
        self._user_mode  = self._db.get_setting("output_mode", "Auto")
        self._set_state(AppState.RECORDING)
        try:
            self._recorder.start()
        except Exception as exc:
            logger.exception("Cannot start microphone: %s", exc)
            self.error_occurred.emit(str(exc))
            self._set_state(AppState.IDLE)
        else:
            logger.info("Recording started. Active app: '%s'", self._active_app)

    def stop_recording(self) -> None:
        if self._state != AppState.RECORDING:
            logger.warning("stop_recording ignored (state=%s)", self._state.name)
            return
        self._set_state(AppState.PROCESSING)
        try:
            audio = self._recorder.stop()
        except Exception as exc:
            logger.exception("Cannot stop microphone: %s", exc)
            self.error_occurred.emit(str(exc))
            self._set_state(AppState.IDLE)
            return

        if audio is None or len(audio) == 0:
            self.error_occurred.emit("No audio was captured.")
            self._set_state(AppState.IDLE)
            return

        threading.Thread(
            target=self._run_session, args=(audio,), daemon=True
        ).start()

    def _run_session(self, audio: np.ndarray) -> None:
        record_id = str(uuid.uuid4())
        self._current_id = record_id
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Save raw audio for potential replay
        audio_path = AUDIO_DIR / f"{record_id}.wav"
        try:
            import soundfile as sf  # type: ignore
            sf.write(str(audio_path), audio, SAMPLE_RATE)
        except Exception:
            audio_path = Path("")   # audio save is non-critical

        self._db.create_recording(
            record_id=record_id,
            created_at=created_at,
            audio_path=str(audio_path),
            mode=self._user_mode,
        )
        self.recordings_changed.emit()

        try:
            result = self._process_audio(audio, self._user_mode)
            self._db.update_recording_result(
                record_id=record_id,
                raw_text=result["raw_text"],
                faithful_text=result["faithful_text"],
                segments_json=json.dumps(result["segments"], ensure_ascii=False),
                asr_model=result["asr_model"],
            )
            # Optionally auto-inject into the focused window
            inject = self._db.get_setting("auto_inject", "false") == "true"
            if inject and result["faithful_text"]:
                try:
                    self._injector.insert(result["faithful_text"])
                except Exception as exc:
                    logger.warning("Text injection failed: %s", exc)

            # Auto-copy the corrected text to clipboard immediately
            if result["faithful_text"]:
                try:
                    from output.clipboard import ClipboardManager
                    ClipboardManager.copy(result["faithful_text"])
                    logger.info("Auto-copied corrected text to clipboard.")
                except Exception as exc:
                    logger.warning("Auto-copy failed: %s", exc)

            self.transcript_ready.emit(result["raw_text"])
            self.result_ready.emit(record_id, result["faithful_text"])
            self._set_state(AppState.COMPLETED)
        except Exception as exc:
            logger.exception("Session processing failed for %s: %s", record_id, exc)
            self._db.update_recording_error(record_id, str(exc))
            self.error_occurred.emit(str(exc))
            self._set_state(AppState.ERROR)
        finally:
            self.recordings_changed.emit()
            # Return to IDLE after a brief pause so the UI can show COMPLETED/ERROR
            import time
            time.sleep(1.5)
            self._set_state(AppState.IDLE)

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------

    def _process_audio(self, audio: np.ndarray, mode: str) -> dict:
        with self._inference_lock:
            self._ensure_asr()
            self._ensure_nlp()

            prefs   = self._db.get_preferences()
            glossary = prefs["vocabulary"]
            result   = self._asr.transcribe(
                audio,
                initial_prompt=(", ".join(glossary[:100]) or None),
            )
            raw_text = result.get("text", "").strip()
            if not raw_text:
                raise ValueError("No speech could be transcribed from this recording.")

            nlp_result = self._nlp.process(
                raw_text, personal_dictionary=set(glossary)
            )
            context = self._context.detect(
                nlp_result=nlp_result,
                active_app=self._active_app,
                user_mode=mode if mode != "Auto" else "",
            )
            formatter = self._formatters.get(context, ChatFormatter())
            faithful  = formatter.format(
                nlp_result.get("clean_text", raw_text), nlp_result
            ).strip()

            return {
                "raw_text":     raw_text,
                "faithful_text": faithful,
                "segments":     result.get("segments", []),
                "asr_model":    self._db.get_setting("whisper_model", "base"),
            }

    # ------------------------------------------------------------------
    # Reprocess existing recording
    # ------------------------------------------------------------------

    def reprocess_recording(self, record_id: str) -> None:
        rec = self._db.get_recording(record_id)
        if rec is None:
            self.error_occurred.emit("Recording not found.")
            return
        audio_path = Path(rec.get("audio_path", ""))
        if not audio_path.is_file():
            self.error_occurred.emit("Audio file not found for this recording.")
            return
        threading.Thread(
            target=self._reprocess_worker,
            args=(record_id, audio_path, rec["mode"]),
            daemon=True,
        ).start()

    def _reprocess_worker(
        self, record_id: str, audio_path: Path, mode: str
    ) -> None:
        try:
            import soundfile as sf  # type: ignore
            audio, _ = sf.read(str(audio_path), dtype="float32")
        except Exception as exc:
            self.error_occurred.emit(f"Cannot read audio: {exc}")
            return
        try:
            result = self._process_audio(audio, mode)
            self._db.update_recording_result(
                record_id=record_id,
                raw_text=result["raw_text"],
                faithful_text=result["faithful_text"],
                segments_json=json.dumps(result["segments"], ensure_ascii=False),
                asr_model=result["asr_model"],
            )
            self.result_ready.emit(record_id, result["faithful_text"])
        except Exception as exc:
            logger.exception("Reprocess failed: %s", exc)
            self._db.update_recording_error(record_id, str(exc))
            self.error_occurred.emit(str(exc))
        finally:
            self.recordings_changed.emit()

    # ------------------------------------------------------------------
    # Natural rewrite
    # ------------------------------------------------------------------

    def request_rewrite(self, record_id: str, tone: str) -> None:
        """Generate a natural rewrite for an existing recording (async)."""
        if tone not in ALLOWED_TONES:
            self.error_occurred.emit("Choose Professional, Friendly, or Concise.")
            return
        rec = self._db.get_recording(record_id)
        if rec is None or rec.get("status") in ("processing", "error"):
            self.error_occurred.emit("A successful transcript is required first.")
            return
        threading.Thread(
            target=self._rewrite_worker,
            args=(record_id, rec, tone),
            daemon=True,
        ).start()

    def _rewrite_worker(self, record_id: str, rec: dict, tone: str) -> None:
        try:
            with self._inference_lock:
                rewrite = self._generate_rewrite(rec, tone)
            trial_id = self._db.create_rewrite_trial(
                recording_id=record_id,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                asr_model=rec.get("asr_model", "base"),
                tone=tone,
                candidate=rewrite["text"],
                similarity=rewrite.get("similarity"),
                warnings_json=json.dumps(rewrite.get("warnings", []), ensure_ascii=False),
            )
            self._db.update_recording_rewrite(
                record_id=record_id,
                rewrite_text=rewrite["text"],
                trial_id=trial_id,
                similarity=rewrite.get("similarity"),
                warnings_json=json.dumps(rewrite.get("warnings", []), ensure_ascii=False),
                tone=tone,
            )
            rewrite["trial_id"] = trial_id
            self.rewrite_ready.emit(record_id, rewrite)
        except Exception as exc:
            logger.exception("Rewrite failed: %s", exc)
            self.error_occurred.emit(str(exc))

    def _generate_rewrite(self, rec: dict, tone: str) -> dict:
        from nlp.rewrite import NaturalRewriter

        self._ensure_nlp()
        if self._natural_rewriter is None:
            self._natural_rewriter = NaturalRewriter()

        prefs = self._db.get_preferences()
        required_terms = list(prefs["vocabulary"])
        try:
            entities = self._nlp._ner.extract(rec["faithful_text"])
            required_terms.extend(e["text"] for e in entities)
        except Exception:
            pass

        uncertain = self._db.uncertain_words(rec.get("segments_json", "[]"))
        required_terms.extend(
            w["text"].strip(".,!?;:") for w in uncertain if w.get("text")
        )

        mode = rec.get("mode", "")
        template = prefs["email_template"] if mode == "Email" else (
            prefs["notes_template"] if mode == "Notes" else ""
        )
        context_str = "\n".join(
            v for v in (prefs["writing_preferences"], template) if v.strip()
        )

        def similarity(a: str, b: str) -> float | None:
            try:
                return self._nlp._embeddings.similarity(a, b)
            except Exception:
                return None

        return self._natural_rewriter.rewrite(
            rec["faithful_text"],
            tone,
            context=context_str,
            required_terms=sorted(set(t for t in required_terms if t)),
            protected_markers=[w["text"] for w in uncertain if w.get("text")],
            similarity_fn=similarity,
        )

    # ------------------------------------------------------------------
    # Confirm recording
    # ------------------------------------------------------------------

    def confirm_recording(
        self,
        record_id: str,
        clean_text: str,
        output_style: str = "faithful",
        tone: str = "Professional",
        trial_id: int | None = None,
    ) -> None:
        rec = self._db.get_recording(record_id)
        if rec is None:
            self.error_occurred.emit("Recording not found.")
            return
        if not clean_text.strip():
            self.error_occurred.emit("Cannot confirm an empty transcript.")
            return
        ratio = _edit_ratio(rec.get("faithful_text", ""), clean_text)
        wer   = _word_error_rate(rec.get("raw_text", ""), clean_text)
        self._db.confirm_recording(
            record_id=record_id,
            clean_text=clean_text,
            output_style=output_style,
            tone=tone,
            edit_ratio=ratio,
            wer=wer,
            trial_id=trial_id,
        )
        self.recordings_changed.emit()

    # ------------------------------------------------------------------
    # Delete recording
    # ------------------------------------------------------------------

    def delete_recording(self, record_id: str) -> None:
        audio_path_str = self._db.get_audio_path(record_id)
        if audio_path_str:
            p = Path(audio_path_str)
            if p.is_file():
                try:
                    p.unlink()
                except Exception as exc:
                    logger.warning("Could not delete audio file: %s", exc)
        self._db.delete_recording(record_id)
        self.recordings_changed.emit()

    # ------------------------------------------------------------------
    # Hotkey update
    # ------------------------------------------------------------------

    def update_hotkey(self, hotkey: str) -> None:
        self._hotkey_manager.update_hotkey(hotkey)
        self._db.set_setting("hotkey", hotkey)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(self) -> None:
        logger.info("Warming up models…")
        try:
            self._ensure_asr()
            silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
            self._asr.transcribe(silence)
            logger.info("Warmup: Whisper ready.")
        except Exception as exc:
            logger.warning("Warmup Whisper failed: %s", exc)
        try:
            self._ensure_nlp()
            self._nlp.process("hello world")
            logger.info("Warmup: NLP pipeline ready.")
        except Exception as exc:
            logger.warning("Warmup NLP failed: %s", exc)
        try:
            self._recorder.warmup_vad()
            logger.info("Warmup: VAD ready.")
        except Exception as exc:
            logger.warning("Warmup VAD failed: %s", exc)
        logger.info("All models warmed up — ready to dictate.")
        self.state_changed.emit(self._state)

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _ensure_asr(self) -> None:
        if self._asr is None:
            model = self._db.get_setting("whisper_model", "base")
            self._asr = WhisperEngine(model_size=model)

    def _ensure_nlp(self) -> None:
        if self._nlp is None:
            self._nlp = NLPPipeline()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_active_app(self) -> str:
        try:
            from context.application import get_active_application
            return get_active_application()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._state == AppState.RECORDING:
            try:
                self._recorder.stop()
            except Exception:
                pass
        self._hotkey_manager.stop()
        self._db.close()
        logger.info("AppController shut down.")
