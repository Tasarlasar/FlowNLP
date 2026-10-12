"""
SQLite Storage Layer — FlowNLP App
Persists recordings, settings, rewrite trials, and the personal dictionary.

Database location: ~/.flownlp/flownlp_app.db

Schema
------
recordings(id, created_at, audio_path, mode, raw_text, faithful_text,
           clean_text, segments_json, asr_model, output_style, tone,
           rewrite_text, rewrite_trial_id, rewrite_similarity,
           rewrite_warnings_json, edit_ratio, raw_to_confirmed_wer, status, error)
settings(key, value)
rewrite_trials(id, recording_id, created_at, asr_model, tone, candidate,
               similarity, warnings_json, accepted, edit_ratio)
dictionary(word)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path.home() / ".flownlp" / "flownlp_app.db"
UNCERTAIN_WORD_THRESHOLD = 0.55


class Database:
    """Thread-safe SQLite wrapper for FlowNLP App persistent storage."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self._path = path
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Create the database file and all tables on first run."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        self._run_migrations()
        logger.info("Database initialised at %s", self._path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )
        self._conn.commit()

    def get_all_settings(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM settings").fetchall()
        return {row["key"]: row["value"] for row in rows}

    # ------------------------------------------------------------------
    # Personal dictionary
    # ------------------------------------------------------------------

    def get_dictionary_words(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT word FROM dictionary ORDER BY word"
        ).fetchall()
        return [row["word"] for row in rows]

    def add_word(self, word: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO dictionary (word) VALUES (?)", (word,)
        )
        self._conn.commit()

    def remove_word(self, word: str) -> None:
        self._conn.execute("DELETE FROM dictionary WHERE word = ?", (word,))
        self._conn.commit()

    def set_dictionary_words(self, words: list[str]) -> None:
        self._conn.execute("DELETE FROM dictionary")
        self._conn.executemany(
            "INSERT OR IGNORE INTO dictionary (word) VALUES (?)",
            [(w.strip(),) for w in words if w.strip()],
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Recordings — write
    # ------------------------------------------------------------------

    def create_recording(
        self,
        record_id: str,
        created_at: str,
        audio_path: str,
        mode: str,
        output_style: str = "faithful",
        tone: str = "Professional",
    ) -> None:
        self._conn.execute(
            """INSERT INTO recordings
               (id, created_at, audio_path, mode, output_style, tone, status)
               VALUES (?, ?, ?, ?, ?, ?, 'processing')""",
            (record_id, created_at, audio_path, mode, output_style, tone),
        )
        self._conn.commit()

    def update_recording_result(
        self,
        record_id: str,
        raw_text: str,
        faithful_text: str,
        segments_json: str,
        asr_model: str,
    ) -> None:
        self._conn.execute(
            """UPDATE recordings
               SET raw_text = ?, faithful_text = ?, clean_text = ?,
                   segments_json = ?, asr_model = ?,
                   output_style = 'faithful', rewrite_text = '',
                   rewrite_trial_id = NULL, rewrite_similarity = NULL,
                   rewrite_warnings_json = '[]',
                   edit_ratio = NULL, raw_to_confirmed_wer = NULL,
                   status = 'draft', error = ''
               WHERE id = ?""",
            (raw_text, faithful_text, faithful_text,
             segments_json, asr_model, record_id),
        )
        self._conn.commit()

    def update_recording_error(self, record_id: str, error: str) -> None:
        self._conn.execute(
            "UPDATE recordings SET status = 'error', error = ? WHERE id = ?",
            (error, record_id),
        )
        self._conn.commit()

    def update_recording_rewrite(
        self,
        record_id: str,
        rewrite_text: str,
        trial_id: int,
        similarity: float | None,
        warnings_json: str,
        tone: str,
    ) -> None:
        self._conn.execute(
            """UPDATE recordings
               SET rewrite_text = ?, rewrite_trial_id = ?,
                   rewrite_similarity = ?, rewrite_warnings_json = ?, tone = ?
               WHERE id = ?""",
            (rewrite_text, trial_id, similarity, warnings_json, tone, record_id),
        )
        self._conn.commit()

    def confirm_recording(
        self,
        record_id: str,
        clean_text: str,
        output_style: str,
        tone: str,
        edit_ratio: float | None,
        wer: float | None,
        trial_id: int | None,
    ) -> None:
        self._conn.execute(
            """UPDATE recordings
               SET clean_text = ?, output_style = ?, tone = ?,
                   edit_ratio = ?, raw_to_confirmed_wer = ?,
                   rewrite_trial_id = ?, status = 'confirmed'
               WHERE id = ?""",
            (clean_text, output_style, tone, edit_ratio, wer, trial_id, record_id),
        )
        if trial_id is not None and output_style == "natural":
            ratio = edit_ratio
            self._conn.execute(
                """UPDATE rewrite_trials
                   SET accepted = 1, edit_ratio = ?
                   WHERE id = ?""",
                (ratio, trial_id),
            )
        self._conn.commit()

    def delete_recording(self, record_id: str) -> None:
        self._conn.execute("DELETE FROM recordings WHERE id = ?", (record_id,))
        self._conn.execute(
            "DELETE FROM rewrite_trials WHERE recording_id = ?", (record_id,)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Recordings — read
    # ------------------------------------------------------------------

    def get_recording(self, record_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM recordings WHERE id = ?", (record_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_recordings(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM recordings ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_audio_path(self, record_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT audio_path FROM recordings WHERE id = ?", (record_id,)
        ).fetchone()
        return row["audio_path"] if row else None

    # ------------------------------------------------------------------
    # Rewrite trials
    # ------------------------------------------------------------------

    def create_rewrite_trial(
        self,
        recording_id: str,
        created_at: str,
        asr_model: str,
        tone: str,
        candidate: str,
        similarity: float | None,
        warnings_json: str,
    ) -> int:
        cursor = self._conn.execute(
            """INSERT INTO rewrite_trials
               (recording_id, created_at, asr_model, tone, candidate,
                similarity, warnings_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (recording_id, created_at, asr_model, tone,
             candidate, similarity, warnings_json),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    # ------------------------------------------------------------------
    # Evaluation aggregates
    # ------------------------------------------------------------------

    def get_evaluation(self) -> dict:
        records = self._conn.execute(
            """SELECT asr_model, output_style, tone,
                      edit_ratio, raw_to_confirmed_wer
               FROM recordings WHERE status = 'confirmed'"""
        ).fetchall()
        trials = self._conn.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(accepted),0) AS accepted"
            " FROM rewrite_trials"
        ).fetchone()

        by_model: dict[str, int] = {}
        by_style: dict[str, int] = {}
        by_tone:  dict[str, int] = {}
        edit_ratios, wer_proxies = [], []

        for r in records:
            by_model[r["asr_model"]]    = by_model.get(r["asr_model"], 0) + 1
            by_style[r["output_style"]] = by_style.get(r["output_style"], 0) + 1
            by_tone[r["tone"]]          = by_tone.get(r["tone"], 0) + 1
            if r["edit_ratio"] is not None:
                edit_ratios.append(r["edit_ratio"])
            if r["raw_to_confirmed_wer"] is not None:
                wer_proxies.append(r["raw_to_confirmed_wer"])

        total_trials   = trials["total"]
        accepted_trials = trials["accepted"]
        return {
            "confirmed_count":    len(records),
            "rewrite_trials":     total_trials,
            "rewrite_acceptance": round(accepted_trials / total_trials, 3)
                                  if total_trials else None,
            "average_edit_ratio": round(sum(edit_ratios) / len(edit_ratios), 3)
                                  if edit_ratios else None,
            "average_wer_proxy":  round(sum(wer_proxies) / len(wer_proxies), 3)
                                  if wer_proxies else None,
            "by_model":           by_model,
            "by_style":           by_style,
            "by_tone":            by_tone,
        }

    # ------------------------------------------------------------------
    # Preferences helpers (vocabulary, writing prefs, templates)
    # ------------------------------------------------------------------

    def get_preferences(self) -> dict:
        keys = ("vocabulary", "writing_preferences",
                "email_template", "notes_template")
        rows = self._conn.execute(
            f"SELECT key, value FROM settings WHERE key IN "
            f"({','.join('?' for _ in keys)})",
            keys,
        ).fetchall()
        values = {row["key"]: row["value"] for row in rows}
        try:
            vocabulary = json.loads(values.get("vocabulary", "[]"))
        except json.JSONDecodeError:
            vocabulary = []
        return {
            "vocabulary":          vocabulary if isinstance(vocabulary, list) else [],
            "writing_preferences": values.get("writing_preferences", ""),
            "email_template":      values.get("email_template", ""),
            "notes_template":      values.get("notes_template", ""),
        }

    def set_preferences(
        self,
        vocabulary: list[str],
        writing_preferences: str,
        email_template: str,
        notes_template: str,
    ) -> None:
        pairs = [
            ("vocabulary",          json.dumps(vocabulary, ensure_ascii=False)),
            ("writing_preferences", writing_preferences),
            ("email_template",      email_template),
            ("notes_template",      notes_template),
        ]
        for key, value in pairs:
            self.set_setting(key, value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        d["confirmed"] = (d.get("status") == "confirmed")
        try:
            d["rewrite_warnings"] = json.loads(d.get("rewrite_warnings_json") or "[]")
        except json.JSONDecodeError:
            d["rewrite_warnings"] = []
        return d

    def uncertain_words(self, segments_json: str) -> list[dict]:
        try:
            segments = json.loads(segments_json or "[]")
        except json.JSONDecodeError:
            return []
        result = []
        for seg in segments:
            for word in seg.get("words", []):
                prob = word.get("probability")
                if prob is not None and prob < UNCERTAIN_WORD_THRESHOLD:
                    result.append({
                        "text":        word.get("text", "").strip(),
                        "start":       word.get("start", seg.get("start", 0)),
                        "end":         word.get("end",   seg.get("end",   0)),
                        "probability": round(float(prob), 2),
                    })
        return result

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS recordings (
                id                      TEXT PRIMARY KEY,
                created_at              TEXT NOT NULL,
                audio_path              TEXT NOT NULL DEFAULT '',
                mode                    TEXT NOT NULL DEFAULT 'Chat',
                raw_text                TEXT NOT NULL DEFAULT '',
                faithful_text           TEXT NOT NULL DEFAULT '',
                clean_text              TEXT NOT NULL DEFAULT '',
                segments_json           TEXT NOT NULL DEFAULT '[]',
                asr_model               TEXT NOT NULL DEFAULT 'base',
                output_style            TEXT NOT NULL DEFAULT 'faithful',
                tone                    TEXT NOT NULL DEFAULT 'Professional',
                rewrite_text            TEXT NOT NULL DEFAULT '',
                rewrite_trial_id        INTEGER,
                rewrite_similarity      REAL,
                rewrite_warnings_json   TEXT NOT NULL DEFAULT '[]',
                edit_ratio              REAL,
                raw_to_confirmed_wer    REAL,
                status                  TEXT NOT NULL DEFAULT 'processing',
                error                   TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rewrite_trials (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id  TEXT    NOT NULL,
                created_at    TEXT    NOT NULL,
                asr_model     TEXT    NOT NULL,
                tone          TEXT    NOT NULL,
                candidate     TEXT    NOT NULL,
                similarity    REAL,
                warnings_json TEXT    NOT NULL DEFAULT '[]',
                accepted      INTEGER NOT NULL DEFAULT 0,
                edit_ratio    REAL
            );

            CREATE TABLE IF NOT EXISTS dictionary (
                word TEXT PRIMARY KEY
            );
        """)
        self._conn.commit()

    def _run_migrations(self) -> None:
        """Add any columns that older DB files might be missing."""
        existing = {
            row["name"]
            for row in self._conn.execute(
                "PRAGMA table_info(recordings)"
            ).fetchall()
        }
        new_columns = {
            "faithful_text":         "TEXT NOT NULL DEFAULT ''",
            "segments_json":         "TEXT NOT NULL DEFAULT '[]'",
            "asr_model":             "TEXT NOT NULL DEFAULT 'base'",
            "output_style":          "TEXT NOT NULL DEFAULT 'faithful'",
            "tone":                  "TEXT NOT NULL DEFAULT 'Professional'",
            "rewrite_text":          "TEXT NOT NULL DEFAULT ''",
            "rewrite_trial_id":      "INTEGER",
            "rewrite_similarity":    "REAL",
            "rewrite_warnings_json": "TEXT NOT NULL DEFAULT '[]'",
            "edit_ratio":            "REAL",
            "raw_to_confirmed_wer":  "REAL",
            "audio_path":            "TEXT NOT NULL DEFAULT ''",
        }
        for col, defn in new_columns.items():
            if col not in existing:
                self._conn.execute(
                    f"ALTER TABLE recordings ADD COLUMN {col} {defn}"
                )
        self._conn.commit()
