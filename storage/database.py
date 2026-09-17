"""
SQLite Storage Layer
Persists application settings and the personal dictionary.

Schema
------
settings(key TEXT PRIMARY KEY, value TEXT)
dictionary(word TEXT PRIMARY KEY)
"""

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path.home() / ".flownlp" / "flownlp.db"


class Database:
    """
    Thread-safe SQLite wrapper for FlowNLP persistent storage.
    The database file is created at `~/.flownlp/flownlp.db` on first run.
    """

    def __init__(self, path: Path = DB_PATH) -> None:
        self._path = path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Create the database file and tables if they do not exist."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("Database initialised at %s.", self._path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Return the stored value for `key`, or `default` if not found."""
        cursor = self._conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        """Upsert a setting key/value pair."""
        self._conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Personal dictionary
    # ------------------------------------------------------------------

    def get_dictionary_words(self) -> List[str]:
        """Return all words in the personal dictionary."""
        cursor = self._conn.execute("SELECT word FROM dictionary ORDER BY word")
        return [row["word"] for row in cursor.fetchall()]

    def add_word(self, word: str) -> None:
        """Add a word to the personal dictionary (no-op if already present)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO dictionary (word) VALUES (?)", (word,)
        )
        self._conn.commit()

    def remove_word(self, word: str) -> None:
        """Remove a word from the personal dictionary."""
        self._conn.execute("DELETE FROM dictionary WHERE word = ?", (word,))
        self._conn.commit()

    def set_dictionary_words(self, words: List[str]) -> None:
        """Replace the entire personal dictionary with `words`."""
        self._conn.execute("DELETE FROM dictionary")
        self._conn.executemany(
            "INSERT OR IGNORE INTO dictionary (word) VALUES (?)",
            [(w,) for w in words],
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dictionary (
                word TEXT PRIMARY KEY
            );
            """
        )
        self._conn.commit()
