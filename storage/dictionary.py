"""
Personal Dictionary Helper
High-level API on top of the Database layer.
Exposes the dictionary as a set for O(1) lookups used during NLP processing.
"""

import logging
from typing import Set

logger = logging.getLogger(__name__)


class PersonalDictionary:
    """
    Maintains an in-memory set of user-defined words backed by SQLite.

    Parameters
    ----------
    db : Database
        An already-initialised Database instance.
    """

    def __init__(self, db) -> None:
        self._db = db
        self._words: Set[str] = set()
        self._loaded = False

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Reload words from the database."""
        self._words = set(self._db.get_dictionary_words())
        self._loaded = True
        logger.debug("PersonalDictionary loaded: %d words.", len(self._words))

    def contains(self, word: str) -> bool:
        """Return True if `word` (case-insensitive) is in the dictionary."""
        if not self._loaded:
            self.load()
        return word.lower() in {w.lower() for w in self._words}

    def get_all(self) -> Set[str]:
        if not self._loaded:
            self.load()
        return set(self._words)

    # ------------------------------------------------------------------
    # Mutations (update both cache and DB)
    # ------------------------------------------------------------------

    def add(self, word: str) -> None:
        self._db.add_word(word)
        self._words.add(word)

    def remove(self, word: str) -> None:
        self._db.remove_word(word)
        self._words.discard(word)

    def replace_all(self, words: list[str]) -> None:
        self._db.set_dictionary_words(words)
        self._words = set(words)
