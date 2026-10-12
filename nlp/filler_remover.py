"""
Filler Word Remover
Removes hesitation fillers from speech transcripts while preserving
context-important words.

Hybrid approach:
 - Rule-based: lexicon of known filler words/phrases
 - Context-aware: checks surrounding tokens to avoid removing
   semantically meaningful uses
"""

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Strict hesitation fillers (always removed)
STRICT_FILLERS: set[str] = {
    "um", "uh", "hmm", "mm", "ah", "er",
}

# Context-sensitive fillers (removed only when acting as discourse markers/interjections)
CONTEXT_FILLERS: set[str] = {
    "like", "basically", "literally", "actually",
    "right", "so", "well", "okay", "ok", "just", "yeah",
}

# Canonical filler lexicon (lower-cased)
FILLER_WORDS: set[str] = STRICT_FILLERS | CONTEXT_FILLERS

# Multi-word fillers require a separate phrase-level check
FILLER_PHRASES: list[str] = [
    "you know",
    "i mean",
    "kind of",
    "sort of",
]


class FillerRemover:
    """Strips filler words and phrases from a transcript with POS awareness."""

    def __init__(
        self,
        extra_fillers: Optional[set[str]] = None,
    ) -> None:
        self._fillers = FILLER_WORDS.copy()
        if extra_fillers:
            self._fillers.update(w.lower() for w in extra_fillers)

        # Build compiled regex for phrase-level fillers (longest first)
        sorted_phrases = sorted(FILLER_PHRASES, key=len, reverse=True)
        phrase_pattern = "|".join(
            r"(?:,\s*)?\b" + re.escape(p) + r"\b(?:\s*,)?"
            for p in sorted_phrases
        )
        self._phrase_re = re.compile(phrase_pattern, re.IGNORECASE)

    def remove(
        self,
        text: str,
        tokens: Optional[List[dict]] = None,
        personal_dictionary: Optional[set[str]] = None,
    ) -> str:
        """
        Return `text` with filler words/phrases removed.

        Parameters
        ----------
        text : str
            Normalised transcript.
        tokens : list[dict], optional
            Pre-computed token list from Tokenizer; used for context checks.
        personal_dictionary : set[str], optional
            User custom vocabulary to protect from removal.
        """
        if not text:
            return text

        dict_lower = {w.lower() for w in personal_dictionary} if personal_dictionary else set()

        # Build map from token word lower -> POS for quick lookup
        pos_map: dict[str, str] = {}
        if tokens:
            for tok in tokens:
                w_bare = re.sub(r"[^\w'-]", "", tok.get("text", "")).lower()
                if w_bare:
                    pos_map[w_bare] = tok.get("pos", "").upper()

        # 1. Remove multi-word filler phrases first
        text = self._phrase_re.sub("", text)

        # 2. Remove single-word fillers with POS and dictionary awareness
        words = text.split()
        cleaned: list[str] = []
        for word in words:
            bare = re.sub(r"[^\w'-]", "", word).lower()

            # Protect personal dictionary terms
            if bare in dict_lower:
                cleaned.append(word)
                continue

            # Strict fillers (um, uh, er) are always removed
            if bare in STRICT_FILLERS:
                logger.debug("Removed strict filler: '%s'", word)
                self._remove_preceding_comma(cleaned)
                continue

            # Context-sensitive fillers (like, so, just...)
            if bare in CONTEXT_FILLERS:
                pos = pos_map.get(bare, "")

                # If "like" is used as a verb, preposition (ADP), or conjunction (SCONJ), preserve it!
                if bare == "like" and pos in ("VERB", "ADP", "SCONJ"):
                    cleaned.append(word)
                    continue

                # If "so" is used as an adverb/conjunction leading a sentence, preserve it
                if bare in ("so", "just", "well", "right") and pos in ("ADV", "CONJ", "CCONJ", "ADJ", "NOUN"):
                    cleaned.append(word)
                    continue

                # Otherwise remove discourse filler
                logger.debug("Removed context filler: '%s' (POS: %s)", word, pos)
                self._remove_preceding_comma(cleaned)
                continue

            # Any other filler registered in self._fillers
            if bare in self._fillers and bare not in CONTEXT_FILLERS:
                logger.debug("Removed extra filler: '%s'", word)
                self._remove_preceding_comma(cleaned)
                continue

            cleaned.append(word)

        # 3. Tidy up leftover whitespace / double spaces
        result = " ".join(cleaned)
        result = re.sub(r"\s+([,.;:!?])", r"\1", result)
        result = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", result)
        result = re.sub(r"^[,;:!?]+\s*", "", result)
        result = re.sub(r"\s*[,;:]$", "", result)
        result = re.sub(r" {2,}", " ", result).strip()
        return result

    @staticmethod
    def _remove_preceding_comma(cleaned: list[str]) -> None:
        if cleaned and cleaned[-1].endswith(","):
            cleaned[-1] = cleaned[-1].rstrip(",")

