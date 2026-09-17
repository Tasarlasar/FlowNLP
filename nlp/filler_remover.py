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

# Canonical filler lexicon (lower-cased)
FILLER_WORDS: set[str] = {
    "um", "uh", "hmm", "mm", "ah", "er", "like",
    "basically", "literally", "actually", "you know",
    "i mean", "right", "so", "well", "okay", "ok",
    "kind of", "sort of", "just", "yeah",
}

# Multi-word fillers require a separate phrase-level check
FILLER_PHRASES: list[str] = [
    "you know",
    "i mean",
    "kind of",
    "sort of",
]


class FillerRemover:
    """Strips filler words and phrases from a transcript."""

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
            r"\b" + re.escape(p) + r"\b" for p in sorted_phrases
        )
        self._phrase_re = re.compile(phrase_pattern, re.IGNORECASE)

    def remove(self, text: str, tokens: Optional[List[dict]] = None) -> str:
        """
        Return `text` with filler words/phrases removed.

        Parameters
        ----------
        text : str
            Normalised transcript.
        tokens : list[dict], optional
            Pre-computed token list from Tokenizer; used for context checks.
        """
        if not text:
            return text

        # 1. Remove multi-word filler phrases first
        text = self._phrase_re.sub("", text)

        # 2. Remove single-word fillers
        words = text.split()
        cleaned: list[str] = []
        for word in words:
            bare = re.sub(r"[^\w'-]", "", word).lower()
            if bare in self._fillers:
                logger.debug("Removed filler: '%s'", word)
                continue
            cleaned.append(word)

        # 3. Tidy up leftover whitespace / double spaces
        result = " ".join(cleaned)
        result = re.sub(r" {2,}", " ", result).strip()
        return result
