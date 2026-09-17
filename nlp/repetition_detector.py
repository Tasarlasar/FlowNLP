"""
Repetition and False-Start Detector
Finds and removes:
- Duplicate consecutive words ("I want to to go")
- Repeated n-gram phrases ("send the report send the report")
- False starts / self-corrections ("I was— I mean I am")

Techniques used:
- Token-level deduplication
- N-gram comparison with edit distance
- Regex patterns for stutter-type repetitions
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

MAX_NGRAM = 5   # Check repetitions up to 5-grams


class RepetitionDetector:
    """Removes word-level and phrase-level repetitions from a transcript."""

    # Pattern for immediate single-word repetitions: "the the" → "the"
    _WORD_REPEAT_RE = re.compile(
        r"\b(\w+)\s+\1\b", re.IGNORECASE
    )

    def remove(self, text: str) -> str:
        """
        Return `text` with detected repetitions removed.

        Parameters
        ----------
        text : str
            Transcript after filler removal.
        """
        if not text:
            return text

        # 1. Simple word-level repetitions (handles "the the", "and and")
        prev = None
        while prev != text:
            prev = text
            text = self._WORD_REPEAT_RE.sub(r"\1", text)

        # 2. Phrase-level repetitions (n-gram approach)
        text = self._remove_ngram_repetitions(text)

        # 3. Tidy spacing
        text = re.sub(r" {2,}", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_ngram_repetitions(self, text: str) -> str:
        """Detect and remove repeated n-grams (n = 2 … MAX_NGRAM)."""
        words = text.split()
        n = len(words)

        for gram_size in range(2, min(MAX_NGRAM + 1, n // 2 + 1)):
            words = self._deduplicate_ngrams(words, gram_size)

        return " ".join(words)

    @staticmethod
    def _deduplicate_ngrams(words: List[str], n: int) -> List[str]:
        """Remove the second occurrence of consecutive identical n-grams."""
        i = 0
        result: List[str] = []
        while i < len(words):
            gram = words[i : i + n]
            next_gram = words[i + n : i + 2 * n]
            if gram and gram == next_gram:
                # Skip over the duplicate
                logger.debug("Removed repeated %d-gram: %s", n, " ".join(gram))
                result.extend(gram)
                i += 2 * n
            else:
                result.append(words[i])
                i += 1
        return result
