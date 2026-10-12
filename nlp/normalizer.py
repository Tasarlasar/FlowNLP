"""
Text Normalizer
Cleans raw ASR output before any linguistic processing.

Handles:
- Excess whitespace
- Non-standard quotation marks
- Repeated punctuation
- Common ASR artifacts (e.g. "um-hm", stray symbols)
- Unicode normalisation
"""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)


class Normalizer:
    """Applies deterministic text normalisation rules to a raw transcript."""

    # Ordered list of (compiled_pattern, replacement) pairs
    _RULES: list[tuple] = [
        # Unicode normalise
        # (handled in method body, not regex)

        # Curly/smart quotes → straight quotes
        (re.compile(r"[\u2018\u2019]"), "'"),
        (re.compile(r"[\u201C\u201D]"), '"'),

        # Em-dash / en-dash → space
        (re.compile(r"[\u2013\u2014]"), " "),

        # Collapse multiple punctuation (e.g. "!!!" → "!")
        (re.compile(r"([!?]){2,}"), r"\1"),

        # Remove stray non-ASCII symbols (keep letters, digits, basic punct)
        (re.compile(r"[^\x00-\x7F]+"), " "),

        # Collapse multiple spaces / tabs
        (re.compile(r"[ \t]{2,}"), " "),

        # Remove leading/trailing whitespace per line
        (re.compile(r"^ +| +$", re.MULTILINE), ""),
    ]

    def normalize(self, text: str) -> str:
        """
        Return a cleaned version of `text`.

        Parameters
        ----------
        text : str
            Raw transcript from the ASR engine.
        """
        if not text:
            return ""

        # Unicode NFC normalisation
        text = unicodedata.normalize("NFC", text)

        for pattern, replacement in self._RULES:
            text = pattern.sub(replacement, text)

        return text.strip()
