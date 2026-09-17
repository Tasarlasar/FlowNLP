"""
Notes Formatter
Structures dictated text as a note entry.
Detects list-like content and formats accordingly.
"""

import re
import logging
from datetime import datetime

from formatter.base import FormatterBase

logger = logging.getLogger(__name__)

# Keywords that suggest a list should be created
_LIST_SIGNALS = re.compile(
    r"\b(first|second|third|then|next|also|finally|and then|lastly)\b",
    re.IGNORECASE,
)


class NotesFormatter(FormatterBase):
    """Formats dictated text for note-taking applications."""

    def format(self, clean_text: str, nlp_result: dict) -> str:
        if not clean_text:
            return ""

        text = clean_text.strip()

        # If list signals are detected, try to split into bullet points
        if _LIST_SIGNALS.search(text):
            text = self._to_bullets(text)

        logger.debug("NotesFormatter output: %s", text[:80])
        return text

    @staticmethod
    def _to_bullets(text: str) -> str:
        """Convert comma/conjunction-separated items into a bullet list."""
        # Split on list-signal words
        parts = re.split(
            r"\b(?:first|second|third|then|next|also|finally|and then|lastly),?\s*",
            text,
            flags=re.IGNORECASE,
        )
        parts = [p.strip().rstrip(".") for p in parts if p.strip()]
        if len(parts) <= 1:
            return text
        return "\n".join(f"• {p[0].upper() + p[1:]}" for p in parts)
