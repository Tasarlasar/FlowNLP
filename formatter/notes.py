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


_BULLET_TRIGGER = re.compile(
    r"\b(?:bullet point|bullet|list item)\s*",
    re.IGNORECASE,
)


class NotesFormatter(FormatterBase):
    """Formats dictated text for note-taking applications."""

    def format(self, clean_text: str, nlp_result: dict) -> str:
        if not clean_text:
            return ""

        text = clean_text.strip()

        # Handle explicit bullet point dictations (e.g. "bullet point buy milk")
        if _BULLET_TRIGGER.search(text):
            text = _BULLET_TRIGGER.sub("• ", text)
            items = [line.strip() for line in text.split("\n") if line.strip()]
            formatted_items = []
            for item in items:
                if item.startswith("• "):
                    body = item[2:].strip()
                    formatted_items.append(f"• {body[0].upper() + body[1:] if body else ''}")
                else:
                    formatted_items.append(item)
            text = "\n".join(formatted_items)
            logger.debug("NotesFormatter explicit bullet output: %s", text[:80])
            return text

        # If list transition signals are detected, try to split into bullet points
        if _LIST_SIGNALS.search(text):
            text = self._to_bullets(text)

        logger.debug("NotesFormatter output: %s", text[:80])
        return text

    @staticmethod
    def _to_bullets(text: str) -> str:
        """Convert comma/conjunction-separated items into a bullet list."""
        parts = re.split(
            r"\b(?:first|second|third|then|next|also|finally|and then|lastly),?\s*",
            text,
            flags=re.IGNORECASE,
        )
        parts = [p.strip().rstrip(".") for p in parts if p.strip()]
        if len(parts) <= 1:
            return text
        return "\n".join(f"• {p[0].upper() + p[1:]}" for p in parts)

