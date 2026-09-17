"""
Code Formatter
Transforms dictated code-related text into programming-oriented format.

Handles:
- Converting natural-language variable names to snake_case / camelCase
- Stripping terminal punctuation
- Detecting code comments vs code statements
"""

import re
import logging

from formatter.base import FormatterBase

logger = logging.getLogger(__name__)

_COMMENT_SIGNALS = re.compile(
    r"^(comment|note|TODO|fixme|hack)\b",
    re.IGNORECASE,
)


class CodeFormatter(FormatterBase):
    """Formats dictated text for code editors."""

    def format(self, clean_text: str, nlp_result: dict) -> str:
        if not clean_text:
            return ""

        text = clean_text.strip()

        # Remove trailing sentence punctuation that isn't valid in code
        text = text.rstrip(".!?")

        # Detect comment-style dictation
        if _COMMENT_SIGNALS.match(text):
            # Strip the signal word and format as a Python comment
            body = _COMMENT_SIGNALS.sub("", text).strip()
            text = f"# {body[0].upper() + body[1:] if body else ''}"

        logger.debug("CodeFormatter output: %s", text[:80])
        return text
