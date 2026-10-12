"""
Search Formatter
Condenses a dictated phrase into a concise search query.

Removes:
- Stop words that add no search value
- Filler phrases
- Trailing question marks
"""

import re
import logging

from formatter.base import FormatterBase

logger = logging.getLogger(__name__)

# High-frequency words that rarely add search value when leading a query
_LEAD_PHRASES = re.compile(
    r"^(please |can you |could you |i want to |i need to |"
    r"search for |look up |find |tell me |what is |how do i )",
    re.IGNORECASE,
)


class SearchFormatter(FormatterBase):
    """Formats dictated text as a concise search query."""

    def format(self, clean_text: str, nlp_result: dict) -> str:
        if not clean_text:
            return ""

        text = clean_text.strip().rstrip("?.")

        # Remove common leading phrases (repeatedly until none left)
        prev = None
        while prev != text:
            prev = text
            text = _LEAD_PHRASES.sub("", text).strip()

        # Lowercase for search consistency
        text = text.strip().lower()

        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)

        logger.debug("SearchFormatter output: %s", text[:80])
        return text
