"""
Email Formatter
Converts a cleaned transcript into professional email-ready text.

Applies:
- Professional tone adjustments
- Sentence capitalisation
- Greeting / sign-off detection (best-effort)
"""

import re
import logging

from formatter.base import FormatterBase

logger = logging.getLogger(__name__)


class EmailFormatter(FormatterBase):
    """Formats dictated text for use in email clients."""

    # Contractions → formal equivalents
    _CONTRACTIONS = {
        r"\bi'm\b": "I am",
        r"\bdon't\b": "do not",
        r"\bcan't\b": "cannot",
        r"\bwon't\b": "will not",
        r"\bdidn't\b": "did not",
        r"\bisn't\b": "is not",
        r"\baren't\b": "are not",
        r"\bwasn't\b": "was not",
        r"\bweren't\b": "were not",
        r"\bhasn't\b": "has not",
        r"\bhaven't\b": "have not",
        r"\bhadn't\b": "had not",
        r"\bwouldn't\b": "would not",
        r"\bshouldn't\b": "should not",
        r"\bcouldn't\b": "could not",
        r"\bit's\b": "it is",
        r"\bthat's\b": "that is",
        r"\bthey're\b": "they are",
        r"\bwe're\b": "we are",
        r"\byou're\b": "you are",
        r"\bhe's\b": "he is",
        r"\bshe's\b": "she is",
    }

    def format(self, clean_text: str, nlp_result: dict) -> str:
        if not clean_text:
            return ""

        text = clean_text

        # Expand contractions for a professional tone
        for pattern, replacement in self._CONTRACTIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Ensure first letter is capitalised
        text = text[0].upper() + text[1:]

        logger.debug("EmailFormatter output: %s", text[:80])
        return text
