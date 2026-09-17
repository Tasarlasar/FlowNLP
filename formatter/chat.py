"""
Chat Formatter
Keeps the text conversational and natural — minimal transformation.
"""

import logging

from formatter.base import FormatterBase

logger = logging.getLogger(__name__)


class ChatFormatter(FormatterBase):
    """Formats dictated text for chat applications (Slack, Teams, WhatsApp …)."""

    def format(self, clean_text: str, nlp_result: dict) -> str:
        if not clean_text:
            return ""

        # Chat text: just ensure it starts with a capital letter
        text = clean_text.strip()
        if text:
            text = text[0].upper() + text[1:]

        logger.debug("ChatFormatter output: %s", text[:80])
        return text
