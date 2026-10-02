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

_VARIABLE_SIGNAL = re.compile(r"^(variable|var)\s+(.+)", re.IGNORECASE)
_FUNCTION_SIGNAL = re.compile(r"^(function|def)\s+(.+)", re.IGNORECASE)
_CLASS_SIGNAL = re.compile(r"^class\s+(.+)", re.IGNORECASE)
_CONSTANT_SIGNAL = re.compile(r"^constant\s+(.+)", re.IGNORECASE)


class CodeFormatter(FormatterBase):
    """Formats dictated text for code editors."""

    def format(self, clean_text: str, nlp_result: dict) -> str:
        if not clean_text:
            return ""

        text = clean_text.strip().rstrip(".!?")

        # 1. Detect comment-style dictation
        if _COMMENT_SIGNALS.match(text):
            body = _COMMENT_SIGNALS.sub("", text).strip()
            text = f"# {body[0].upper() + body[1:] if body else ''}"
            logger.debug("CodeFormatter comment output: %s", text)
            return text

        # 2. Variable / Identifier dictation
        m_var = _VARIABLE_SIGNAL.match(text)
        if m_var:
            words = re.findall(r"\w+", m_var.group(2))
            return "_".join(w.lower() for w in words)

        m_func = _FUNCTION_SIGNAL.match(text)
        if m_func:
            words = re.findall(r"\w+", m_func.group(2))
            return "_".join(w.lower() for w in words) + "()"

        m_class = _CLASS_SIGNAL.match(text)
        if m_class:
            words = re.findall(r"\w+", m_class.group(1))
            return "".join(w.capitalize() for w in words)

        m_const = _CONSTANT_SIGNAL.match(text)
        if m_const:
            words = re.findall(r"\w+", m_const.group(1))
            return "_".join(w.upper() for w in words)

        logger.debug("CodeFormatter output: %s", text[:80])
        return text

