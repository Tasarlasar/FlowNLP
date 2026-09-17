"""
Grammar and Spelling Corrector
Uses LanguageTool as the primary backend with an optional transformer
model for advanced cases.

LanguageTool runs fully locally via language_tool_python (Java).
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class GrammarCorrector:
    """
    Corrects grammar and spelling in a transcript.

    Primary:   LanguageTool (rule-based, language-model enhanced)
    Fallback:  Return text unchanged if LanguageTool is unavailable.
    """

    def __init__(self, language: str = "en-US") -> None:
        self._language = language
        self._tool = None
        self._available = False

    def _load(self) -> None:
        if self._tool is not None or not self._available is False:
            return
        try:
            import language_tool_python  # type: ignore
            self._tool = language_tool_python.LanguageTool(self._language)
            self._available = True
            logger.info("LanguageTool loaded for %s.", self._language)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LanguageTool unavailable (%s) — grammar correction disabled.", exc
            )
            self._available = False

    def correct(self, text: str) -> str:
        """
        Apply grammar and spelling corrections to `text`.

        Parameters
        ----------
        text : str
            Transcript after filler and repetition removal.

        Returns
        -------
        str
            Corrected text, or original text if correction fails.
        """
        if not text:
            return text

        self._load()

        if not self._available or self._tool is None:
            return text

        try:
            corrected = self._tool.correct(text)
            if corrected != text:
                logger.debug("Grammar correction: '%s' → '%s'", text[:60], corrected[:60])
            return corrected
        except Exception as exc:  # noqa: BLE001
            logger.warning("Grammar correction failed: %s", exc)
            return text

    def get_matches(self, text: str) -> List[dict]:
        """Return raw LanguageTool match objects for diagnostic purposes."""
        self._load()
        if not self._available or self._tool is None:
            return []
        try:
            matches = self._tool.check(text)
            return [
                {
                    "message": m.message,
                    "offset": m.offset,
                    "length": m.errorLength,
                    "replacements": m.replacements[:3],
                    "rule_id": m.ruleId,
                }
                for m in matches
            ]
        except Exception:  # noqa: BLE001
            return []
