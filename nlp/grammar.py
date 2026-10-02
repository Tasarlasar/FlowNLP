"""
Grammar and Spelling Corrector
Uses LanguageTool as the primary backend with an optional transformer
model for advanced cases.

LanguageTool runs fully locally via language_tool_python (Java).
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

GEC_MODEL_ID = "vennify/t5-base-grammar-correction"


class GrammarCorrector:
    """
    Corrects grammar and spelling in a transcript.

    Primary:   LanguageTool (rule-based, language-model enhanced)
    Enhancement: Local T5 grammar-correction model.
    Fallback:  Return text unchanged if both correction backends are unavailable.
    """

    def __init__(self, language: str = "en-US") -> None:
        self._language = language
        self._tool = None
        self._available = False
        self._gec_model = None
        self._gec_tokenizer = None
        self._gec_unavailable = False

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
        corrected = self._correct_with_model(text)

        if self._available and self._tool is not None:
            try:
                corrected = self._tool.correct(corrected)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Grammar correction failed: %s", exc)

        if corrected != text:
            logger.debug("Grammar correction: '%s' → '%s'", text[:60], corrected[:60])
        return corrected

    def _correct_with_model(self, text: str) -> str:
        self._load_gec_model()
        if self._gec_model is None or self._gec_tokenizer is None:
            return text

        try:
            import torch

            inputs = self._gec_tokenizer(
                f"grammar: {text}",
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.inference_mode():
                generated = self._gec_model.generate(
                    **inputs,
                    num_beams=4,
                    max_new_tokens=256,
                )
            corrected = self._gec_tokenizer.decode(
                generated[0], skip_special_tokens=True
            ).strip()
            return corrected or text
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local grammar model failed: %s", exc)
            return text

    def _load_gec_model(self) -> None:
        if self._gec_model is not None or self._gec_unavailable:
            return

        try:
            from transformers import (  # type: ignore
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )

            self._gec_tokenizer = AutoTokenizer.from_pretrained(GEC_MODEL_ID)
            self._gec_model = AutoModelForSeq2SeqLM.from_pretrained(GEC_MODEL_ID)
            self._gec_model.eval()
            logger.info("Local grammar model loaded: %s", GEC_MODEL_ID)
        except Exception as exc:  # noqa: BLE001
            self._gec_unavailable = True
            logger.warning("Local grammar model unavailable (%s).", exc)

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
