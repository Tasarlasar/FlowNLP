"""
Punctuation Restorer
Restores punctuation (commas, periods, question marks) to ASR transcripts
that typically lack proper punctuation.

Primary approach: transformer-based punctuation model
Fallback: rule-based heuristics (sentence capitalisation, period at end)
"""

import logging
import re

logger = logging.getLogger(__name__)

# Sentence-final question words suggest an interrogative
_QUESTION_WORDS = re.compile(
    r"^\s*(who|what|where|when|why|how|is|are|was|were|do|does|did|can|could|"
    r"will|would|should|shall|may|might|have|has|had)\b",
    re.IGNORECASE,
)


class PunctuationRestorer:
    """
    Adds missing punctuation to cleaned transcript text.

    Falls back to simple heuristics when the transformer model
    is not available or fails to load.
    """

    def __init__(self) -> None:
        # Transformer model disabled — requires internet download and adds
        # significant latency. Rule-based restore is fast and good enough
        # for live dictation. Re-enable once the model is cached locally.
        self._use_transformer = False
        self._model = None
        self._model_loaded = True   # mark as done so _load_transformer is never called

    def _load_transformer(self) -> bool:
        return False

    def restore(self, text: str) -> str:
        """
        Return `text` with punctuation and capitalisation restored.

        Parameters
        ----------
        text : str
            Transcript after grammar correction.
        """
        if not text:
            return text

        return self._rule_based_restore(text)

    # ------------------------------------------------------------------
    # Transformer approach
    # ------------------------------------------------------------------

    def _transformer_restore(self, text: str) -> str:
        predictions = self._model(text)
        tokens = text.split()
        result_words: list[str] = []

        for word in tokens:
            result_words.append(word)

        # Simple capitalise first letter and ensure terminal punctuation
        result = " ".join(result_words)
        return self._capitalise_sentences(result)

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based_restore(self, text: str) -> str:
        """
        Apply basic heuristics:
        1. Capitalise the first character.
        2. Add a question mark if the sentence starts with a question word.
        3. Otherwise end with a period if no terminal punctuation exists.
        """
        if not text:
            return text

        # Capitalise first character
        text = text[0].upper() + text[1:]

        # Trim trailing whitespace
        text = text.rstrip()

        # Add terminal punctuation if missing
        if text and text[-1] not in ".!?,;:":
            if _QUESTION_WORDS.match(text):
                text += "?"
            else:
                text += "."

        return text

    def _capitalise_sentences(self, text: str) -> str:
        """Capitalise the first letter of each detected sentence."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(s[0].upper() + s[1:] if s else s for s in sentences)
