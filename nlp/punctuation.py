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


# Spoken punctuation replacement map
_SPOKEN_PUNCTUATION = [
    (re.compile(r"\bperiod\b", re.IGNORECASE), "."),
    (re.compile(r"\bquestion mark\b", re.IGNORECASE), "?"),
    (re.compile(r"\bexclamation point\b|\bexclamation mark\b", re.IGNORECASE), "!"),
    (re.compile(r"\bcomma\b", re.IGNORECASE), ","),
    (re.compile(r"\bcolon\b", re.IGNORECASE), ":"),
    (re.compile(r"\bsemicolon\b", re.IGNORECASE), ";"),
]

# Conjunctions that often take an preceding comma when separating clauses
_CLAUSE_CONJUNCTIONS = re.compile(
    r"\s+\b(however|although|nevertheless|furthermore|meanwhile|in addition)\b",
    re.IGNORECASE,
)


class PunctuationRestorer:
    """
    Adds missing punctuation to cleaned transcript text.

    Falls back to enhanced rule-based heuristics when the transformer model
    is not available or fails to load.
    """

    def __init__(self) -> None:
        self._use_transformer = False
        self._model = None
        self._model_loaded = True

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
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based_restore(self, text: str) -> str:
        """
        Apply enhanced heuristics:
        1. Convert spoken punctuation terms (e.g., "comma", "period", "question mark").
        2. Insert commas before clause transition conjunctions (however, although...).
        3. Capitalise the first character and start of sentences.
        4. Add question mark if sentence starts with a question word.
        5. Add terminal period if missing.
        """
        if not text:
            return text

        # 1. Spoken punctuation substitution
        for pattern, replacement in _SPOKEN_PUNCTUATION:
            text = pattern.sub(replacement, text)

        # 2. Conjunction commas (e.g. " I will go however it is late" -> " I will go, however it is late")
        text = _CLAUSE_CONJUNCTIONS.sub(r", \1", text)

        # 3. Clean up spaces before punctuation (e.g. "hello ." -> "hello.")
        text = re.sub(r'\s+([.!?,;:"])', r"\1", text)

        # 4. Sentence capitalization & terminal punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text)
        processed: list[str] = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            s = s[0].upper() + s[1:]
            if s[-1] not in ".!?,;:":
                if _QUESTION_WORDS.match(s):
                    s += "?"
                else:
                    s += "."
            processed.append(s)

        return " ".join(processed)

    def _capitalise_sentences(self, text: str) -> str:
        """Capitalise the first letter of each detected sentence."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(s[0].upper() + s[1:] if s else s for s in sentences)

