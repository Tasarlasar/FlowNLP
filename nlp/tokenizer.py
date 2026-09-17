"""
Tokenizer
Wraps spaCy for tokenisation, POS tagging, and lemmatization.
Falls back to NLTK if spaCy is unavailable.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class Tokenizer:
    """
    Produces a list of token dicts from a normalised string.

    Each token dict contains:
        {
            "text":    str,   # surface form
            "lemma":   str,   # base form
            "pos":     str,   # coarse POS tag (NOUN, VERB, …)
            "tag":     str,   # fine-grained tag
            "is_stop": bool,  # whether it is a stopword
            "is_punct": bool, # whether it is punctuation
        }
    """

    def __init__(self) -> None:
        self._nlp = None
        self._backend: str = "none"

    def _load(self) -> None:
        if self._nlp is not None:
            return

        try:
            import spacy  # type: ignore
            self._nlp = spacy.load("en_core_web_sm")
            self._backend = "spacy"
            logger.info("Tokenizer using spaCy (en_core_web_sm).")
        except (ImportError, OSError):
            logger.warning(
                "spaCy/en_core_web_sm unavailable — falling back to NLTK."
            )
            import nltk  # type: ignore
            nltk.download("punkt", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("wordnet", quiet=True)
            self._nlp = nltk
            self._backend = "nltk"

    def tokenize(self, text: str) -> List[dict]:
        """Return a list of token dicts for `text`."""
        self._load()
        if not text:
            return []

        if self._backend == "spacy":
            return self._spacy_tokenize(text)
        return self._nltk_tokenize(text)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _spacy_tokenize(self, text: str) -> List[dict]:
        doc = self._nlp(text)
        return [
            {
                "text": tok.text,
                "lemma": tok.lemma_,
                "pos": tok.pos_,
                "tag": tok.tag_,
                "is_stop": tok.is_stop,
                "is_punct": tok.is_punct,
            }
            for tok in doc
        ]

    def _nltk_tokenize(self, text: str) -> List[dict]:
        from nltk.tokenize import word_tokenize  # type: ignore
        from nltk.tag import pos_tag  # type: ignore
        from nltk.stem import WordNetLemmatizer  # type: ignore

        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(text)
        tagged = pos_tag(words)
        return [
            {
                "text": word,
                "lemma": lemmatizer.lemmatize(word.lower()),
                "pos": tag,
                "tag": tag,
                "is_stop": False,
                "is_punct": not word.isalnum(),
            }
            for word, tag in tagged
        ]
