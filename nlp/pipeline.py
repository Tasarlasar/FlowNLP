"""
NLP Pipeline
Orchestrates every NLP stage in the correct order and returns a
single result dictionary consumed by the context and formatting layers.
"""

import logging
from typing import Optional

from nlp.normalizer import Normalizer
from nlp.tokenizer import Tokenizer
from nlp.filler_remover import FillerRemover
from nlp.repetition_detector import RepetitionDetector
from nlp.grammar import GrammarCorrector
from nlp.punctuation import PunctuationRestorer
from nlp.ner import NERProcessor
from nlp.intent import IntentClassifier
from nlp.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


class NLPPipeline:
    """
    Applies the full NLP processing chain to a raw ASR transcript.

    Processing order
    ----------------
    1.  Normalization
    2.  Tokenization
    3.  Filler word removal
    4.  Repetition / false-start detection
    5.  Grammar & spelling correction
    6.  Punctuation restoration
    7.  NER
    8.  Intent classification
    9.  Sentence embeddings
    """

    def __init__(self) -> None:
        self._normalizer = Normalizer()
        self._tokenizer = Tokenizer()
        self._filler_remover = FillerRemover()
        self._repetition_detector = RepetitionDetector()
        self._grammar_corrector = GrammarCorrector()
        self._punctuation_restorer = PunctuationRestorer()
        self._ner = NERProcessor()
        self._intent_classifier = IntentClassifier()
        self._embeddings = EmbeddingEngine()

    def process(
        self,
        raw_text: str,
        context: Optional[str] = None,
        personal_dictionary: Optional[set[str]] = None,
    ) -> dict:
        """
        Run the full NLP pipeline.

        Parameters
        ----------
        raw_text : str
            The raw transcript from the ASR engine.
        context : str, optional
            An optional hint about the target context (EMAIL, CHAT, …).
        personal_dictionary : set[str], optional
            User custom dictionary words to protect.

        Returns
        -------
        dict with keys:
            raw_text, normalized_text, tokens, clean_text,
            entities, intent, embedding, clean_after_grammar
        """
        logger.debug("NLP pipeline start: '%s'", raw_text[:80])
        result: dict = {"raw_text": raw_text}

        # 1. Normalisation
        normalized = self._normalizer.normalize(raw_text)
        result["normalized_text"] = normalized

        # 2. Tokenisation
        tokens = self._tokenizer.tokenize(normalized)
        result["tokens"] = tokens

        # 3. Filler removal (pass tokens and personal_dictionary)
        no_fillers = self._filler_remover.remove(
            normalized, tokens=tokens, personal_dictionary=personal_dictionary
        )
        result["text_after_filler"] = no_fillers

        # 4. Repetition detection
        no_reps = self._repetition_detector.remove(no_fillers)
        result["text_after_repetition"] = no_reps

        # 5. Grammar & spelling
        corrected = self._grammar_corrector.correct(no_reps)
        result["clean_after_grammar"] = corrected

        # 6. Punctuation
        punctuated = self._punctuation_restorer.restore(corrected)
        result["clean_text"] = punctuated

        # 7. NER
        entities = self._ner.extract(punctuated)
        result["entities"] = entities

        # 8. Intent
        intent = self._intent_classifier.classify(punctuated)
        result["intent"] = intent

        # 9. Embeddings
        embedding = self._embeddings.encode(punctuated)
        result["embedding"] = embedding

        logger.debug("NLP pipeline done. Intent=%s", intent.get("label"))
        return result
