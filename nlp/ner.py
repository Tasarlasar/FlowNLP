"""
Named Entity Recognition
Extracts named entities from cleaned transcript text.

Primary:  spaCy (en_core_web_sm / en_core_web_trf)
Fallback: BERT-based NER via Hugging Face Transformers
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Entities we care about in dictation scenarios
RELEVANT_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "LOC",
    "DATE", "TIME", "MONEY", "PRODUCT", "EVENT",
}


class NERProcessor:
    """
    Extracts named entities from text.

    Returns a list of dicts:
        [{"text": str, "label": str, "start": int, "end": int}, …]
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
            logger.info("NER using spaCy (en_core_web_sm).")
        except (ImportError, OSError):
            try:
                from transformers import pipeline  # type: ignore
                self._nlp = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple",
                )
                self._backend = "transformers"
                logger.info("NER using BERT transformer model.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("NER unavailable: %s", exc)
                self._backend = "none"

    def extract(self, text: str) -> List[dict]:
        """
        Extract named entities from `text`.

        Parameters
        ----------
        text : str
            Punctuated, cleaned transcript.

        Returns
        -------
        list of entity dicts with keys: text, label, start, end
        """
        if not text:
            return []

        self._load()

        if self._backend == "spacy":
            return self._spacy_extract(text)
        if self._backend == "transformers":
            return self._transformer_extract(text)
        return []

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _spacy_extract(self, text: str) -> List[dict]:
        doc = self._nlp(text)
        entities = []
        for ent in doc.ents:
            # Normalise GPE (geo-political entity) to LOCATION
            label = "LOCATION" if ent.label_ == "GPE" else ent.label_
            if ent.label_ in RELEVANT_ENTITY_TYPES or label in RELEVANT_ENTITY_TYPES:
                entities.append(
                    {
                        "text": ent.text,
                        "label": label,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                )
        return entities

    def _transformer_extract(self, text: str) -> List[dict]:
        raw = self._nlp(text)
        return [
            {
                "text": r["word"],
                "label": r["entity_group"],
                "start": r.get("start", -1),
                "end": r.get("end", -1),
            }
            for r in raw
        ]
