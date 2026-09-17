"""
Sentence Embedding Engine
Converts text into dense vector representations using Sentence Transformers.

Used for:
- Semantic similarity between utterances
- Context detection support
- Intent classification support
- Evaluation (semantic preservation)
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"   # Fast, small, good quality


class EmbeddingEngine:
    """
    Encodes text into sentence-level embeddings.

    Parameters
    ----------
    model_name : str
        Any Sentence Transformers model name or local path.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self._model_name)
            logger.info("SentenceTransformer loaded: %s", self._model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SentenceTransformer unavailable: %s", exc)
            self._model = None

    def encode(self, text: str) -> Optional[np.ndarray]:
        """
        Encode `text` into a vector.

        Returns None if the model is unavailable.
        """
        if not text:
            return None
        self._load()
        if self._model is None:
            return None
        try:
            return self._model.encode(text, convert_to_numpy=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Encoding failed: %s", exc)
            return None

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Cosine similarity between the embeddings of two texts.
        Returns 0.0 if encoding fails.
        """
        vec_a = self.encode(text_a)
        vec_b = self.encode(text_b)

        if vec_a is None or vec_b is None:
            return 0.0

        dot = float(np.dot(vec_a, vec_b))
        norm = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if norm == 0:
            return 0.0
        return round(dot / norm, 4)
