"""
Intent Classifier
Determines what the user is trying to accomplish with their dictation.

Approach
--------
Level 1 — TF-IDF + Logistic Regression / SVM (classical baseline)
Level 2 — DistilBERT fine-tuned on intent data (advanced)

The classifier is lazily loaded; on first use it tries to load a saved
model from the `models/` directory and falls back to the TF-IDF baseline
with keyword heuristics if no trained model exists.
"""

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

INTENT_LABELS = [
    "GENERAL_TEXT",
    "CHAT",
    "EMAIL",
    "NOTE",
    "SEARCH",
    "QUESTION",
    "REQUEST",
    "CODE",
    "REMINDER",
]

# Keyword heuristics for zero-training fallback
_KEYWORD_MAP: dict[str, list[str]] = {
    "EMAIL":    ["send", "email", "forward", "cc", "bcc", "subject", "reply", "dear"],
    "SEARCH":   ["search", "look up", "google", "find", "what is", "how to", "where is"],
    "CODE":     ["function", "class", "import", "def ", "variable", "print", "return", "loop"],
    "NOTE":     ["note", "remember", "remind me", "todo", "task", "write down"],
    "QUESTION": ["who", "what", "where", "when", "why", "how", "is it", "are there"],
    "REMINDER": ["remind", "alarm", "schedule", "at ", "tomorrow", "on monday"],
    "CHAT":     ["hey", "hi", "hello", "thanks", "thank you", "lol", "haha"],
}


class IntentClassifier:
    """
    Classifies the intent of a cleaned transcript.

    Returns
    -------
    {
        "label": str,         # e.g. "EMAIL"
        "confidence": float,  # 0.0 – 1.0
        "method": str,        # "ml_model" | "tfidf" | "keyword"
    }
    """

    def __init__(
        self,
        model_dir: str = "models",
    ) -> None:
        self._model_dir = model_dir
        self._vectorizer = None
        self._clf = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        model_path = os.path.join(self._model_dir, "intent_model.pkl")
        vec_path = os.path.join(self._model_dir, "intent_vectorizer.pkl")

        if os.path.exists(model_path) and os.path.exists(vec_path):
            try:
                import pickle  # noqa: S403 — internal use only

                with open(vec_path, "rb") as f:
                    self._vectorizer = pickle.load(f)  # noqa: S301
                with open(model_path, "rb") as f:
                    self._clf = pickle.load(f)  # noqa: S301
                logger.info("Intent classifier loaded from %s.", self._model_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load intent model: %s", exc)
        else:
            logger.info(
                "No trained intent model found at '%s' — using keyword fallback.",
                model_path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str) -> dict:
        """Classify the intent of `text`."""
        self._load()

        if not text:
            return {"label": "GENERAL_TEXT", "confidence": 1.0, "method": "default"}

        if self._vectorizer is not None and self._clf is not None:
            return self._ml_classify(text)

        return self._keyword_classify(text)

    def train(self, texts: list[str], labels: list[str]) -> None:
        """
        Train (or re-train) the TF-IDF + SVM classifier and persist it.

        Parameters
        ----------
        texts : list[str]
        labels : list[str]   — must be values from INTENT_LABELS
        """
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.svm import LinearSVC  # type: ignore
        import pickle
        import os

        logger.info("Training intent classifier on %d samples …", len(texts))
        self._vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = self._vectorizer.fit_transform(texts)
        self._clf = LinearSVC()
        self._clf.fit(X, labels)

        os.makedirs(self._model_dir, exist_ok=True)
        with open(os.path.join(self._model_dir, "intent_vectorizer.pkl"), "wb") as f:
            pickle.dump(self._vectorizer, f)
        with open(os.path.join(self._model_dir, "intent_model.pkl"), "wb") as f:
            pickle.dump(self._clf, f)
        logger.info("Intent model saved to %s.", self._model_dir)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ml_classify(self, text: str) -> dict:
        X = self._vectorizer.transform([text])
        label = self._clf.predict(X)[0]
        # LinearSVC doesn't give probabilities; use 0.85 as fixed confidence
        return {"label": label, "confidence": 0.85, "method": "tfidf"}

    def _keyword_classify(self, text: str) -> dict:
        lower = text.lower()
        scores: dict[str, int] = {}
        for intent, keywords in _KEYWORD_MAP.items():
            scores[intent] = sum(1 for kw in keywords if kw in lower)

        best = max(scores, key=scores.__getitem__)
        if scores[best] == 0:
            return {"label": "GENERAL_TEXT", "confidence": 0.5, "method": "keyword"}
        total = sum(scores.values())
        confidence = round(scores[best] / total, 2) if total else 0.5
        return {"label": best, "confidence": confidence, "method": "keyword"}
