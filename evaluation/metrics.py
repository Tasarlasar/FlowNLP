"""
Evaluation Metrics
Classification (intent / NER) and semantic preservation metrics.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Classification metrics (intent, NER)
# ------------------------------------------------------------------

def compute_classification_metrics(
    y_true: List[str], y_pred: List[str]
) -> dict:
    """
    Compute accuracy, precision, recall, and F1 for a classification task.

    Parameters
    ----------
    y_true : list[str]   — ground-truth labels
    y_pred : list[str]   — predicted labels

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1
    """
    try:
        from sklearn.metrics import (  # type: ignore
            accuracy_score,
            precision_recall_fscore_support,
        )

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        result = {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        }
        logger.debug("Classification metrics: %s", result)
        return result

    except ImportError:
        logger.warning("scikit-learn not installed — cannot compute metrics.")
        return {"accuracy": -1, "precision": -1, "recall": -1, "f1": -1}


def compute_ner_metrics(
    y_true: List[List[str]], y_pred: List[List[str]]
) -> dict:
    """
    Compute entity-level precision, recall, and F1.

    Parameters
    ----------
    y_true : list of entity label sequences
    y_pred : list of predicted entity label sequences
    """
    try:
        from seqeval.metrics import (  # type: ignore
            precision_score, recall_score, f1_score
        )
        return {
            "precision": round(precision_score(y_true, y_pred), 4),
            "recall":    round(recall_score(y_true, y_pred), 4),
            "f1":        round(f1_score(y_true, y_pred), 4),
        }
    except ImportError:
        logger.warning("seqeval not installed — using token-level fallback.")
        flat_true = [label for seq in y_true for label in seq]
        flat_pred = [label for seq in y_pred for label in seq]
        return compute_classification_metrics(flat_true, flat_pred)


# ------------------------------------------------------------------
# Semantic preservation
# ------------------------------------------------------------------

def semantic_preservation_score(original: str, processed: str) -> float:
    """
    Measure how well the processed text preserves the meaning of the original
    using cosine similarity of sentence embeddings.

    Returns float in [0, 1] — 1.0 means identical meaning.
    """
    try:
        from nlp.embeddings import EmbeddingEngine
        engine = EmbeddingEngine()
        score = engine.similarity(original, processed)
        logger.debug("Semantic preservation: %.4f", score)
        return score
    except Exception as exc:  # noqa: BLE001
        logger.warning("Semantic preservation failed: %s", exc)
        return -1.0


# ------------------------------------------------------------------
# Human evaluation helper
# ------------------------------------------------------------------

def average_human_scores(scores: List[dict]) -> dict:
    """
    Average a list of human evaluation score dicts.

    Each dict should contain numeric values for:
    readability, grammar, punctuation, meaning, naturalness, usefulness

    Returns the averages.
    """
    if not scores:
        return {}

    keys = scores[0].keys()
    totals = {k: 0.0 for k in keys}
    for entry in scores:
        for k in keys:
            totals[k] += float(entry.get(k, 0))

    n = len(scores)
    return {k: round(v / n, 2) for k, v in totals.items()}
