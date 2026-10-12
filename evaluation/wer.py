"""
WER / CER Calculator
Wraps the `jiwer` library to compute Word Error Rate and
Character Error Rate between a reference and a hypothesis transcript.
"""

import logging
from typing import Union

logger = logging.getLogger(__name__)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER).

    Parameters
    ----------
    reference : str
        Ground-truth transcript.
    hypothesis : str
        ASR or pipeline output.

    Returns
    -------
    float — WER value (0.0 = perfect, 1.0 = completely wrong).
    """
    try:
        from jiwer import wer  # type: ignore
        score = wer(reference, hypothesis)
        logger.debug("WER: %.4f", score)
        return round(score, 4)
    except ImportError:
        logger.warning("jiwer not installed — returning -1 for WER.")
        return -1.0


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER).

    Returns
    -------
    float — CER value (0.0 = perfect).
    """
    try:
        from jiwer import cer  # type: ignore
        score = cer(reference, hypothesis)
        logger.debug("CER: %.4f", score)
        return round(score, 4)
    except ImportError:
        logger.warning("jiwer not installed — returning -1 for CER.")
        return -1.0


def compare_transcripts(reference: str, raw_asr: str, flownlp_output: str) -> dict:
    """
    Compare raw ASR vs FlowNLP against a reference transcript.

    Returns a dict with WER/CER for both outputs side-by-side.
    """
    return {
        "reference": reference,
        "baseline": {
            "text": raw_asr,
            "wer": compute_wer(reference, raw_asr),
            "cer": compute_cer(reference, raw_asr),
        },
        "flownlp": {
            "text": flownlp_output,
            "wer": compute_wer(reference, flownlp_output),
            "cer": compute_cer(reference, flownlp_output),
        },
    }
