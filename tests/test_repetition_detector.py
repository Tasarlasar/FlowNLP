"""
Unit tests for Repetition and False-Start Detector
"""

import pytest
from nlp.repetition_detector import RepetitionDetector


@pytest.fixture
def detector():
    return RepetitionDetector()


def test_word_repetition(detector):
    raw = "I want to to go to the the store"
    cleaned = detector.remove(raw)
    assert cleaned == "I want to go to the store"


def test_ngram_repetition(detector):
    raw = "send the report send the report to Arun"
    cleaned = detector.remove(raw)
    assert cleaned == "send the report to Arun"


def test_empty_string(detector):
    assert detector.remove("") == ""
