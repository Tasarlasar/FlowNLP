"""
Unit tests for Punctuation Restorer
"""

import pytest
from nlp.punctuation import PunctuationRestorer


@pytest.fixture
def restorer():
    return PunctuationRestorer()


def test_rule_based_terminal_period(restorer):
    raw = "hello world"
    result = restorer.restore(raw)
    assert result == "Hello world."


def test_question_mark_restoration(restorer):
    raw = "what time is the meeting"
    result = restorer.restore(raw)
    assert result == "What time is the meeting?"


def test_spoken_punctuation(restorer):
    raw = "hello comma world period"
    result = restorer.restore(raw)
    assert result == "Hello, world."


def test_conjunction_comma(restorer):
    raw = "i want to attend however i am busy"
    result = restorer.restore(raw)
    assert "however," in result or ", however" in result
