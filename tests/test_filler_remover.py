"""
Unit tests for Filler Word Remover
"""

import pytest
from nlp.filler_remover import FillerRemover


@pytest.fixture
def remover():
    return FillerRemover()


def test_strict_filler_removal(remover):
    raw = "um hello uh this is ah a test er"
    cleaned = remover.remove(raw)
    assert "um" not in cleaned.split()
    assert "uh" not in cleaned.split()
    assert "ah" not in cleaned.split()
    assert "er" not in cleaned.split()
    assert "hello this is a test" in cleaned


def test_phrase_filler_removal(remover):
    raw = "i mean send the report you know tomorrow"
    cleaned = remover.remove(raw)
    assert "i mean" not in cleaned
    assert "you know" not in cleaned
    assert "send the report" in cleaned


def test_filler_removal_cleans_surrounding_punctuation(remover):
    raw = "Um, I was, like, basically, you know, trying to, uh, send it."
    cleaned = remover.remove(raw)
    assert cleaned == "I was trying to send it."


def test_pos_aware_like_preservation(remover):
    # When tokens indicate "like" is a verb, it should be kept
    tokens = [
        {"text": "I", "pos": "PRON"},
        {"text": "like", "pos": "VERB"},
        {"text": "Python", "pos": "PROPN"},
    ]
    raw = "I like Python"
    cleaned = remover.remove(raw, tokens=tokens)
    assert "like" in cleaned.split()


def test_personal_dictionary_protection(remover):
    raw = "um Aneesh sent the file"
    cleaned = remover.remove(raw, personal_dictionary={"Aneesh"})
    assert "Aneesh" in cleaned
    assert "um" not in cleaned
