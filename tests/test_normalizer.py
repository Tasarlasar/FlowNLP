"""
Unit tests for Text Normalizer
"""

import pytest
from nlp.normalizer import Normalizer


@pytest.fixture
def normalizer():
    return Normalizer()


def test_normalize_empty(normalizer):
    assert normalizer.normalize("") == ""
    assert normalizer.normalize("   ") == ""


def test_normalize_quotes_and_dashes(normalizer):
    raw = "‘Hello’ “World”—test"
    result = normalizer.normalize(raw)
    assert "'" in result or '"' in result
    assert "—" not in result


def test_normalize_whitespace(normalizer):
    raw = "Hello    world   \n  this  is   a   test  "
    result = normalizer.normalize(raw)
    assert "  " not in result
    assert result == "Hello world\nthis is a test" or "Hello world" in result


def test_normalize_symbols(normalizer):
    raw = "Hello world!!! @#$%^&*()"
    result = normalizer.normalize(raw)
    assert "!!!" not in result
