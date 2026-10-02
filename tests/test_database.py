"""
Unit tests for SQLite Database Storage Layer
"""

import pytest
from pathlib import Path
from storage.database import Database


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_flownlp.db"
    db = Database(path=db_file)
    db.initialise()
    yield db
    db.close()


def test_settings_crud(temp_db):
    assert temp_db.get_setting("hotkey") is None
    temp_db.set_setting("hotkey", "ctrl+shift+space")
    assert temp_db.get_setting("hotkey") == "ctrl+shift+space"

    temp_db.set_setting("hotkey", "ctrl+alt+d")
    assert temp_db.get_setting("hotkey") == "ctrl+alt+d"


def test_dictionary_management(temp_db):
    words = temp_db.get_dictionary_words()
    assert words == []

    temp_db.add_word("Aneesh")
    temp_db.add_word("FlowNLP")
    words = temp_db.get_dictionary_words()
    assert "Aneesh" in words
    assert "FlowNLP" in words

    temp_db.remove_word("Aneesh")
    words = temp_db.get_dictionary_words()
    assert "Aneesh" not in words
    assert "FlowNLP" in words

    temp_db.set_dictionary_words(["Alpha", "Beta"])
    words = temp_db.get_dictionary_words()
    assert words == ["Alpha", "Beta"]
