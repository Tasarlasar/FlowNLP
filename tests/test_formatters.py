"""
Unit tests for Context Formatters
"""

import pytest
from formatter.email import EmailFormatter
from formatter.chat import ChatFormatter
from formatter.notes import NotesFormatter
from formatter.search import SearchFormatter
from formatter.code import CodeFormatter


def test_email_formatter():
    fmt = EmailFormatter()
    res = fmt.format("i'm writing to let you know that it's ready", {})
    assert "I am" in res
    assert "it is" in res


def test_chat_formatter():
    fmt = ChatFormatter()
    res = fmt.format("hey there how are you", {})
    assert res == "Hey there how are you"


def test_search_formatter():
    fmt = SearchFormatter()
    res = fmt.format("please search for best python libraries?", {})
    assert res == "best python libraries"


def test_notes_formatter():
    fmt = NotesFormatter()
    res1 = fmt.format("first buy milk then call Arun finally submit report", {})
    assert "•" in res1

    res2 = fmt.format("bullet point clean workspace", {})
    assert res2 == "• Clean workspace"


def test_code_formatter():
    fmt = CodeFormatter()
    # Comments
    res1 = fmt.format("comment fix database connection bug", {})
    assert res1 == "# Fix database connection bug"

    # Variables
    res2 = fmt.format("variable user name", {})
    assert res2 == "user_name"

    # Functions
    res3 = fmt.format("function process audio", {})
    assert res3 == "process_audio()"

    # Classes
    res4 = fmt.format("class audio recorder", {})
    assert res4 == "AudioRecorder"
