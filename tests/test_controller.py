"""Tests for full-session controller processing."""

import numpy as np
from PySide6.QtCore import QObject

from app.controller import Controller
from app.states import AppState
from formatter.chat import ChatFormatter


class FakeRecorder:
    def extract_speech(self, audio):
        return audio


class FakeASR:
    def transcribe(self, audio):
        return {"text": "Aneesh will send the report."}


class FakeNLP:
    def __init__(self):
        self.dictionary = None

    def process(self, text, personal_dictionary):
        self.dictionary = personal_dictionary
        return {"clean_text": text, "intent": {"label": "CHAT"}}


class FakeDatabase:
    def get_dictionary_words(self):
        return ["Aneesh", "FlowNLP"]


class FakeContext:
    def detect(self, **kwargs):
        return "CHAT"


class FakeInjector:
    def __init__(self):
        self.inserted = []

    def insert(self, text):
        self.inserted.append(text)
        return True


def test_session_processing_uses_personal_dictionary_and_inserts_once():
    controller = Controller.__new__(Controller)
    QObject.__init__(controller)
    controller._state = AppState.PROCESSING
    controller._recorder = FakeRecorder()
    controller._asr = FakeASR()
    controller._nlp = FakeNLP()
    controller._db = FakeDatabase()
    controller._context = FakeContext()
    controller._injector = FakeInjector()
    controller._active_app = ""
    controller._user_mode = "Auto"
    controller._formatters = {"CHAT": ChatFormatter()}

    controller._process_session(np.zeros(1, dtype=np.float32))

    assert controller._nlp.dictionary == {"Aneesh", "FlowNLP"}
    assert controller._injector.inserted == ["Aneesh will send the report."]
    assert controller.state == AppState.IDLE