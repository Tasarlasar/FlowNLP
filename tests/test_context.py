"""
Unit tests for Context Detector
"""

import pytest
from context.detector import ContextDetector


@pytest.fixture
def detector():
    return ContextDetector()


def test_user_mode_override(detector):
    ctx = detector.detect({}, active_app="Slack", user_mode="Email")
    assert ctx == "EMAIL"


def test_active_app_mapping(detector):
    ctx = detector.detect({}, active_app="Microsoft Outlook", user_mode="Auto")
    assert ctx == "EMAIL"

    ctx_chat = detector.detect({}, active_app="Slack.exe", user_mode="Auto")
    assert ctx_chat == "CHAT"

    ctx_code = detector.detect({}, active_app="Code.exe", user_mode="Auto")
    assert ctx_code == "CODE"


def test_intent_mapping_fallback(detector):
    nlp_result = {"intent": {"label": "SEARCH"}}
    ctx = detector.detect(nlp_result, active_app="UnknownApp", user_mode="Auto")
    assert ctx == "SEARCH"
