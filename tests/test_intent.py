"""
Unit tests for Intent Classifier
"""

import pytest
from nlp.intent import IntentClassifier


@pytest.fixture
def classifier():
    return IntentClassifier()


def test_keyword_fallback_email(classifier):
    res = classifier.classify("send an email to Arun regarding subject report")
    assert res["label"] in ("EMAIL", "REQUEST")
    assert res["confidence"] > 0.0


def test_keyword_fallback_search(classifier):
    res = classifier.classify("search for python sentence transformers tutorial")
    assert res["label"] == "SEARCH"


def test_keyword_fallback_code(classifier):
    res = classifier.classify("function to import pandas and return dataframe")
    assert res["label"] in ("CODE", "REQUEST")


def test_ml_train_and_predict(classifier, tmp_path):
    clf = IntentClassifier(model_dir=str(tmp_path))
    texts = [
        "please send an email to John",
        "forward this email message",
        "search for weather forecast online",
        "google search best restaurants near me",
    ]
    labels = ["EMAIL", "EMAIL", "SEARCH", "SEARCH"]
    clf.train(texts, labels)

    res1 = clf.classify("send an email message")
    assert res1["label"] == "EMAIL"

    res2 = clf.classify("search for news online")
    assert res2["label"] == "SEARCH"
