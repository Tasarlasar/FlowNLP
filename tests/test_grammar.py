"""Tests for grammar correction orchestration."""

from nlp.grammar import GrammarCorrector


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        self.input_text = text
        return {"input_ids": [1]}

    def decode(self, generated, skip_special_tokens):
        return "Based on your resume, you will be able to explain every technology tool mentioned."


class FakeModel:
    def eval(self):
        return self

    def generate(self, **kwargs):
        return [[2]]


def test_grammar_model_correction_is_used():
    corrector = GrammarCorrector()
    corrector._gec_model = FakeModel()
    corrector._gec_tokenizer = FakeTokenizer()

    corrected = corrector.correct(
        "Based on your resume, you will be able to explain every technology tools mentioned."
    )

    assert corrected == "Based on your resume, you will be able to explain every technology tool mentioned."