"""Tests for local rewrite output cleanup and preservation checks."""

from nlp.rewrite import clean_model_output, validate_rewrite


def test_clean_model_output_removes_preamble_and_quotes():
    raw = 'Certainly! Here is a professional version:\n\n"Please approve my request."'
    assert clean_model_output(raw) == "Please approve my request."


def test_rewrite_validator_accepts_similar_text_with_required_details():
    warnings = validate_rewrite(
        "Please approve workshop 17 on June 3.",
        "Please approve the workshop on June 3 (17).",
        0.91,
        required_terms={"workshop"},
    )
    assert warnings == []


def test_rewrite_validator_flags_semantic_drift_and_changed_number():
    warnings = validate_rewrite(
        "Please approve workshop 17.",
        "I recommend a new training program.",
        0.42,
        required_terms={"workshop"},
    )
    assert any("change the original meaning" in warning for warning in warnings)
    assert any("number or date differs" in warning for warning in warnings)
    assert any("changed or omitted 'workshop'" in warning for warning in warnings)


def test_rewrite_validator_requires_uncertain_markers_to_survive():
    warnings = validate_rewrite(
        "Please approve UNCERTAINSPAN0QZX.",
        "Please approve the request.",
        0.84,
        protected_markers={"UNCERTAINSPAN0QZX"},
    )
    assert any("uncertain audio span" in warning for warning in warnings)