"""
Formatter Base Class
All context-specific formatters inherit from FormatterBase and implement
the `format` method.
"""

from abc import ABC, abstractmethod


class FormatterBase(ABC):
    """
    Transforms cleaned, structured NLP output into application-ready text.

    Parameters passed to `format`:
    - clean_text : str        — punctuated, grammar-corrected transcript
    - nlp_result : dict       — full NLP pipeline result dict
    """

    @abstractmethod
    def format(self, clean_text: str, nlp_result: dict) -> str:
        """Return context-appropriate text for injection."""
