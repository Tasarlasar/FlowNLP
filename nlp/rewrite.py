"""Optional, local natural-language rewriting for reviewed dictation drafts."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)

REWRITE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REWRITE_TONES = {"Professional", "Friendly", "Concise"}
MIN_SEMANTIC_SIMILARITY = 0.78

_INTRO_RE = re.compile(
    r"^(?:certainly[,!]?\s*|sure[,!]?\s*|here(?:'s| is)\s+(?:a\s+)?(?:more\s+)?(?:natural\s+and\s+)?(?:\w+\s+)?(?:version|rewrite)\s*:?\s*)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d+(?:[./:-]\d+)*\b")


def clean_model_output(text: str) -> str:
    """Remove common chat preambles and quote wrappers from model output."""
    cleaned = text.strip()
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = _INTRO_RE.sub("", cleaned).strip()
    if "\n" in cleaned:
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) > 1 and lines[0].endswith(":"):
            cleaned = " ".join(lines[1:])
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'`":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def validate_rewrite(
    source: str,
    candidate: str,
    similarity: float | None,
    required_terms: Iterable[str] = (),
    protected_markers: Iterable[str] = (),
) -> list[str]:
    """Return review warnings when a suggestion may alter source details."""
    warnings: list[str] = []
    if not candidate.strip():
        return ["No rewrite was produced. The faithful cleanup remains available."]

    if similarity is None:
        warnings.append("Meaning similarity could not be checked. Compare both versions.")
    elif similarity < MIN_SEMANTIC_SIMILARITY:
        warnings.append("This rewrite may change the original meaning. Compare both versions carefully.")

    source_numbers = _NUMBER_RE.findall(source)
    candidate_numbers = _NUMBER_RE.findall(candidate)
    if sorted(source_numbers) != sorted(candidate_numbers):
        warnings.append("A number or date differs from the transcript. Verify it before saving.")

    lowered_candidate = candidate.casefold()
    for term in required_terms:
        if term.casefold() not in lowered_candidate:
            warnings.append(f"The rewrite may have changed or omitted '{term}'. Verify it.")

    for marker in protected_markers:
        if marker not in candidate:
            warnings.append("An uncertain audio span was not preserved by the rewrite.")
            break

    return warnings


class NaturalRewriter:
    """Run an instruction-tuned model locally and return a reviewable draft."""

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._unavailable = False

    def _load(self) -> None:
        if self._model is not None or self._unavailable:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            self._tokenizer = AutoTokenizer.from_pretrained(REWRITE_MODEL_ID)
            self._model = AutoModelForCausalLM.from_pretrained(
                REWRITE_MODEL_ID,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self._model.eval()
            logger.info("Local rewrite model loaded: %s", REWRITE_MODEL_ID)
        except Exception as exc:  # noqa: BLE001
            self._unavailable = True
            logger.warning("Local rewrite model unavailable (%s).", exc)

    def rewrite(
        self,
        text: str,
        tone: str,
        *,
        context: str = "",
        required_terms: Iterable[str] = (),
        protected_markers: Iterable[str] = (),
        similarity_fn: Callable[[str, str], float] | None = None,
    ) -> dict:
        if tone not in REWRITE_TONES:
            raise ValueError("Choose Professional, Friendly, or Concise tone.")
        if not text.strip():
            raise ValueError("There is no text to rewrite.")

        self._load()
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("The local rewrite model could not be loaded. Check the model download and try again.")

        import torch

        system_message = (
            "You edit dictated text. Return only one rewritten version. "
            "Preserve the exact meaning, requests, facts, numbers, names, and uncertainty. "
            "Do not add, omit, infer, explain, greet, or answer anything. "
            "Keep all uppercase protected markers exactly as written."
        )
        user_message = f"Rewrite this text in a {tone.lower()} tone. Preserve its meaning:\n{text}"
        if context.strip():
            user_message += f"\nUser's saved writing preferences (follow only when consistent with the text):\n{context.strip()}"

        prompt = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.inference_mode():
            generated = self._model.generate(
                **model_inputs,
                max_new_tokens=192,
                do_sample=False,
                repetition_penalty=1.05,
            )
        generated_tokens = generated[0][model_inputs["input_ids"].shape[1]:]
        candidate = clean_model_output(
            self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        )

        similarity = similarity_fn(text, candidate) if similarity_fn and candidate else None
        warnings = validate_rewrite(
            text,
            candidate,
            similarity,
            required_terms=required_terms,
            protected_markers=protected_markers,
        )
        return {
            "text": candidate,
            "tone": tone,
            "similarity": similarity,
            "warnings": warnings,
            "safe": not warnings,
        }