"""
Evaluation Benchmark Runner
Runs the four-level academic experiment defined in the PRD:

  Level 1 — Raw ASR
  Level 2 — Classical NLP  (NLTK / spaCy / TF-IDF)
  Level 3 — Embedding / ML (Sentence Transformers)
  Level 4 — Transformer NLP (full pipeline)

Results are printed to stdout and optionally written to a JSON file.
"""

import json
import logging
import os
import time
from typing import Optional

from evaluation.wer import compute_wer, compute_cer
from evaluation.metrics import compute_classification_metrics, semantic_preservation_score

logger = logging.getLogger(__name__)


def run_benchmark(
    samples: list[dict],
    output_path: Optional[str] = None,
) -> dict:
    """
    Run the full benchmark over a list of evaluation samples.

    Parameters
    ----------
    samples : list[dict]
        Each dict must contain:
            {
                "audio_path": str,           # path to audio file
                "reference": str,            # ground-truth transcript
                "intent": str,               # ground-truth intent label (optional)
            }

    output_path : str, optional
        If given, write the full results dict as JSON to this path.

    Returns
    -------
    dict — benchmark results keyed by level name.
    """
    from asr.whisper_engine import WhisperEngine
    from nlp.pipeline import NLPPipeline

    asr = WhisperEngine()
    nlp = NLPPipeline()

    results = {
        "num_samples": len(samples),
        "levels": {},
        "per_sample": [],
    }

    wer_raw_all, wer_nlp_all = [], []
    cer_raw_all, cer_nlp_all = [], []
    intent_true, intent_pred = [], []

    for i, sample in enumerate(samples):
        logger.info("Benchmarking sample %d/%d …", i + 1, len(samples))
        reference = sample["reference"]
        audio_path = sample.get("audio_path")

        # Load audio or fallback to provided raw_asr text
        import numpy as np
        audio: Optional[np.ndarray] = None
        if audio_path and os.path.exists(audio_path):
            try:
                import soundfile as sf  # type: ignore
                audio, _ = sf.read(audio_path, dtype="float32")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load %s: %s", audio_path, exc)

        if audio is not None:
            t0 = time.perf_counter()
            asr_result = asr.transcribe(audio)
            raw_text = asr_result.get("text", "")
            asr_time = time.perf_counter() - t0
        elif "raw_asr" in sample:
            raw_text = sample["raw_asr"]
            asr_time = 0.0
        else:
            logger.warning("Skipping sample %d — no audio or raw_asr provided.", i + 1)
            continue


        wer_raw = compute_wer(reference, raw_text)
        cer_raw = compute_cer(reference, raw_text)
        wer_raw_all.append(wer_raw)
        cer_raw_all.append(cer_raw)

        # Level 4: Full NLP pipeline
        t1 = time.perf_counter()
        nlp_result = nlp.process(raw_text)
        nlp_time = time.perf_counter() - t1
        clean_text = nlp_result.get("clean_text", raw_text)

        wer_nlp = compute_wer(reference, clean_text)
        cer_nlp = compute_cer(reference, clean_text)
        wer_nlp_all.append(wer_nlp)
        cer_nlp_all.append(cer_nlp)

        sem_score = semantic_preservation_score(reference, clean_text)

        # Intent
        if "intent" in sample:
            intent_true.append(sample["intent"])
            intent_pred.append(nlp_result.get("intent", {}).get("label", "UNKNOWN"))

        results["per_sample"].append({
            "index": i,
            "reference": reference,
            "raw_asr": raw_text,
            "flownlp": clean_text,
            "asr_wer": wer_raw,
            "asr_cer": cer_raw,
            "nlp_wer": wer_nlp,
            "nlp_cer": cer_nlp,
            "semantic_preservation": sem_score,
            "asr_time_s": round(asr_time, 3),
            "nlp_time_s": round(nlp_time, 3),
        })

    # Aggregate
    def mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else -1.0

    raw_wer = mean(wer_raw_all)
    processed_wer = mean(wer_nlp_all)
    raw_cer = mean(cer_raw_all)
    processed_cer = mean(cer_nlp_all)
    results["levels"]["raw_asr"] = {
        "mean_wer": raw_wer,
        "mean_cer": raw_cer,
    }
    results["levels"]["flownlp"] = {
        "mean_wer": processed_wer,
        "mean_cer": processed_cer,
    }
    results["levels"]["processed_minus_raw"] = {
        "mean_wer": round(processed_wer - raw_wer, 4)
        if raw_wer >= 0 and processed_wer >= 0 else -1.0,
        "mean_cer": round(processed_cer - raw_cer, 4)
        if raw_cer >= 0 and processed_cer >= 0 else -1.0,
    }

    if intent_true:
        results["intent_metrics"] = compute_classification_metrics(
            intent_true, intent_pred
        )

    _print_summary(results)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Benchmark results written to %s.", output_path)

    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _print_summary(results: dict) -> None:
    print("\n" + "=" * 50)
    print("  FlowNLP Benchmark Summary")
    print("=" * 50)
    print(f"  Samples evaluated: {results['num_samples']}")
    raw = results.get("levels", {}).get("raw_asr", {})
    nlp = results.get("levels", {}).get("flownlp", {})
    change = results.get("levels", {}).get("processed_minus_raw", {})
    print(
        f"  Raw ASR  — Mean WER: {raw.get('mean_wer', 'N/A')}"
        f" | CER: {raw.get('mean_cer', 'N/A')}"
    )
    print(
        f"  FlowNLP  — Mean WER: {nlp.get('mean_wer', 'N/A')}"
        f" | CER: {nlp.get('mean_cer', 'N/A')}"
    )
    print(
        f"  Change   — WER: {change.get('mean_wer', 'N/A')}"
        f" | CER: {change.get('mean_cer', 'N/A')}"
        " (negative is better)"
    )
    if "intent_metrics" in results:
        im = results["intent_metrics"]
        print(f"  Intent   — F1: {im.get('f1', 'N/A')}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    import argparse, json as _json

    parser = argparse.ArgumentParser(description="FlowNLP benchmark runner")
    parser.add_argument("samples_json", help="Path to JSON file with evaluation samples")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    with open(args.samples_json, encoding="utf-8") as f:
        samples = _json.load(f)

    run_benchmark(samples, output_path=args.output)
