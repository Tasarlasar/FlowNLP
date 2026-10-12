"""Tests for the raw-versus-processed benchmark report."""

import asr.whisper_engine as whisper_module
import evaluation.benchmark as benchmark_module
import nlp.pipeline as pipeline_module


def test_benchmark_reports_raw_and_processed_transcripts(monkeypatch, capsys):
    class FakeASR:
        pass

    class FakeNLP:
        def process(self, raw_text):
            return {"clean_text": "processed transcript", "intent": {"label": "NOTE"}}

    monkeypatch.setattr(whisper_module, "WhisperEngine", FakeASR)
    monkeypatch.setattr(pipeline_module, "NLPPipeline", FakeNLP)
    monkeypatch.setattr(
        benchmark_module,
        "compute_wer",
        lambda reference, hypothesis: 0.4 if hypothesis == "raw transcript" else 0.1,
    )
    monkeypatch.setattr(
        benchmark_module,
        "compute_cer",
        lambda reference, hypothesis: 0.3 if hypothesis == "raw transcript" else 0.05,
    )
    monkeypatch.setattr(benchmark_module, "semantic_preservation_score", lambda *_: 0.9)

    results = benchmark_module.run_benchmark([
        {"raw_asr": "raw transcript", "reference": "reference transcript"}
    ])

    sample = results["per_sample"][0]
    assert sample["raw_asr"] == "raw transcript"
    assert sample["flownlp"] == "processed transcript"
    assert results["levels"]["raw_asr"] == {"mean_wer": 0.4, "mean_cer": 0.3}
    assert results["levels"]["flownlp"] == {"mean_wer": 0.1, "mean_cer": 0.05}
    assert results["levels"]["processed_minus_raw"] == {
        "mean_wer": -0.3,
        "mean_cer": -0.25,
    }
    output = capsys.readouterr().out
    assert "Raw ASR" in output
    assert "FlowNLP" in output
    assert "negative is better" in output