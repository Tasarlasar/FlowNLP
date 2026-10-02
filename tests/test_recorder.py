"""Tests for complete-session audio capture."""

import numpy as np

import audio.recorder as recorder_module
from audio.audio_utils import SAMPLE_RATE
from audio.recorder import SessionRecorder


class FakeInputStream:
    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


def test_stop_returns_full_session_in_order(monkeypatch):
    monkeypatch.setattr(recorder_module.sd, "InputStream", lambda **kwargs: FakeInputStream())
    recorder = SessionRecorder(use_vad=False)
    recorder.start()

    first = np.arange(SAMPLE_RATE * 2, dtype=np.float32).reshape(-1, 1)
    second = np.arange(SAMPLE_RATE * 2, SAMPLE_RATE * 4, dtype=np.float32).reshape(-1, 1)
    recorder._audio_callback(first, len(first), None, None)
    recorder._audio_callback(second, len(second), None, None)

    captured = recorder.stop()

    assert captured is not None
    np.testing.assert_array_equal(captured, np.concatenate((first, second)).flatten())