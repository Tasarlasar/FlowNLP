"""
Abstract base class for ASR engines.
New engines (e.g. a cloud-based provider) only need to implement
the `transcribe` method to slot into the pipeline.
"""

from abc import ABC, abstractmethod

import numpy as np


class ASREngine(ABC):
    """Interface every ASR back-end must satisfy."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> dict:
        """
        Convert audio to text.

        Parameters
        ----------
        audio : np.ndarray
            Float32 mono PCM at 16 kHz.

        Returns
        -------
        dict with at minimum:
            {
                "text": str,          # full transcript
                "language": str,      # detected language code
                "segments": list,     # optional word/segment timestamps
            }
        """
