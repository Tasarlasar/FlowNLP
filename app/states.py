"""
Application state machine for FlowNLP.
Defines every possible state the application can be in at runtime.
"""

from enum import Enum, auto


class AppState(Enum):
    IDLE = auto()        # Waiting for hotkey press
    RECORDING = auto()   # Microphone active, audio being captured
    PROCESSING = auto()  # ASR + NLP pipeline running
    COMPLETED = auto()   # Text injected / ready to copy
    ERROR = auto()       # An unrecoverable error occurred
