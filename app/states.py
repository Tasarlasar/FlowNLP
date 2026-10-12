"""
Application state machine for FlowNLP App.
Defines every possible state the application can be in at runtime.
"""

from enum import Enum, auto


class AppState(Enum):
    IDLE       = auto()   # Waiting for hotkey or button press
    RECORDING  = auto()   # Microphone active, audio being captured
    PROCESSING = auto()   # ASR + NLP pipeline running
    COMPLETED  = auto()   # Text ready to review / copy / inject
    ERROR      = auto()   # An error occurred during recording or processing
