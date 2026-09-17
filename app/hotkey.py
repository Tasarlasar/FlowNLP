"""
Global Hotkey Manager
Registers a system-wide keyboard shortcut that starts / stops dictation
regardless of which application is currently in focus.

Default shortcut: Ctrl+Shift+Space
"""

import logging
import threading
from typing import Callable, Optional

from pynput import keyboard

logger = logging.getLogger(__name__)

DEFAULT_HOTKEY = "<ctrl>+<shift>+<space>"


class HotkeyManager:
    """
    Listens for a configurable global hotkey and toggles recording.

    Usage
    -----
    manager = HotkeyManager(on_start=start_fn, on_stop=stop_fn)
    manager.start()
    ...
    manager.stop()
    """

    def __init__(
        self,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        hotkey: str = DEFAULT_HOTKEY,
    ) -> None:
        self._on_start = on_start
        self._on_stop = on_stop
        self._hotkey_str = hotkey
        self._listener: Optional[keyboard.GlobalHotKeys] = None
        self._recording = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin listening for the global hotkey."""
        self._listener = keyboard.GlobalHotKeys(
            {self._hotkey_str: self._toggle}
        )
        self._listener.daemon = True
        self._listener.start()
        logger.info("HotkeyManager listening on %s", self._hotkey_str)

    def stop(self) -> None:
        """Stop the hotkey listener."""
        if self._listener:
            self._listener.stop()
            self._listener = None
        logger.info("HotkeyManager stopped.")

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _toggle(self) -> None:
        with self._lock:
            if self._recording:
                self._recording = False
                logger.debug("Hotkey: stopping recording.")
                self._on_stop()
            else:
                self._recording = True
                logger.debug("Hotkey: starting recording.")
                self._on_start()
