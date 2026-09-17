"""
Clipboard Manager
Copies text to the system clipboard and optionally pastes it into
the currently focused application.
"""

import logging
import time

import pyperclip  # type: ignore

logger = logging.getLogger(__name__)

PASTE_DELAY_S = 0.1   # Short pause before simulating paste


class ClipboardManager:
    """Provides clipboard read/write operations."""

    @staticmethod
    def copy(text: str) -> bool:
        """
        Write `text` to the system clipboard.

        Returns True on success, False on failure.
        """
        try:
            pyperclip.copy(text)
            logger.debug("Copied %d chars to clipboard.", len(text))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Clipboard copy failed: %s", exc)
            return False

    @staticmethod
    def paste_via_shortcut() -> bool:
        """
        Simulate Ctrl+V / Cmd+V to paste clipboard contents into the
        currently focused application.

        Returns True if the keystroke was sent without error.
        """
        try:
            import platform
            import pyautogui  # type: ignore

            time.sleep(PASTE_DELAY_S)
            if platform.system() == "Darwin":
                pyautogui.hotkey("command", "v")
            else:
                pyautogui.hotkey("ctrl", "v")
            logger.debug("Paste shortcut sent.")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Paste shortcut failed: %s", exc)
            return False

    @staticmethod
    def get() -> str:
        """Return current clipboard text."""
        try:
            return pyperclip.paste()
        except Exception:  # noqa: BLE001
            return ""
