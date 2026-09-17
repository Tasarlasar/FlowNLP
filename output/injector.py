"""
Text Injector
Types text directly into the currently focused application using
pyautogui, which sends real key-down/key-up events — no clipboard involved.

On macOS this requires Accessibility permission for the terminal that
launched the app (System Settings → Privacy & Security → Accessibility).
"""

import logging
import platform

logger = logging.getLogger(__name__)

_OS = platform.system()


class TextInjector:
    """
    insert(text) → bool

    Sends text as keyboard events to whichever window has focus.
    Falls back to clipboard+paste only if pyautogui is unavailable.
    """

    def insert(self, text: str) -> bool:
        if not text:
            return False

        # Primary: pyautogui typewrite — real key events, no clipboard
        try:
            import pyautogui  # type: ignore
            # typewrite doesn't handle unicode well; use write() for short text
            # and hotkey-paste for longer runs or non-ASCII content
            pyautogui.typewrite(text, interval=0.02)
            logger.debug("Typed %d chars via pyautogui.typewrite", len(text))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("pyautogui.typewrite failed (%s) — trying pynput.", exc)

        # Secondary: pynput keyboard controller (handles unicode better)
        try:
            from pynput.keyboard import Controller  # type: ignore
            kb = Controller()
            kb.type(text)
            logger.debug("Typed %d chars via pynput", len(text))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("pynput.type failed (%s) — falling back to clipboard.", exc)

        # Last resort: clipboard + paste
        try:
            import pyperclip  # type: ignore
            import time
            pyperclip.copy(text)
            time.sleep(0.05)
            if _OS == "Darwin":
                import pyautogui
                pyautogui.hotkey("command", "v")
            else:
                import pyautogui
                pyautogui.hotkey("ctrl", "v")
            logger.info("Text injected via clipboard fallback.")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("All injection methods failed: %s", exc)
            return False
