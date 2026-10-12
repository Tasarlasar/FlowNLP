"""
Active Application Detector
Returns the name of the currently focused desktop application.

Implementation is OS-aware:
- macOS: uses AppKit / subprocess
- Windows: uses win32gui (pywin32)
- Linux: uses xdotool via subprocess
"""

import logging
import platform
import subprocess

logger = logging.getLogger(__name__)

_OS = platform.system()


def get_active_application() -> str:
    """
    Return the name of the currently focused application.
    Returns an empty string on failure.
    """
    try:
        if _OS == "Darwin":
            return _get_active_macos()
        if _OS == "Windows":
            return _get_active_windows()
        if _OS == "Linux":
            return _get_active_linux()
    except Exception as exc:  # noqa: BLE001
        logger.debug("get_active_application failed: %s", exc)
    return ""


# ------------------------------------------------------------------
# Platform implementations
# ------------------------------------------------------------------

def _get_active_macos() -> str:
    script = (
        'tell application "System Events" to get name of '
        'first application process whose frontmost is true'
    )
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=2,
    )
    return result.stdout.strip()


def _get_active_windows() -> str:
    try:
        import win32gui  # type: ignore
        import win32process  # type: ignore
        import psutil  # type: ignore

        hwnd = win32gui.GetForegroundWindow()
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process = psutil.Process(pid)
        return process.name()
    except ImportError:
        return ""


def _get_active_linux() -> str:
    result = subprocess.run(
        ["xdotool", "getactivewindow", "getwindowname"],
        capture_output=True,
        text=True,
        timeout=2,
    )
    return result.stdout.strip()
