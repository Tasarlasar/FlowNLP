"""
Main Window — FlowNLP App
A full-featured, dark-themed desktop application window with:
  • Sidebar navigation (Record / Library / Insights)
  • Settings button in sidebar footer
  • Global hotkey hint
  • Status bar showing model/version info
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QSize, Slot
from PySide6.QtGui import QFont, QIcon, QColor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QStackedWidget,
    QVBoxLayout, QWidget,
)

from app.states import AppState
from ui.record_tab   import RecordTab
from ui.library_tab  import LibraryTab
from ui.insights_tab import InsightsTab

logger = logging.getLogger(__name__)

# ── palette ──────────────────────────────────────────────────────────────────
C_BG        = "#1e1e2e"
C_SIDEBAR   = "#181825"
C_SURFACE   = "#2a2a3e"
C_ACCENT    = "#7c6af7"
C_GREEN     = "#50fa7b"
C_RED       = "#ff5555"
C_ORANGE    = "#ffb86c"
C_MUTED     = "#6272a4"
C_TEXT      = "#cdd6f4"
C_TEXT_DIM  = "#888ba8"


class _NavButton(QPushButton):
    """Left sidebar navigation button."""

    def __init__(self, icon_char: str, label: str, parent=None) -> None:
        super().__init__(parent)
        self._icon_char = icon_char
        self._label = label
        self._active = False
        self.setCheckable(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(48)
        self._apply_style(False)

    def set_active(self, active: bool) -> None:
        self._active = active
        self._apply_style(active)

    def _apply_style(self, active: bool) -> None:
        bg     = C_SURFACE if active else "transparent"
        color  = C_TEXT    if active else C_TEXT_DIM
        border = f"border-left: 3px solid {C_ACCENT};" if active else "border-left: 3px solid transparent;"
        self.setText(f"  {self._icon_char}  {self._label}")
        self.setStyleSheet(
            f"QPushButton {{"
            f"  background: {bg}; color: {color};"
            f"  {border}"
            f"  border-right: none; border-top: none; border-bottom: none;"
            f"  text-align: left; padding: 0 16px;"
            f"  font-size: 13px;"
            f"}}"
            f"QPushButton:hover {{ background: {C_SURFACE}; color: {C_TEXT}; }}"
        )


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self, controller, db, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._db         = db
        self._setup_ui()
        self._connect_signals()
        self._navigate(0)   # start on Record tab

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        self.setWindowTitle("FlowNLP")
        self.setMinimumSize(900, 620)
        self.resize(1100, 700)
        self.setStyleSheet(f"QMainWindow {{ background: {C_BG}; }}")

        # Central container
        central = QWidget()
        central.setStyleSheet(f"background: {C_BG};")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ───────────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet(f"background: {C_SIDEBAR};")
        sv = QVBoxLayout(sidebar)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(0)

        # Logo / brand
        brand = QLabel(" FlowNLP")
        brand.setFixedHeight(56)
        brand.setStyleSheet(
            f"color: {C_TEXT}; font-size: 17px; font-weight: 700;"
            f" background: {C_SIDEBAR}; padding-left: 18px;"
            f" border-bottom: 1px solid {C_SURFACE};"
        )
        sv.addWidget(brand)
        sv.addSpacing(8)

        # Nav buttons
        self._nav_btns: list[_NavButton] = []
        nav_items = [
            ("⏺", "Record"),
            ("📋", "Library"),
            ("📊", "Insights"),
        ]
        for i, (icon, label) in enumerate(nav_items):
            btn = _NavButton(icon, label)
            btn.clicked.connect(lambda checked, idx=i: self._navigate(idx))
            sv.addWidget(btn)
            self._nav_btns.append(btn)

        sv.addStretch()

        # Hotkey hint
        hotkey_lbl = QLabel("  Ctrl+Shift+Space")
        hotkey_lbl.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 10px; padding: 6px 0;"
        )
        sv.addWidget(hotkey_lbl)

        # Settings button
        self._settings_btn = QPushButton("  ⚙  Settings")
        self._settings_btn.setFixedHeight(44)
        self._settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._settings_btn.setStyleSheet(
            f"QPushButton {{"
            f"  background: transparent; color: {C_TEXT_DIM};"
            f"  border: none; text-align: left; padding: 0 16px; font-size: 12px;"
            f"}}"
            f"QPushButton:hover {{ background: {C_SURFACE}; color: {C_TEXT}; }}"
        )
        self._settings_btn.clicked.connect(self._open_settings)
        sv.addWidget(self._settings_btn)

        # Status indicator in sidebar footer
        self._state_indicator = QLabel("  ● Ready")
        self._state_indicator.setFixedHeight(36)
        self._state_indicator.setStyleSheet(
            f"color: {C_MUTED}; font-size: 11px;"
            f" border-top: 1px solid {C_SURFACE};"
        )
        sv.addWidget(self._state_indicator)

        root.addWidget(sidebar)

        # ── Vertical divider ──────────────────────────────────────────────────
        div = QFrame()
        div.setFrameShape(QFrame.Shape.VLine)
        div.setStyleSheet(f"color: {C_SURFACE};")
        div.setFixedWidth(1)
        root.addWidget(div)

        # ── Content stack ─────────────────────────────────────────────────────
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background: {C_BG};")

        self._record_tab   = RecordTab(self._controller, self._db)
        self._library_tab  = LibraryTab(self._controller, self._db)
        self._insights_tab = InsightsTab(self._db)

        self._stack.addWidget(self._record_tab)   # index 0
        self._stack.addWidget(self._library_tab)  # index 1
        self._stack.addWidget(self._insights_tab) # index 2

        root.addWidget(self._stack, 1)

    # ── Signal connections ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._controller.state_changed.connect(self._on_state_changed)
        self._controller.result_ready.connect(self._on_result_ready)
        self._controller.error_occurred.connect(self._on_error)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _navigate(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        for i, btn in enumerate(self._nav_btns):
            btn.set_active(i == index)
        # Refresh Insights when switching to it
        if index == 2:
            self._insights_tab.refresh()

    # ── Slots ─────────────────────────────────────────────────────────────────

    @Slot(object)
    def _on_state_changed(self, state: AppState) -> None:
        styles = {
            AppState.IDLE:       (f"  ● Ready",          C_MUTED),
            AppState.RECORDING:  (f"  ● Listening…",     C_RED),
            AppState.PROCESSING: (f"  ⟳ Processing…",   C_ORANGE),
            AppState.COMPLETED:  (f"  ✓ Done",           C_GREEN),
            AppState.ERROR:      (f"  ✗ Error",          C_RED),
        }
        text, color = styles.get(state, (f"  ● Ready", C_MUTED))
        self._state_indicator.setText(text)
        self._state_indicator.setStyleSheet(
            f"color: {color}; font-size: 11px;"
            f" border-top: 1px solid {C_SURFACE};"
        )

    @Slot(str, str)
    def _on_result_ready(self, _record_id: str, _text: str) -> None:
        # Flash Library nav button briefly to hint that a new item is ready
        lib_btn = self._nav_btns[1]
        lib_btn._label = "Library  ●"
        lib_btn._apply_style(self._stack.currentIndex() == 1)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(4000, self._reset_library_label)

    def _reset_library_label(self) -> None:
        lib_btn = self._nav_btns[1]
        lib_btn._label = "Library"
        lib_btn._apply_style(self._stack.currentIndex() == 1)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        logger.warning("UI received error: %s", message)

    # ── Settings ──────────────────────────────────────────────────────────────

    def _open_settings(self) -> None:
        from ui.settings_dialog import SettingsDialog
        dlg = SettingsDialog(
            controller=self._controller, db=self._db, parent=self
        )
        dlg.exec()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: N802
        self._controller.shutdown()
        super().closeEvent(event)
