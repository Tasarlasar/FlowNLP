"""
Main Window
PySide6 desktop UI for FlowNLP.

States displayed:
  IDLE       — waiting for hotkey
  RECORDING  — pulsing indicator, live transcript preview
  PROCESSING — spinner / progress indicator
  COMPLETED  — shows formatted output, copy button
  ERROR      — shows error message, retry option
"""

import logging

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.states import AppState

logger = logging.getLogger(__name__)

# Context modes offered in the mode selector
CONTEXT_MODES = ["Auto", "Chat", "Email", "Notes", "Search", "Code"]


class MainWindow(QMainWindow):
    """
    Primary application window.

    Parameters
    ----------
    controller : Controller
        The central controller that drives all subsystems.
    """

    def __init__(self, controller) -> None:
        super().__init__()
        self._controller = controller
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowTitle("FlowNLP")
        self.setMinimumWidth(400)
        self.setMinimumHeight(320)

        # ---- Central widget ----
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # ---- Title ----
        title = QLabel("FlowNLP")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # ---- Status indicator ----
        self._status_label = QLabel("● Ready")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(self._status_label)

        # ---- Transcript / output area ----
        self._output_area = QTextEdit()
        self._output_area.setReadOnly(True)
        self._output_area.setPlaceholderText(
            "Loading models… (this takes ~30s on first launch, then it's instant)"
        )
        self._output_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._output_area)

        # ---- Mode selector row ----
        mode_row = QHBoxLayout()
        mode_label = QLabel("Mode:")
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(CONTEXT_MODES)
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # ---- Button row ----
        btn_row = QHBoxLayout()
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._copy_btn = QPushButton("Copy")
        self._settings_btn = QPushButton("Settings")
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._copy_btn)
        btn_row.addWidget(self._settings_btn)
        layout.addLayout(btn_row)

        # ---- Hotkey hint ----
        hint = QLabel("Global shortcut: Ctrl+Shift+Space")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(hint)

    def _connect_signals(self) -> None:
        # Controller → UI
        self._controller.state_changed.connect(self._on_state_changed)
        self._controller.transcript_ready.connect(self._on_transcript_ready)
        self._controller.result_ready.connect(self._on_result_ready)
        self._controller.error_occurred.connect(self._on_error)

        # Buttons
        self._stop_btn.clicked.connect(self._controller.stop_recording)
        self._copy_btn.clicked.connect(self._copy_output)
        self._settings_btn.clicked.connect(self._open_settings)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_state_changed(self, state: AppState) -> None:
        styles = {
            AppState.IDLE:       ("● Ready — press Ctrl+Shift+Space to dictate", "#888"),
            AppState.RECORDING:  ("● Listening…",   "#e74c3c"),
            AppState.PROCESSING: ("⟳ Finishing…",   "#f39c12"),
            AppState.COMPLETED:  ("✓ Done",         "#2ecc71"),
            AppState.ERROR:      ("✗ Error",        "#e74c3c"),
        }
        text, color = styles.get(state, ("● Ready", "#888"))
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color}; font-size: 14px;")
        self._stop_btn.setEnabled(state == AppState.RECORDING)

        # Clear the output area at the start of each new session
        if state == AppState.RECORDING:
            self._output_area.clear()

        logger.debug("UI state → %s", state.name)

    @Slot(str)
    def _on_transcript_ready(self, text: str) -> None:
        # Show raw chunk in status label only — don't overwrite the clean output
        self._status_label.setText(f"● Heard: {text[:60]}{'…' if len(text) > 60 else ''}")

    @Slot(str)
    def _on_result_ready(self, text: str) -> None:
        # Append each processed chunk to the output area
        current = self._output_area.toPlainText()
        separator = " " if current and not current.endswith("\n") else ""
        self._output_area.setPlainText(current + separator + text)
        # Scroll to bottom
        sb = self._output_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._output_area.setPlainText(f"Error: {message}")

    def _copy_output(self) -> None:
        from output.clipboard import ClipboardManager
        ClipboardManager.copy(self._output_area.toPlainText())

    def _open_settings(self) -> None:
        from ui.settings import SettingsDialog
        dlg = SettingsDialog(controller=self._controller, parent=self)
        dlg.exec()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._controller.shutdown()
        super().closeEvent(event)
