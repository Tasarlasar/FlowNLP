"""
Record Tab
The main capture panel. Shows an animated waveform visualiser, a
record/stop button, mode selector, live status, and the most-recent
output text.
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, Slot, Signal
)
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)

from app.states import AppState

CONTEXT_MODES = ["Auto", "Chat", "Email", "Notes", "Search", "Code"]

# ── colour palette (matches overall app theme) ──────────────────────────────
C_BG        = "#1e1e2e"
C_SURFACE   = "#2a2a3e"
C_ACCENT    = "#7c6af7"      # purple
C_GREEN     = "#50fa7b"
C_RED       = "#ff5555"
C_ORANGE    = "#ffb86c"
C_MUTED     = "#6272a4"
C_TEXT      = "#cdd6f4"
C_TEXT_DIM  = "#888ba8"


# ── Waveform widget ──────────────────────────────────────────────────────────
class WaveformWidget(QWidget):
    """Animated bars that pulse when recording, idle when not."""

    BAR_COUNT = 28
    BAR_WIDTH = 4
    BAR_GAP   = 3

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._active  = False
        self._heights = [0.05] * self.BAR_COUNT
        self._targets = [0.05] * self.BAR_COUNT
        self._timer   = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self.setFixedHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_active(self, active: bool) -> None:
        self._active = active
        if not active:
            self._targets = [0.05] * self.BAR_COUNT

    def _tick(self) -> None:
        if self._active:
            for i in range(self.BAR_COUNT):
                if random.random() < 0.35:
                    self._targets[i] = random.uniform(0.15, 1.0)
        for i in range(self.BAR_COUNT):
            diff = self._targets[i] - self._heights[i]
            self._heights[i] += diff * 0.25
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        total = self.BAR_COUNT * (self.BAR_WIDTH + self.BAR_GAP) - self.BAR_GAP
        x_off = (w - total) / 2

        for i, ratio in enumerate(self._heights):
            bar_h = max(4, int(ratio * (h - 8)))
            x = int(x_off + i * (self.BAR_WIDTH + self.BAR_GAP))
            y = (h - bar_h) // 2

            if self._active:
                alpha = int(180 + 75 * ratio)
                color = QColor(124, 106, 247, alpha)   # accent purple
            else:
                color = QColor(98, 114, 164, 90)       # muted

            p.setBrush(QBrush(color))
            p.setPen(Qt.PenStyle.NoPen)
            radius = self.BAR_WIDTH // 2
            p.drawRoundedRect(x, y, self.BAR_WIDTH, bar_h, radius, radius)
        p.end()


# ── Record Tab ───────────────────────────────────────────────────────────────
class RecordTab(QWidget):
    """
    Signals
    -------
    mode_changed(str)  — emitted when the user selects a different mode
    """

    mode_changed = Signal(str)

    def __init__(self, controller, db, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._db = db
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick_timer)
        self._elapsed = 0
        self._setup_ui()
        self._connect_signals()

    # ── UI construction ──────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(0)

        # Section label
        section = QLabel("CAPTURE")
        section.setStyleSheet(
            f"color: {C_MUTED}; font-size: 10px; letter-spacing: 2px;"
            " font-family: 'Courier New', monospace;"
        )
        root.addWidget(section)
        root.addSpacing(16)

        # Waveform
        self._waveform = WaveformWidget()
        root.addWidget(self._waveform)
        root.addSpacing(24)

        # Record button row
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._record_btn = QPushButton("⏺  Start Recording")
        self._record_btn.setFixedSize(220, 52)
        self._record_btn.setStyleSheet(self._record_btn_style(False))
        self._record_btn.clicked.connect(self._toggle_recording)
        btn_row.addWidget(self._record_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)
        root.addSpacing(12)

        # Timer label
        self._timer_label = QLabel("00:00")
        self._timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._timer_label.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 13px;"
            " font-family: 'Courier New', monospace;"
        )
        root.addWidget(self._timer_label)
        root.addSpacing(28)

        # ── Divider ──────────────────────────────────────────────────────────
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"color: {C_SURFACE};")
        root.addWidget(div)
        root.addSpacing(20)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Output mode")
        mode_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(CONTEXT_MODES)
        self._mode_combo.setStyleSheet(self._combo_style())
        self._mode_combo.currentTextChanged.connect(self.mode_changed.emit)
        mode_row.addWidget(mode_lbl)
        mode_row.addStretch()
        mode_row.addWidget(self._mode_combo)
        root.addLayout(mode_row)
        root.addSpacing(20)

        # Status strip
        self._status_label = QLabel("● Ready — press Ctrl+Shift+Space to dictate")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(
            f"color: {C_MUTED}; font-size: 12px; padding: 8px 0;"
        )
        root.addWidget(self._status_label)
        root.addSpacing(16)

        # Output text area
        out_lbl = QLabel("LATEST OUTPUT")
        out_lbl.setStyleSheet(
            f"color: {C_MUTED}; font-size: 10px; letter-spacing: 2px;"
            " font-family: 'Courier New', monospace;"
        )
        root.addWidget(out_lbl)
        root.addSpacing(6)

        self._output_area = QTextEdit()
        self._output_area.setReadOnly(True)
        self._output_area.setPlaceholderText(
            "Your formatted dictation will appear here after recording…\n\n"
            "(Models load in the background — first dictation may take ~30 s)"
        )
        self._output_area.setStyleSheet(self._text_area_style())
        self._output_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self._output_area)
        root.addSpacing(12)

        # Copy button
        copy_row = QHBoxLayout()
        copy_row.addStretch()
        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setFixedWidth(90)
        self._copy_btn.setStyleSheet(self._secondary_btn_style())
        self._copy_btn.clicked.connect(self._copy_output)
        copy_row.addWidget(self._copy_btn)
        root.addLayout(copy_row)

        # Restore saved mode
        saved_mode = self._db.get_setting("output_mode", "Auto")
        idx = self._mode_combo.findText(saved_mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)

    # ── Signal connections ───────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        c = self._controller
        c.state_changed.connect(self._on_state_changed)
        c.transcript_ready.connect(self._on_transcript)
        c.result_ready.connect(self._on_result)
        c.error_occurred.connect(self._on_error)

    # ── Slots ────────────────────────────────────────────────────────────────

    @Slot(object)
    def _on_state_changed(self, state: AppState) -> None:
        recording = (state == AppState.RECORDING)
        processing = (state == AppState.PROCESSING)

        self._waveform.set_active(recording)
        self._record_btn.setEnabled(not processing)

        if recording:
            self._record_btn.setText("⏹  Stop Recording")
            self._record_btn.setStyleSheet(self._record_btn_style(True))
            self._elapsed = 0
            self._timer_label.setText("00:00")
            self._timer.start()
            self._status_label.setText("● Listening…")
            self._status_label.setStyleSheet(
                f"color: {C_RED}; font-size: 12px; padding: 8px 0;"
            )
            self._output_area.clear()
        elif processing:
            self._record_btn.setText("⟳ Processing…")
            self._timer.stop()
            self._status_label.setText("⟳ Running pipeline…")
            self._status_label.setStyleSheet(
                f"color: {C_ORANGE}; font-size: 12px; padding: 8px 0;"
            )
        else:
            self._record_btn.setText("⏺  Start Recording")
            self._record_btn.setStyleSheet(self._record_btn_style(False))
            self._timer.stop()
            if state == AppState.COMPLETED:
                self._status_label.setText("✓ Done — saved to Library")
                self._status_label.setStyleSheet(
                    f"color: {C_GREEN}; font-size: 12px; padding: 8px 0;"
                )
            elif state == AppState.IDLE:
                self._status_label.setText(
                    "● Ready — press Ctrl+Shift+Space to dictate"
                )
                self._status_label.setStyleSheet(
                    f"color: {C_MUTED}; font-size: 12px; padding: 8px 0;"
                )

    @Slot(str)
    def _on_transcript(self, text: str) -> None:
        preview = text[:80] + ("…" if len(text) > 80 else "")
        self._status_label.setText(f"Heard: {preview}")

    @Slot(str, str)
    def _on_result(self, _record_id: str, text: str) -> None:
        self._output_area.setPlainText(text)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._status_label.setText(f"✗ {message}")
        self._status_label.setStyleSheet(
            f"color: {C_RED}; font-size: 12px; padding: 8px 0;"
        )
        self._output_area.setPlainText(f"Error: {message}")

    def _toggle_recording(self) -> None:
        if self._controller.state == AppState.RECORDING:
            self._controller.stop_recording()
        elif self._controller.state == AppState.IDLE:
            # Sync mode before starting
            mode = self._mode_combo.currentText()
            self._db.set_setting("output_mode", mode)
            self._controller.start_recording()

    def _tick_timer(self) -> None:
        self._elapsed += 1
        m, s = divmod(self._elapsed, 60)
        self._timer_label.setText(f"{m:02d}:{s:02d}")

    def _copy_output(self) -> None:
        text = self._output_area.toPlainText()
        if text:
            from output.clipboard import ClipboardManager
            ClipboardManager.copy(text)
            self._copy_btn.setText("Copied!")
            QTimer.singleShot(1500, lambda: self._copy_btn.setText("Copy"))

    # ── Style helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _record_btn_style(active: bool) -> str:
        bg = C_RED if active else C_ACCENT
        return (
            f"QPushButton {{"
            f"  background-color: {bg};"
            f"  color: white;"
            f"  border: none;"
            f"  border-radius: 26px;"
            f"  font-size: 14px;"
            f"  font-weight: 600;"
            f"}}"
            f"QPushButton:hover {{ background-color: {'#ff6e6e' if active else '#9580ff'}; }}"
            f"QPushButton:disabled {{ background-color: {C_MUTED}; }}"
        )

    @staticmethod
    def _secondary_btn_style() -> str:
        return (
            f"QPushButton {{"
            f"  background-color: {C_SURFACE};"
            f"  color: {C_TEXT};"
            f"  border: 1px solid #44475a;"
            f"  border-radius: 6px;"
            f"  padding: 6px 14px;"
            f"  font-size: 12px;"
            f"}}"
            f"QPushButton:hover {{ background-color: #313244; }}"
        )

    @staticmethod
    def _combo_style() -> str:
        return (
            f"QComboBox {{"
            f"  background-color: {C_SURFACE};"
            f"  color: {C_TEXT};"
            f"  border: 1px solid #44475a;"
            f"  border-radius: 6px;"
            f"  padding: 5px 10px;"
            f"  min-width: 110px;"
            f"}}"
            f"QComboBox::drop-down {{ border: none; }}"
            f"QComboBox QAbstractItemView {{"
            f"  background: {C_SURFACE}; color: {C_TEXT};"
            f"  selection-background-color: {C_ACCENT};"
            f"}}"
        )

    @staticmethod
    def _text_area_style() -> str:
        return (
            f"QTextEdit {{"
            f"  background-color: {C_SURFACE};"
            f"  color: {C_TEXT};"
            f"  border: 1px solid #44475a;"
            f"  border-radius: 8px;"
            f"  padding: 12px;"
            f"  font-size: 13px;"
            f"  line-height: 1.5;"
            f"}}"
        )
