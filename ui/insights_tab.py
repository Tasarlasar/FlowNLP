"""
Insights Tab
Displays aggregated evaluation metrics from all confirmed recordings:
  • confirmed count
  • rewrite acceptance rate
  • average edit ratio
  • WER proxy
  • breakdowns by model / style / tone
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

C_BG       = "#1e1e2e"
C_SURFACE  = "#2a2a3e"
C_SURFACE2 = "#313244"
C_ACCENT   = "#7c6af7"
C_GREEN    = "#50fa7b"
C_ORANGE   = "#ffb86c"
C_MUTED    = "#6272a4"
C_TEXT     = "#cdd6f4"
C_TEXT_DIM = "#888ba8"
C_RED      = "#ff5555"


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _fmt(value: float | None, dp: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{dp}f}"


class _MetricCard(QWidget):
    def __init__(
        self, label: str, value: str, sub: str = "", color: str = C_TEXT
    ) -> None:
        super().__init__()
        self.setStyleSheet(
            f"background: {C_SURFACE}; border-radius: 10px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(4)

        lbl = QLabel(label.upper())
        lbl.setStyleSheet(
            f"color: {C_MUTED}; font-size: 10px; letter-spacing: 2px;"
            " font-family: 'Courier New', monospace;"
        )
        layout.addWidget(lbl)

        val = QLabel(value)
        val.setStyleSheet(
            f"color: {color}; font-size: 28px; font-weight: 700;"
        )
        layout.addWidget(val)

        if sub:
            sub_lbl = QLabel(sub)
            sub_lbl.setWordWrap(True)
            sub_lbl.setStyleSheet(
                f"color: {C_TEXT_DIM}; font-size: 11px;"
            )
            layout.addWidget(sub_lbl)

    def update_value(self, value: str, color: str = C_TEXT) -> None:
        # Re-create is simpler than tracking individual label refs
        pass  # handled by full rebuild in InsightsTab.refresh()


class _BreakdownRow(QWidget):
    def __init__(
        self, label: str, data: dict[str, int], total: int
    ) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        hdr = QLabel(label.upper())
        hdr.setStyleSheet(
            f"color: {C_MUTED}; font-size: 10px; letter-spacing: 2px;"
            " font-family: 'Courier New', monospace;"
        )
        layout.addWidget(hdr)

        if not data:
            empty = QLabel("No data yet")
            empty.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
            layout.addWidget(empty)
            return

        for key, count in sorted(data.items(), key=lambda kv: -kv[1]):
            row = QHBoxLayout()
            key_lbl = QLabel(key)
            key_lbl.setMinimumWidth(100)
            key_lbl.setStyleSheet(f"color: {C_TEXT}; font-size: 12px;")
            row.addWidget(key_lbl)

            # Bar
            pct = count / max(total, 1)
            bar_outer = QWidget()
            bar_outer.setFixedHeight(8)
            bar_outer.setStyleSheet(
                f"background: {C_SURFACE2}; border-radius: 4px;"
            )
            bar_outer.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            bar_inner = QWidget(bar_outer)
            bar_inner.setFixedHeight(8)
            bar_inner.setStyleSheet(
                f"background: {C_ACCENT}; border-radius: 4px;"
            )
            bar_inner.setFixedWidth(
                max(4, int(pct * 220))   # approximate; resizes fine visually
            )
            row.addWidget(bar_outer)

            cnt_lbl = QLabel(f"{count}  ({pct*100:.0f}%)")
            cnt_lbl.setFixedWidth(80)
            cnt_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            cnt_lbl.setStyleSheet(
                f"color: {C_TEXT_DIM}; font-size: 11px;"
            )
            row.addWidget(cnt_lbl)
            layout.addLayout(row)


class InsightsTab(QWidget):
    """Evaluation / analytics tab."""

    def __init__(self, db, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db
        self._setup_ui()
        self.refresh()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {C_BG}; }}"
            f"QScrollBar:vertical {{ background: {C_SURFACE}; width: 6px; }}"
            f"QScrollBar::handle:vertical {{ background: #44475a; border-radius: 3px; }}"
        )

        inner = QWidget()
        inner.setStyleSheet(f"background: {C_BG};")
        self._inner_layout = QVBoxLayout(inner)
        self._inner_layout.setContentsMargins(32, 28, 32, 28)
        self._inner_layout.setSpacing(0)
        self._build_content()

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def _build_content(self) -> None:
        layout = self._inner_layout

        # Header row
        hdr_row = QHBoxLayout()
        section = QLabel("INSIGHTS")
        section.setStyleSheet(
            f"color: {C_MUTED}; font-size: 10px; letter-spacing: 2px;"
            " font-family: 'Courier New', monospace;"
        )
        hdr_row.addWidget(section)
        hdr_row.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet(
            f"QPushButton {{ background: {C_SURFACE}; color: {C_TEXT};"
            f" border: 1px solid #44475a; border-radius: 6px;"
            f" padding: 5px 12px; font-size: 11px; }}"
            f"QPushButton:hover {{ background: {C_SURFACE2}; }}"
        )
        refresh_btn.clicked.connect(self.refresh)
        hdr_row.addWidget(refresh_btn)
        layout.addLayout(hdr_row)
        layout.addSpacing(20)

        # Metric cards — 2×2 grid
        self._card_grid = QGridLayout()
        self._card_grid.setSpacing(14)
        layout.addLayout(self._card_grid)
        layout.addSpacing(28)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"color: {C_SURFACE};")
        layout.addWidget(div)
        layout.addSpacing(24)

        # Breakdown section label
        bk_lbl = QLabel("BREAKDOWN")
        bk_lbl.setStyleSheet(
            f"color: {C_MUTED}; font-size: 10px; letter-spacing: 2px;"
            " font-family: 'Courier New', monospace;"
        )
        layout.addWidget(bk_lbl)
        layout.addSpacing(16)

        self._breakdown_layout = QVBoxLayout()
        self._breakdown_layout.setSpacing(22)
        layout.addLayout(self._breakdown_layout)

        # WER note
        layout.addSpacing(20)
        note = QLabel(
            "⚠ WER proxy compares raw ASR output with your confirmed transcript. "
            "It includes your style edits. Use labeled reference audio for true WER."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 11px;"
            f" background: {C_SURFACE}; border-radius: 6px; padding: 10px;"
        )
        layout.addWidget(note)
        layout.addStretch()

    # ── Refresh ───────────────────────────────────────────────────────────────

    @Slot()
    def refresh(self) -> None:
        data = self._db.get_evaluation()

        # Rebuild metric cards
        while self._card_grid.count():
            item = self._card_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        confirmed = data.get("confirmed_count", 0)
        acceptance = data.get("rewrite_acceptance")
        edit_ratio = data.get("average_edit_ratio")
        wer_proxy  = data.get("average_wer_proxy")

        cards = [
            (
                "Confirmed", str(confirmed),
                "total recordings saved", C_GREEN if confirmed else C_TEXT
            ),
            (
                "Rewrite Acceptance", _pct(acceptance),
                "natural rewrites accepted", C_ACCENT
            ),
            (
                "Avg Edit Ratio", _fmt(edit_ratio),
                "edits vs. faithful cleanup", C_ORANGE
            ),
            (
                "Avg WER Proxy", _fmt(wer_proxy),
                "raw ASR → confirmed text", C_TEXT
            ),
        ]
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for (row, col), (label, value, sub, color) in zip(positions, cards):
            card = _MetricCard(label, value, sub, color)
            self._card_grid.addWidget(card, row, col)

        # Rebuild breakdowns
        while self._breakdown_layout.count():
            item = self._breakdown_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        total = confirmed
        breakdowns = [
            ("By Whisper Model", data.get("by_model", {})),
            ("By Output Style",  data.get("by_style", {})),
            ("By Tone",          data.get("by_tone", {})),
        ]
        for label, bdata in breakdowns:
            row = _BreakdownRow(label, bdata, total)
            self._breakdown_layout.addWidget(row)
