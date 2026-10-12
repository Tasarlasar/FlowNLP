"""
Library Tab
Shows the full history of recordings with a detail/review panel on the
right. The detail panel mirrors the web UI's Listen+Review section:
  • raw ASR transcript
  • faithful cleanup (editable)
  • natural rewrite panel with tone picker
  • uncertain-word list
  • confirm / copy / delete controls
"""

from __future__ import annotations

import json
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, Slot, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

# ── palette (shared with other tabs) ─────────────────────────────────────────
C_BG       = "#1e1e2e"
C_SURFACE  = "#2a2a3e"
C_SURFACE2 = "#313244"
C_ACCENT   = "#7c6af7"
C_GREEN    = "#50fa7b"
C_RED      = "#ff5555"
C_ORANGE   = "#ffb86c"
C_MUTED    = "#6272a4"
C_TEXT     = "#cdd6f4"
C_TEXT_DIM = "#888ba8"
C_YELLOW   = "#f1fa8c"

TONES = ["Professional", "Friendly", "Concise"]


def _fmt_date(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%d %b %Y  %H:%M")
    except Exception:
        return iso


class LibraryTab(QWidget):
    """Recording library with an inline review panel."""

    def __init__(self, controller, db, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller  = controller
        self._db          = db
        self._current_id: str | None = None
        self._rewrite_trial_id: int | None = None
        self._setup_ui()
        self._connect_signals()
        self.refresh_list()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(
            "QSplitter::handle { background: #44475a; }"
        )

        # ── Left: recording list ─────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(220)
        left.setMaximumWidth(300)
        left.setStyleSheet(f"background: {C_SURFACE};")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)

        list_header = QLabel("  RECORDINGS")
        list_header.setStyleSheet(
            f"background: {C_BG}; color: {C_MUTED}; font-size: 10px;"
            " letter-spacing: 2px; padding: 10px 8px;"
            " font-family: 'Courier New', monospace;"
        )
        lv.addWidget(list_header)

        self._list = QListWidget()
        self._list.setStyleSheet(self._list_style())
        self._list.currentItemChanged.connect(self._on_item_selected)
        lv.addWidget(self._list)

        splitter.addWidget(left)

        # ── Right: detail panel ──────────────────────────────────────────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {C_BG}; }}"
            f"QScrollBar:vertical {{ background: {C_SURFACE}; width: 6px; }}"
            f"QScrollBar::handle:vertical {{ background: #44475a; border-radius: 3px; }}"
        )

        right = QWidget()
        right.setStyleSheet(f"background: {C_BG};")
        self._detail_layout = QVBoxLayout(right)
        self._detail_layout.setContentsMargins(28, 24, 28, 24)
        self._detail_layout.setSpacing(0)

        placeholder = self._empty_label_widget()
        self._detail_layout.addWidget(placeholder)
        self._detail_layout.addStretch()

        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setSizes([240, 700])

        root.addWidget(splitter)

    def _connect_signals(self) -> None:
        self._controller.recordings_changed.connect(self.refresh_list)
        self._controller.rewrite_ready.connect(self._on_rewrite_ready)

    # ── List management ───────────────────────────────────────────────────────

    @Slot()
    def refresh_list(self) -> None:
        current_id = self._current_id
        self._list.blockSignals(True)
        self._list.clear()
        recordings = self._db.list_recordings()
        for rec in recordings:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, rec["id"])
            mode  = rec.get("mode", "")
            date  = _fmt_date(rec.get("created_at", ""))
            badge = " ✓" if rec.get("confirmed") else ""
            item.setText(f"{date}\n{mode}{badge}")
            item.setToolTip(rec["id"])
            self._list.addItem(item)
            if rec["id"] == current_id:
                self._list.setCurrentItem(item)
        self._list.blockSignals(False)
        # Refresh detail if current is still valid
        if current_id:
            rec = self._db.get_recording(current_id)
            if rec:
                self._populate_detail(rec)

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_item_selected(
        self, current: QListWidgetItem, _prev: QListWidgetItem
    ) -> None:
        if current is None:
            return
        record_id = current.data(Qt.ItemDataRole.UserRole)
        self._current_id = record_id
        rec = self._db.get_recording(record_id)
        if rec:
            self._populate_detail(rec)

    # ── Detail panel ──────────────────────────────────────────────────────────

    def _clear_detail(self) -> None:
        while self._detail_layout.count():
            child = self._detail_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _populate_detail(self, rec: dict) -> None:
        self._clear_detail()
        self._rewrite_trial_id = rec.get("rewrite_trial_id")
        dl = self._detail_layout

        # ── Header ───────────────────────────────────────────────────────────
        hdr_row = QHBoxLayout()
        title = QLabel(
            f"{rec.get('mode','?')} · {_fmt_date(rec.get('created_at',''))}"
        )
        title.setStyleSheet(f"color: {C_TEXT}; font-size: 14px; font-weight: 600;")
        hdr_row.addWidget(title)
        hdr_row.addStretch()

        status_text = {
            "draft":       ("Draft",       C_ORANGE),
            "confirmed":   ("Confirmed ✓", C_GREEN),
            "error":       ("Error ✗",     C_RED),
            "processing":  ("Processing…", C_MUTED),
        }.get(rec.get("status", ""), ("?", C_MUTED))
        status_lbl = QLabel(status_text[0])
        status_lbl.setStyleSheet(
            f"color: {status_text[1]}; font-size: 11px; padding: 3px 8px;"
            f" border: 1px solid {status_text[1]}; border-radius: 4px;"
        )
        hdr_row.addWidget(status_lbl)
        dl.addLayout(hdr_row)
        dl.addSpacing(4)

        meta_lbl = QLabel(
            f"Model: {rec.get('asr_model','?')}  ·  Style: {rec.get('output_style','?')}"
            f"  ·  Tone: {rec.get('tone','?')}"
        )
        meta_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
        dl.addWidget(meta_lbl)
        dl.addSpacing(20)

        # ── Error banner ─────────────────────────────────────────────────────
        if rec.get("error"):
            err = QLabel(f"⚠  {rec['error']}")
            err.setWordWrap(True)
            err.setStyleSheet(
                f"color: {C_RED}; background: #3d1515; border-radius: 6px;"
                " padding: 10px; font-size: 12px;"
            )
            dl.addWidget(err)
            dl.addSpacing(12)

        # ── RAW TRANSCRIPT ───────────────────────────────────────────────────
        dl.addWidget(self._section_label("RAW TRANSCRIPT"))
        dl.addSpacing(6)
        raw_box = QTextEdit()
        raw_box.setReadOnly(True)
        raw_box.setPlainText(rec.get("raw_text", ""))
        raw_box.setFixedHeight(80)
        raw_box.setStyleSheet(self._box_style(muted=True))
        dl.addWidget(raw_box)
        dl.addSpacing(18)

        # ── FAITHFUL CLEANUP ─────────────────────────────────────────────────
        dl.addWidget(self._section_label("FAITHFUL CLEANUP"))
        dl.addSpacing(6)
        self._faithful_edit = QTextEdit()
        self._faithful_edit.setPlainText(rec.get("faithful_text", ""))
        self._faithful_edit.setFixedHeight(110)
        self._faithful_edit.setStyleSheet(self._box_style())
        dl.addWidget(self._faithful_edit)
        dl.addSpacing(18)

        # ── Uncertain words ───────────────────────────────────────────────────
        uncertain = self._db.uncertain_words(rec.get("segments_json", "[]"))
        if uncertain:
            dl.addWidget(self._section_label("UNCERTAIN WORDS"))
            dl.addSpacing(6)
            uw_row = QHBoxLayout()
            for uw in uncertain[:12]:
                chip = QLabel(f"  {uw['text']} ({int(uw['probability']*100)}%)  ")
                chip.setStyleSheet(
                    f"color: {C_ORANGE}; background: #3d2a0a;"
                    " border-radius: 4px; font-size: 11px; padding: 2px 4px;"
                )
                uw_row.addWidget(chip)
            uw_row.addStretch()
            dl.addLayout(uw_row)
            dl.addSpacing(18)

        # ── NATURAL REWRITE ───────────────────────────────────────────────────
        dl.addWidget(self._section_label("NATURAL REWRITE"))
        dl.addSpacing(6)

        tone_row = QHBoxLayout()
        tone_row.addWidget(QLabel("Tone:"))
        tone_row.itemAt(0).widget().setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 12px;"
        )
        self._tone_combo = QComboBox()
        self._tone_combo.addItems(TONES)
        saved_tone = rec.get("tone", "Professional")
        idx = self._tone_combo.findText(saved_tone)
        if idx >= 0:
            self._tone_combo.setCurrentIndex(idx)
        self._tone_combo.setStyleSheet(self._combo_style())
        tone_row.addWidget(self._tone_combo)
        tone_row.addStretch()
        self._rewrite_btn = QPushButton("Generate Rewrite")
        self._rewrite_btn.setStyleSheet(self._primary_btn_style())
        self._rewrite_btn.clicked.connect(self._request_rewrite)
        tone_row.addWidget(self._rewrite_btn)
        dl.addLayout(tone_row)
        dl.addSpacing(8)

        self._rewrite_edit = QTextEdit()
        self._rewrite_edit.setPlainText(rec.get("rewrite_text", ""))
        self._rewrite_edit.setFixedHeight(110)
        self._rewrite_edit.setStyleSheet(self._box_style())
        dl.addWidget(self._rewrite_edit)

        # Rewrite warnings
        warnings = rec.get("rewrite_warnings") or []
        if warnings:
            dl.addSpacing(6)
            for w in warnings:
                wl = QLabel(f"⚠  {w}")
                wl.setWordWrap(True)
                wl.setStyleSheet(
                    f"color: {C_YELLOW}; font-size: 11px; padding: 2px 0;"
                )
                dl.addWidget(wl)

        if rec.get("rewrite_similarity") is not None:
            sim_lbl = QLabel(
                f"Semantic similarity: {rec['rewrite_similarity']:.2f}"
            )
            sim_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
            dl.addWidget(sim_lbl)

        dl.addSpacing(20)

        # ── FINAL VERSION ────────────────────────────────────────────────────
        dl.addWidget(self._section_label("FINAL VERSION"))
        dl.addSpacing(6)
        final_lbl = QLabel(
            "Edit below if needed, then click Confirm to save."
        )
        final_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
        dl.addWidget(final_lbl)
        dl.addSpacing(6)

        self._final_edit = QTextEdit()
        self._final_edit.setPlainText(
            rec.get("clean_text") or rec.get("faithful_text", "")
        )
        self._final_edit.setFixedHeight(130)
        self._final_edit.setStyleSheet(self._box_style(highlight=True))
        dl.addWidget(self._final_edit)
        dl.addSpacing(16)

        # ── Metrics ───────────────────────────────────────────────────────────
        if rec.get("edit_ratio") is not None or rec.get("raw_to_confirmed_wer") is not None:
            metrics_row = QHBoxLayout()
            if rec.get("edit_ratio") is not None:
                m = QLabel(f"Edit ratio: {rec['edit_ratio']:.3f}")
                m.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
                metrics_row.addWidget(m)
            if rec.get("raw_to_confirmed_wer") is not None:
                m2 = QLabel(f"  WER proxy: {rec['raw_to_confirmed_wer']:.3f}")
                m2.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
                metrics_row.addWidget(m2)
            metrics_row.addStretch()
            dl.addLayout(metrics_row)
            dl.addSpacing(10)

        # ── Action buttons ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._confirm_btn = QPushButton("Confirm & Save")
        self._confirm_btn.setStyleSheet(self._confirm_btn_style())
        self._confirm_btn.setEnabled(rec.get("status") != "confirmed")
        self._confirm_btn.clicked.connect(self._confirm)
        btn_row.addWidget(self._confirm_btn)

        self._copy_final_btn = QPushButton("Copy")
        self._copy_final_btn.setStyleSheet(self._secondary_btn_style())
        self._copy_final_btn.clicked.connect(self._copy_final)
        btn_row.addWidget(self._copy_final_btn)

        btn_row.addStretch()

        self._reprocess_btn = QPushButton("Reprocess")
        self._reprocess_btn.setStyleSheet(self._secondary_btn_style())
        self._reprocess_btn.clicked.connect(self._reprocess)
        btn_row.addWidget(self._reprocess_btn)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setStyleSheet(self._delete_btn_style())
        self._delete_btn.clicked.connect(self._delete)
        btn_row.addWidget(self._delete_btn)

        dl.addLayout(btn_row)
        dl.addStretch()

    # ── Actions ───────────────────────────────────────────────────────────────

    def _request_rewrite(self) -> None:
        if not self._current_id:
            return
        tone = self._tone_combo.currentText()
        self._rewrite_btn.setText("Generating…")
        self._rewrite_btn.setEnabled(False)
        self._controller.request_rewrite(self._current_id, tone)

    @Slot(str, dict)
    def _on_rewrite_ready(self, record_id: str, rewrite: dict) -> None:
        self._rewrite_btn.setText("Generate Rewrite")
        self._rewrite_btn.setEnabled(True)
        if record_id != self._current_id:
            return
        self._rewrite_trial_id = rewrite.get("trial_id")
        self._rewrite_edit.setPlainText(rewrite.get("text", ""))
        self._final_edit.setPlainText(rewrite.get("text", ""))

    def _confirm(self) -> None:
        if not self._current_id:
            return
        rec = self._db.get_recording(self._current_id)
        if not rec:
            return
        final = self._final_edit.toPlainText().strip()
        rewrite_text = self._rewrite_edit.toPlainText().strip()
        output_style = "natural" if (
            rewrite_text and final == rewrite_text
        ) else "faithful"
        tone = self._tone_combo.currentText()
        self._controller.confirm_recording(
            record_id=self._current_id,
            clean_text=final,
            output_style=output_style,
            tone=tone,
            trial_id=self._rewrite_trial_id if output_style == "natural" else None,
        )
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.setText("Confirmed ✓")

    def _copy_final(self) -> None:
        text = self._final_edit.toPlainText()
        if text:
            from output.clipboard import ClipboardManager
            ClipboardManager.copy(text)
            self._copy_final_btn.setText("Copied!")
            QTimer.singleShot(
                1500, lambda: self._copy_final_btn.setText("Copy")
            )

    def _reprocess(self) -> None:
        if not self._current_id:
            return
        self._controller.reprocess_recording(self._current_id)

    def _delete(self) -> None:
        if not self._current_id:
            return
        reply = QMessageBox.question(
            self,
            "Delete recording",
            "Delete this recording and its audio file? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._controller.delete_recording(self._current_id)
            self._current_id = None
            self._clear_detail()
            self._detail_layout.addWidget(self._empty_label_widget())
            self._detail_layout.addStretch()

    def _empty_label_widget(self) -> QLabel:
        lbl = QLabel("Select a recording from the list to review it.")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px;")
        return lbl

    # ── Style helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {C_MUTED}; font-size: 10px; letter-spacing: 2px;"
            " font-family: 'Courier New', monospace;"
        )
        return lbl

    @staticmethod
    def _box_style(muted: bool = False, highlight: bool = False) -> str:
        border = C_ACCENT if highlight else "#44475a"
        bg = "#252535" if muted else C_SURFACE
        return (
            f"QTextEdit {{"
            f"  background: {bg};"
            f"  color: {C_TEXT};"
            f"  border: 1px solid {border};"
            f"  border-radius: 6px;"
            f"  padding: 10px;"
            f"  font-size: 13px;"
            f"}}"
        )

    @staticmethod
    def _combo_style() -> str:
        return (
            f"QComboBox {{"
            f"  background: {C_SURFACE}; color: {C_TEXT};"
            f"  border: 1px solid #44475a; border-radius: 6px;"
            f"  padding: 5px 10px; min-width: 130px;"
            f"}}"
            f"QComboBox::drop-down {{ border: none; }}"
            f"QComboBox QAbstractItemView {{"
            f"  background: {C_SURFACE}; color: {C_TEXT};"
            f"  selection-background-color: {C_ACCENT};"
            f"}}"
        )

    @staticmethod
    def _primary_btn_style() -> str:
        return (
            f"QPushButton {{"
            f"  background: {C_ACCENT}; color: white;"
            f"  border: none; border-radius: 6px;"
            f"  padding: 7px 16px; font-size: 12px;"
            f"}}"
            f"QPushButton:hover {{ background: #9580ff; }}"
            f"QPushButton:disabled {{ background: {C_MUTED}; }}"
        )

    @staticmethod
    def _secondary_btn_style() -> str:
        return (
            f"QPushButton {{"
            f"  background: {C_SURFACE2}; color: {C_TEXT};"
            f"  border: 1px solid #44475a; border-radius: 6px;"
            f"  padding: 7px 14px; font-size: 12px;"
            f"}}"
            f"QPushButton:hover {{ background: #3d3f5a; }}"
        )

    @staticmethod
    def _confirm_btn_style() -> str:
        return (
            f"QPushButton {{"
            f"  background: #1a4731; color: {C_GREEN};"
            f"  border: 1px solid {C_GREEN}; border-radius: 6px;"
            f"  padding: 7px 16px; font-size: 12px; font-weight: 600;"
            f"}}"
            f"QPushButton:hover {{ background: #1f5a3c; }}"
            f"QPushButton:disabled {{ opacity: 0.5; }}"
        )

    @staticmethod
    def _delete_btn_style() -> str:
        return (
            f"QPushButton {{"
            f"  background: transparent; color: {C_RED};"
            f"  border: 1px solid {C_RED}; border-radius: 6px;"
            f"  padding: 7px 14px; font-size: 12px;"
            f"}}"
            f"QPushButton:hover {{ background: #3d1515; }}"
        )

    @staticmethod
    def _list_style() -> str:
        return (
            f"QListWidget {{"
            f"  background: {C_SURFACE}; color: {C_TEXT};"
            f"  border: none; font-size: 11px;"
            f"  outline: none;"
            f"}}"
            f"QListWidget::item {{"
            f"  padding: 10px 12px; border-bottom: 1px solid #313244;"
            f"}}"
            f"QListWidget::item:selected {{"
            f"  background: #383858; color: white;"
            f"  border-left: 3px solid {C_ACCENT};"
            f"}}"
            f"QListWidget::item:hover:!selected {{ background: #303050; }}"
        )
