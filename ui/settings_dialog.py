"""
Settings Dialog
Tabbed modal with four sections:
  General   — Whisper model, default mode, auto-inject toggle, hotkey
  Audio     — microphone device picker
  Writing   — vocabulary, writing preferences, email template, notes template
  Dictionary — personal word list (add / remove)
"""

from __future__ import annotations

import json
import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFormLayout, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QListWidget, QPushButton, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget,
)

logger = logging.getLogger(__name__)

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
OUTPUT_MODES   = ["Auto", "Chat", "Email", "Notes", "Search", "Code"]

C_BG       = "#1e1e2e"
C_SURFACE  = "#2a2a3e"
C_SURFACE2 = "#313244"
C_ACCENT   = "#7c6af7"
C_TEXT     = "#cdd6f4"
C_TEXT_DIM = "#888ba8"
C_MUTED    = "#6272a4"


class SettingsDialog(QDialog):
    """Full-featured settings dialog."""

    def __init__(self, controller, db, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._db = db
        self.setWindowTitle("Settings — FlowNLP")
        self.setMinimumSize(520, 560)
        self.setStyleSheet(self._dialog_style())
        self._setup_ui()
        self._load()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(20, 20, 20, 20)

        tabs = QTabWidget()
        tabs.setStyleSheet(self._tab_style())

        tabs.addTab(self._general_tab(),    "General")
        tabs.addTab(self._audio_tab(),      "Audio")
        tabs.addTab(self._writing_tab(),    "Writing")
        tabs.addTab(self._dictionary_tab(), "Dictionary")

        root.addWidget(tabs)

        # Button row
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setStyleSheet(
            self._primary_btn_style()
        )
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setStyleSheet(
            self._secondary_btn_style()
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _general_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        form.setSpacing(14)
        form.setContentsMargins(16, 16, 16, 16)

        self._hotkey_edit = QLineEdit()
        self._hotkey_edit.setPlaceholderText("<ctrl>+<shift>+<space>")
        self._hotkey_edit.setStyleSheet(self._input_style())
        form.addRow("Global hotkey:", self._hotkey_edit)

        self._model_combo = QComboBox()
        self._model_combo.addItems(WHISPER_MODELS)
        self._model_combo.setStyleSheet(self._combo_style())
        form.addRow("Whisper model:", self._model_combo)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(OUTPUT_MODES)
        self._mode_combo.setStyleSheet(self._combo_style())
        form.addRow("Default mode:", self._mode_combo)

        self._inject_check = QCheckBox("Auto-inject into active window after recording")
        self._inject_check.setStyleSheet(f"color: {C_TEXT}; font-size: 12px;")
        form.addRow("Auto-inject:", self._inject_check)

        note = QLabel(
            "Auto-inject uses PyAutoGUI to type text into whatever app is focused.\n"
            "Leave off to review results in the Library first."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
        form.addRow("", note)

        return tab

    def _audio_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        form.setSpacing(14)
        form.setContentsMargins(16, 16, 16, 16)

        self._mic_combo = QComboBox()
        self._mic_combo.setStyleSheet(self._combo_style())
        self._populate_microphones()
        form.addRow("Microphone:", self._mic_combo)

        note = QLabel(
            "Restart recording after changing the microphone for the new\n"
            "device to take effect."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
        form.addRow("", note)

        return tab

    def _writing_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        def _area(placeholder: str) -> QTextEdit:
            t = QTextEdit()
            t.setPlaceholderText(placeholder)
            t.setStyleSheet(self._textarea_style())
            t.setFixedHeight(90)
            return t

        layout.addWidget(self._sub_label("Personal vocabulary (comma-separated terms)"))
        self._vocab_edit = _area("e.g. Kubernetes, PyTorch, GDPR…")
        layout.addWidget(self._vocab_edit)

        layout.addWidget(self._sub_label("Writing preferences"))
        self._prefs_edit = _area(
            "e.g. Always use British spelling. Keep sentences under 20 words."
        )
        layout.addWidget(self._prefs_edit)

        layout.addWidget(self._sub_label("Email template hint"))
        self._email_edit = _area(
            "e.g. Start with 'Dear [Name],' and end with 'Kind regards,'"
        )
        layout.addWidget(self._email_edit)

        layout.addWidget(self._sub_label("Notes template hint"))
        self._notes_edit = _area("e.g. Use bullet points. Prefix action items with '→'.")
        layout.addWidget(self._notes_edit)

        return tab

    def _dictionary_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        layout.addWidget(
            self._sub_label("Words that will never be altered by grammar or rewrite")
        )

        self._dict_list = QListWidget()
        self._dict_list.setStyleSheet(self._list_style())
        layout.addWidget(self._dict_list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add word…")
        add_btn.setStyleSheet(self._secondary_btn_style())
        add_btn.clicked.connect(self._add_dict_word)
        rm_btn = QPushButton("Remove selected")
        rm_btn.setStyleSheet(self._secondary_btn_style())
        rm_btn.clicked.connect(self._remove_dict_word)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(rm_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        return tab

    # ── Load / Save ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            db = self._db

            self._hotkey_edit.setText(
                db.get_setting("hotkey", "<ctrl>+<shift>+<space>")
            )
            model = db.get_setting("whisper_model", "base")
            idx = self._model_combo.findText(model)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)

            mode = db.get_setting("output_mode", "Auto")
            idx = self._mode_combo.findText(mode)
            if idx >= 0:
                self._mode_combo.setCurrentIndex(idx)

            auto_inject = db.get_setting("auto_inject", "false") == "true"
            self._inject_check.setChecked(auto_inject)

            prefs = db.get_preferences()
            vocab_list = prefs.get("vocabulary", [])
            self._vocab_edit.setPlainText(", ".join(vocab_list))
            self._prefs_edit.setPlainText(prefs.get("writing_preferences", ""))
            self._email_edit.setPlainText(prefs.get("email_template", ""))
            self._notes_edit.setPlainText(prefs.get("notes_template", ""))

            for word in db.get_dictionary_words():
                self._dict_list.addItem(word)

        except Exception as exc:
            logger.warning("Could not load settings: %s", exc)

    def _save(self) -> None:
        try:
            db = self._db

            new_hotkey = self._hotkey_edit.text().strip()
            old_hotkey = db.get_setting("hotkey", "<ctrl>+<shift>+<space>")
            db.set_setting("hotkey", new_hotkey or old_hotkey)
            if new_hotkey and new_hotkey != old_hotkey:
                self._controller.update_hotkey(new_hotkey)

            db.set_setting("whisper_model", self._model_combo.currentText())
            db.set_setting("output_mode",   self._mode_combo.currentText())
            db.set_setting(
                "auto_inject",
                "true" if self._inject_check.isChecked() else "false"
            )

            vocab_raw = self._vocab_edit.toPlainText()
            vocab_list = [
                v.strip() for v in vocab_raw.split(",") if v.strip()
            ]
            db.set_preferences(
                vocabulary=vocab_list,
                writing_preferences=self._prefs_edit.toPlainText(),
                email_template=self._email_edit.toPlainText(),
                notes_template=self._notes_edit.toPlainText(),
            )

            words = [
                self._dict_list.item(i).text()
                for i in range(self._dict_list.count())
            ]
            db.set_dictionary_words(words)

            logger.info("Settings saved.")
        except Exception as exc:
            logger.warning("Could not save settings: %s", exc)
        self.accept()

    # ── Dictionary helpers ────────────────────────────────────────────────────

    def _add_dict_word(self) -> None:
        word, ok = QInputDialog.getText(self, "Add word", "Enter word:")
        if ok and word.strip():
            self._dict_list.addItem(word.strip())

    def _remove_dict_word(self) -> None:
        row = self._dict_list.currentRow()
        if row >= 0:
            self._dict_list.takeItem(row)

    def _populate_microphones(self) -> None:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    self._mic_combo.addItem(f"{i}: {dev['name']}", userData=i)
        except Exception as exc:
            logger.warning("Could not enumerate audio devices: %s", exc)
            self._mic_combo.addItem("Default")

    # ── Shared widget helpers ─────────────────────────────────────────────────

    @staticmethod
    def _sub_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
        return lbl

    # ── Styles ────────────────────────────────────────────────────────────────

    @staticmethod
    def _dialog_style() -> str:
        return (
            f"QDialog {{ background: {C_BG}; color: {C_TEXT}; }}"
            f"QLabel  {{ color: {C_TEXT}; font-size: 12px; }}"
        )

    @staticmethod
    def _tab_style() -> str:
        return (
            f"QTabWidget::pane {{ border: none; background: {C_BG}; }}"
            f"QTabBar::tab {{"
            f"  background: {C_SURFACE}; color: {C_TEXT_DIM};"
            f"  padding: 8px 18px; border: none;"
            f"}}"
            f"QTabBar::tab:selected {{ background: {C_BG}; color: {C_TEXT};"
            f"  border-bottom: 2px solid {C_ACCENT}; }}"
            f"QTabBar::tab:hover {{ color: {C_TEXT}; }}"
        )

    @staticmethod
    def _input_style() -> str:
        return (
            f"QLineEdit {{"
            f"  background: {C_SURFACE}; color: {C_TEXT};"
            f"  border: 1px solid #44475a; border-radius: 6px;"
            f"  padding: 6px 10px; font-size: 12px;"
            f"}}"
            f"QLineEdit:focus {{ border-color: {C_ACCENT}; }}"
        )

    @staticmethod
    def _combo_style() -> str:
        return (
            f"QComboBox {{"
            f"  background: {C_SURFACE}; color: {C_TEXT};"
            f"  border: 1px solid #44475a; border-radius: 6px;"
            f"  padding: 6px 10px; font-size: 12px; min-width: 160px;"
            f"}}"
            f"QComboBox::drop-down {{ border: none; }}"
            f"QComboBox QAbstractItemView {{"
            f"  background: {C_SURFACE}; color: {C_TEXT};"
            f"  selection-background-color: {C_ACCENT};"
            f"}}"
        )

    @staticmethod
    def _textarea_style() -> str:
        return (
            f"QTextEdit {{"
            f"  background: {C_SURFACE}; color: {C_TEXT};"
            f"  border: 1px solid #44475a; border-radius: 6px;"
            f"  padding: 8px; font-size: 12px;"
            f"}}"
            f"QTextEdit:focus {{ border-color: {C_ACCENT}; }}"
        )

    @staticmethod
    def _list_style() -> str:
        return (
            f"QListWidget {{"
            f"  background: {C_SURFACE}; color: {C_TEXT};"
            f"  border: 1px solid #44475a; border-radius: 6px; font-size: 12px;"
            f"}}"
            f"QListWidget::item {{ padding: 6px 10px; }}"
            f"QListWidget::item:selected {{ background: {C_ACCENT}; color: white; }}"
        )

    @staticmethod
    def _primary_btn_style() -> str:
        return (
            f"QPushButton {{"
            f"  background: {C_ACCENT}; color: white;"
            f"  border: none; border-radius: 6px;"
            f"  padding: 8px 20px; font-size: 13px;"
            f"}}"
            f"QPushButton:hover {{ background: #9580ff; }}"
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
