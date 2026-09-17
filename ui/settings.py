"""
Settings Dialog
Allows the user to configure:
- Preferred microphone device
- Global hotkey
- Whisper model size
- Default output mode
- Personal dictionary entries
"""

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
OUTPUT_MODES = ["Auto", "Chat", "Email", "Notes", "Search", "Code"]


class SettingsDialog(QDialog):
    """Modal settings dialog."""

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self.setWindowTitle("Settings — FlowNLP")
        self.setMinimumWidth(440)
        self._setup_ui()
        self._load_settings()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        tabs = QTabWidget()

        # ----- Tab: General -----
        general_tab = QWidget()
        form = QFormLayout(general_tab)

        self._hotkey_edit = QLineEdit()
        self._hotkey_edit.setPlaceholderText("e.g. ctrl+shift+space")
        form.addRow("Global hotkey:", self._hotkey_edit)

        self._model_combo = QComboBox()
        self._model_combo.addItems(WHISPER_MODELS)
        form.addRow("Whisper model:", self._model_combo)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(OUTPUT_MODES)
        form.addRow("Default mode:", self._mode_combo)

        tabs.addTab(general_tab, "General")

        # ----- Tab: Audio -----
        audio_tab = QWidget()
        audio_form = QFormLayout(audio_tab)

        self._mic_combo = QComboBox()
        self._populate_microphones()
        audio_form.addRow("Microphone:", self._mic_combo)

        tabs.addTab(audio_tab, "Audio")

        # ----- Tab: Dictionary -----
        dict_tab = QWidget()
        dict_layout = QVBoxLayout(dict_tab)
        dict_layout.addWidget(QLabel("Personal dictionary words (one per line):"))
        self._dict_list = QListWidget()
        dict_layout.addWidget(self._dict_list)

        dict_btn_row = QVBoxLayout()
        add_btn = QPushButton("Add word …")
        add_btn.clicked.connect(self._add_dict_word)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_dict_word)
        dict_btn_row.addWidget(add_btn)
        dict_btn_row.addWidget(remove_btn)
        dict_layout.addLayout(dict_btn_row)

        tabs.addTab(dict_tab, "Dictionary")

        layout.addWidget(tabs)

        # ----- Buttons -----
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_settings)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _populate_microphones(self) -> None:
        try:
            import sounddevice as sd  # type: ignore
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    self._mic_combo.addItem(f"{i}: {dev['name']}", userData=i)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not enumerate audio devices: %s", exc)
            self._mic_combo.addItem("Default")

    def _load_settings(self) -> None:
        """Populate form fields from stored settings."""
        try:
            db = self._controller._db
            self._hotkey_edit.setText(db.get_setting("hotkey", "ctrl+shift+space"))
            model = db.get_setting("whisper_model", "base")
            idx = self._model_combo.findText(model)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)

            mode = db.get_setting("output_mode", "Auto")
            idx = self._mode_combo.findText(mode)
            if idx >= 0:
                self._mode_combo.setCurrentIndex(idx)

            words = db.get_dictionary_words()
            for w in words:
                self._dict_list.addItem(w)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load settings: %s", exc)

    def _save_settings(self) -> None:
        try:
            db = self._controller._db
            db.set_setting("hotkey", self._hotkey_edit.text())
            db.set_setting("whisper_model", self._model_combo.currentText())
            db.set_setting("output_mode", self._mode_combo.currentText())

            words = [
                self._dict_list.item(i).text()
                for i in range(self._dict_list.count())
            ]
            db.set_dictionary_words(words)
            logger.info("Settings saved.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save settings: %s", exc)
        self.accept()

    def _add_dict_word(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        word, ok = QInputDialog.getText(self, "Add word", "Word:")
        if ok and word.strip():
            self._dict_list.addItem(word.strip())

    def _remove_dict_word(self) -> None:
        row = self._dict_list.currentRow()
        if row >= 0:
            self._dict_list.takeItem(row)
