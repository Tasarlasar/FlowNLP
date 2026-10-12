"""
FlowNLP App — Desktop Entry Point
Initialises storage, wires up the controller, and launches the PySide6 UI.
"""

from __future__ import annotations

import logging
import os
import sys

# Allow HuggingFace downloads on first run; set to "1" after all models
# are cached locally to prevent any outbound network calls.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from app.controller import AppController
from storage.database import Database
from ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("flownlp.app")


def main() -> None:
    logger.info("Starting FlowNLP App …")

    # High-DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("FlowNLP")
    qt_app.setApplicationDisplayName("FlowNLP")
    qt_app.setApplicationVersion("2.0.0")

    # Persistent storage (SQLite)
    db = Database()
    db.initialise()

    # Central controller wires every subsystem together
    controller = AppController(db=db)

    # Main window
    window = MainWindow(controller=controller, db=db)
    window.show()

    logger.info("FlowNLP App ready.")
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
