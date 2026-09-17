"""
FlowNLP — Entry Point
Initialises the application, sets up the storage layer, starts the UI,
and registers the global hotkey.
"""

import os
import sys
import logging

# Force all HuggingFace libraries to use local cache only.
# Set to "1" once all models have been downloaded at least once.
# Leave as "0" on first run to allow model downloads.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")

from PySide6.QtWidgets import QApplication

from app.controller import Controller
from storage.database import Database
from ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("flownlp")


def main() -> None:
    logger.info("Starting FlowNLP …")

    # Initialise persistent storage
    db = Database()
    db.initialise()

    # Qt application
    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("FlowNLP")
    qt_app.setApplicationVersion("1.0.0")

    # Central controller wires all subsystems together
    controller = Controller(db=db)

    # Main window
    window = MainWindow(controller=controller)
    window.show()

    logger.info("FlowNLP ready.")
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
