"""
Eavsmopper - live meeting transcription + answers from your own files.

Run from the Eavsmopper folder (with the backend venv active):
    python main.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))   # so "backend" and "frontend" import as packages

from backend.settings import load_settings  # noqa: E402
from frontend.overlay import Overlay  # noqa: E402


def main():
    load_dotenv(ROOT / ".env")   # API keys, only needed for the optional cloud providers
    settings_path = ROOT / "settings.json"
    settings = load_settings(settings_path)

    app = QApplication(sys.argv)
    # Our overlay is a "tool" window, which Qt doesn't count as a main window. Without this,
    # closing the folder picker looked like "last window closed" and Qt quit the whole app.
    app.setQuitOnLastWindowClosed(False)
    window = Overlay(ROOT, settings, settings_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
