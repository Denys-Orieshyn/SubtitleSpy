"""SubtitleSpy MVP entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication
from pynput import keyboard

from core.capture import ScreenRegion
from core.image_prep import ThresholdSettings
from core.llm_client import LLMClient
from core.ocr import RapidOcrEngine
from ui.overlay import OverlayWindow


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"

# Replace these with values from tools/get_coords.py after selecting subtitles.
TEST_REGION = ScreenRegion(
    left=400,
    top=800,
    width=1000,
    height=150,
)

PREP_SETTINGS = ThresholdSettings(
    threshold=180,
    use_otsu=True,
    blur_kernel_size=3,
)


class HotkeyBridge(QObject):
    activated = pyqtSignal()


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def create_hotkey_listener(bridge: HotkeyBridge) -> keyboard.GlobalHotKeys:
    return keyboard.GlobalHotKeys(
        {
            "<ctrl>+<space>": bridge.activated.emit,
        }
    )


def main() -> int:
    load_env_file()

    app = QApplication(sys.argv)
    bridge = HotkeyBridge()

    ocr_engine = RapidOcrEngine(min_confidence=0.3)
    llm_client = LLMClient()
    overlay = OverlayWindow(
        region=TEST_REGION,
        prep_settings=PREP_SETTINGS,
        ocr_engine=ocr_engine,
        llm_client=llm_client,
    )
    bridge.activated.connect(overlay.request_translation)

    listener = create_hotkey_listener(bridge)
    listener.start()

    def stop_listener() -> None:
        listener.stop()

    app.aboutToQuit.connect(stop_listener)
    print("SubtitleSpy MVP started. Press Ctrl+Space to translate the selected area.")
    print(f"Capture region: {TEST_REGION}")

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
