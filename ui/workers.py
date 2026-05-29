"""Background workers for the PyQt UI layer."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from core.capture import ScreenRegion, capture_region
from core.image_prep import ThresholdSettings, preprocess_for_ocr
from core.llm_client import LLMClient
from core.ocr import RapidOcrEngine


class TranslationWorker(QThread):
    """Run capture, OCR, and LLM translation away from the GUI thread."""

    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        *,
        region: ScreenRegion,
        prep_settings: ThresholdSettings,
        ocr_engine: RapidOcrEngine,
        llm_client: LLMClient,
        target_language: str = "Russian",
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self._region = region
        self._prep_settings = prep_settings
        self._ocr_engine = ocr_engine
        self._llm_client = llm_client
        self._target_language = target_language

    def run(self) -> None:
        try:
            frame = capture_region(self._region)
            prepared = preprocess_for_ocr(frame, self._prep_settings, output_channels=3)
            ocr_result = self._ocr_engine.recognize(prepared)

            if ocr_result.is_empty:
                self.error_occurred.emit("OCR не нашёл текст в выбранной области.")
                return

            llm_response = self._llm_client.translate_subtitle(
                ocr_result.text,
                target_language=self._target_language,
            )
        except Exception as exc:
            self.error_occurred.emit(f"{type(exc).__name__}: {exc}")
            return

        self.result_ready.emit(llm_response.output_text)
