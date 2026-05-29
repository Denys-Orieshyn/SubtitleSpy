"""Core building blocks for SubtitleSpy.

The package intentionally contains no GUI code. These modules are safe to use
from a worker thread in the PyQt layer that will be added later.
"""

from .capture import ScreenRegion, capture_region
from .image_prep import ThresholdSettings, preprocess_for_ocr
from .llm_client import LLMClient, LLMResponse
from .ocr import OcrLine, OcrResult, RapidOcrEngine

__all__ = [
    "LLMClient",
    "LLMResponse",
    "OcrLine",
    "OcrResult",
    "RapidOcrEngine",
    "ScreenRegion",
    "ThresholdSettings",
    "capture_region",
    "preprocess_for_ocr",
]
