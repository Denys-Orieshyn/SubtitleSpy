"""RapidOCR wrapper used by the SubtitleSpy core."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class OcrLine:
    text: str
    confidence: float | None = None
    box: Any | None = None


@dataclass(frozen=True, slots=True)
class OcrResult:
    text: str
    lines: tuple[OcrLine, ...]
    elapsed: float | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class RapidOcrEngine:
    """Small adapter around rapidocr-onnxruntime.

    Instantiate this class once at application start and reuse it for all
    captures. Model loading is intentionally kept out of per-frame work.
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.3,
        engine: Any | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")

        self.min_confidence = min_confidence
        self._engine = engine if engine is not None else self._create_engine(engine_kwargs or {})

    def recognize(self, image: Any | str | Path, *, min_confidence: float | None = None) -> OcrResult:
        """Recognize text from an image array or image path."""

        threshold = self.min_confidence if min_confidence is None else min_confidence
        if not 0 <= threshold <= 1:
            raise ValueError("min_confidence must be between 0 and 1")

        raw_result = self._engine(str(image) if isinstance(image, Path) else image)
        lines, elapsed = _normalize_rapidocr_result(raw_result)
        filtered = tuple(
            line for line in lines if line.confidence is None or line.confidence >= threshold
        )
        text = _join_lines(line.text for line in filtered)
        return OcrResult(text=text, lines=filtered, elapsed=elapsed)

    @staticmethod
    def _create_engine(engine_kwargs: dict[str, Any]) -> Any:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "The 'rapidocr-onnxruntime' package is required for OCR. "
                "Install project dependencies with 'pip install -r requirements.txt'."
            ) from exc
        return RapidOCR(**engine_kwargs)


def _normalize_rapidocr_result(raw_result: Any) -> tuple[tuple[OcrLine, ...], float | None]:
    elapsed: float | None = None
    entries = raw_result

    if isinstance(raw_result, tuple):
        if raw_result:
            entries = raw_result[0]
        if len(raw_result) > 1 and isinstance(raw_result[1], (int, float)):
            elapsed = float(raw_result[1])

    if not entries:
        return (), elapsed

    return tuple(_iter_ocr_lines(entries)), elapsed


def _iter_ocr_lines(entries: Iterable[Any]) -> Iterable[OcrLine]:
    for entry in entries:
        line = _parse_ocr_entry(entry)
        if line is not None and line.text.strip():
            yield line


def _parse_ocr_entry(entry: Any) -> OcrLine | None:
    if isinstance(entry, dict):
        text = entry.get("text") or entry.get("rec_text") or entry.get("label")
        confidence = entry.get("confidence") or entry.get("score")
        box = entry.get("box") or entry.get("bbox") or entry.get("points")
        return _make_line(text, confidence, box)

    if isinstance(entry, (list, tuple)):
        if len(entry) >= 3:
            return _make_line(entry[1], entry[2], entry[0])
        if len(entry) >= 2:
            return _make_line(entry[0], entry[1], None)
        if len(entry) == 1:
            return _make_line(entry[0], None, None)

    if isinstance(entry, str):
        return OcrLine(text=entry)

    return None


def _make_line(text: Any, confidence: Any, box: Any) -> OcrLine | None:
    if text is None:
        return None

    try:
        confidence_value = None if confidence is None else float(confidence)
    except (TypeError, ValueError):
        confidence_value = None

    return OcrLine(text=str(text).strip(), confidence=confidence_value, box=box)


def _join_lines(lines: Iterable[str]) -> str:
    return " ".join(part.strip() for part in lines if part.strip())
