"""Image preprocessing for subtitle OCR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ThresholdSettings:
    """Settings for subtitle binarization."""

    threshold: int = 180
    max_value: int = 255
    invert: bool = False
    use_otsu: bool = False
    blur_kernel_size: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
        if not 1 <= self.max_value <= 255:
            raise ValueError("max_value must be between 1 and 255")
        if self.blur_kernel_size < 0:
            raise ValueError("blur_kernel_size cannot be negative")
        if self.blur_kernel_size and self.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be odd when enabled")


DEFAULT_THRESHOLD_SETTINGS = ThresholdSettings()


def preprocess_for_ocr(
    image: Any,
    settings: ThresholdSettings = DEFAULT_THRESHOLD_SETTINGS,
    *,
    output_channels: int = 3,
) -> Any:
    """Convert a captured frame into a high-contrast binary OCR input."""

    binary = binarize(to_grayscale(image), settings)
    if output_channels == 1:
        return binary
    if output_channels == 3:
        return ensure_three_channel(binary)
    raise ValueError("output_channels must be 1 or 3")


def to_grayscale(image: Any) -> Any:
    """Return a grayscale image from BGR/BGRA/RGB-like input."""

    cv2 = _load_cv2()
    np = _load_numpy()

    frame = np.asarray(image)
    if frame.ndim == 2:
        return frame
    if frame.ndim != 3:
        raise ValueError("image must be a 2D grayscale or 3D color array")

    channels = frame.shape[2]
    if channels == 1:
        return frame[:, :, 0]
    if channels == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    raise ValueError("image must have 1, 3, or 4 channels")


def binarize(grayscale_image: Any, settings: ThresholdSettings = DEFAULT_THRESHOLD_SETTINGS) -> Any:
    """Apply optional blur and binary thresholding to a grayscale image."""

    cv2 = _load_cv2()
    np = _load_numpy()

    gray = np.asarray(grayscale_image)
    if gray.ndim != 2:
        raise ValueError("grayscale_image must be a 2D array")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    if settings.blur_kernel_size:
        gray = cv2.medianBlur(gray, settings.blur_kernel_size)

    threshold_type = cv2.THRESH_BINARY_INV if settings.invert else cv2.THRESH_BINARY
    threshold = settings.threshold
    if settings.use_otsu:
        threshold_type |= cv2.THRESH_OTSU
        threshold = 0

    _, binary = cv2.threshold(gray, threshold, settings.max_value, threshold_type)
    return binary


def ensure_three_channel(image: Any) -> Any:
    """Convert a grayscale image to a 3-channel BGR image when needed."""

    cv2 = _load_cv2()
    np = _load_numpy()

    frame = np.asarray(image)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    raise ValueError("image must be grayscale, BGR, or BGRA")


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "The 'opencv-python' package is required for image preprocessing. "
            "Install project dependencies with 'pip install -r requirements.txt'."
        ) from exc
    return cv2


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "The 'numpy' package is required for image preprocessing. "
            "Install project dependencies with 'pip install -r requirements.txt'."
        ) from exc
    return np
