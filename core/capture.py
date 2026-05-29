"""Screen capture helpers built on top of mss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ScreenRegion:
    """A rectangular screen area in absolute desktop coordinates."""

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("ScreenRegion.width must be greater than zero")
        if self.height <= 0:
            raise ValueError("ScreenRegion.height must be greater than zero")

    @classmethod
    def from_bbox(cls, left: int, top: int, right: int, bottom: int) -> "ScreenRegion":
        """Create a region from left/top/right/bottom coordinates."""

        return cls(left=left, top=top, width=right - left, height=bottom - top)

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def to_mss_monitor(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


def capture_region(region: ScreenRegion, *, include_alpha: bool = False) -> Any:
    """Capture a screen region and return it as a NumPy array.

    By default the returned array is BGR, which is the most convenient format
    for OpenCV. Set include_alpha=True to keep MSS' native BGRA output.
    """

    mss_module = _load_mss()
    np = _load_numpy()

    with mss_module.mss() as screen:
        shot = screen.grab(region.to_mss_monitor())
        image = np.asarray(shot)

    if include_alpha:
        return np.ascontiguousarray(image)

    return np.ascontiguousarray(image[:, :, :3])


def _load_mss() -> Any:
    try:
        import mss
    except ImportError as exc:
        raise RuntimeError(
            "The 'mss' package is required for screen capture. "
            "Install project dependencies with 'pip install -r requirements.txt'."
        ) from exc
    return mss


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "The 'numpy' package is required for screen capture. "
            "Install project dependencies with 'pip install -r requirements.txt'."
        ) from exc
    return np
