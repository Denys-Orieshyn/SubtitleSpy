"""Pick a screen rectangle and print its coordinates.

Run:
    python tools/get_coords.py

Drag around the subtitle area. On mouse release the window closes and prints:
    left=..., top=..., width=..., height=...
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import mss
import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget


@dataclass(frozen=True, slots=True)
class Selection:
    left: int
    top: int
    width: int
    height: int

    def __str__(self) -> str:
        return (
            f"left={self.left}, top={self.top}, "
            f"width={self.width}, height={self.height}"
        )


class SelectionWindow(QWidget):
    def __init__(self, pixmap: QPixmap, monitor_left: int, monitor_top: int) -> None:
        super().__init__()
        self._pixmap = pixmap
        self._monitor_left = monitor_left
        self._monitor_top = monitor_top
        self._start: QPoint | None = None
        self._current: QPoint | None = None
        self.selection: Selection | None = None

        self.setWindowTitle("SubtitleSpy - select subtitle area")
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setGeometry(
            monitor_left,
            monitor_top,
            pixmap.width(),
            pixmap.height(),
        )

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))

        selection_rect = self._selection_rect()
        if selection_rect.isNull():
            return

        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.fillRect(selection_rect, Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        pen = QPen(QColor(0, 190, 255), 2)
        painter.setPen(pen)
        painter.drawRect(selection_rect.adjusted(0, 0, -1, -1))

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._start = event.position().toPoint()
        self._current = self._start
        self.update()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._start is None:
            return
        self._current = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() != Qt.MouseButton.LeftButton or self._start is None:
            return

        self._current = event.position().toPoint()
        rect = self._selection_rect()
        if rect.width() > 0 and rect.height() > 0:
            self.selection = Selection(
                left=self._monitor_left + rect.left(),
                top=self._monitor_top + rect.top(),
                width=rect.width(),
                height=rect.height(),
            )
        QApplication.quit()

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key() == Qt.Key.Key_Escape:
            QApplication.quit()

    def _selection_rect(self) -> QRect:
        if self._start is None or self._current is None:
            return QRect()
        return QRect(self._start, self._current).normalized()


def capture_virtual_screen() -> tuple[QPixmap, int, int]:
    with mss.mss() as screen:
        monitor = screen.monitors[0]
        shot = screen.grab(monitor)

    image = np.asarray(shot)
    height, width, _channels = image.shape
    bytes_per_line = image.strides[0]
    qimage = QImage(
        image.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_BGRA8888,
    ).copy()

    return QPixmap.fromImage(qimage), monitor["left"], monitor["top"]


def main() -> int:
    print("Switch to the video/player window now. Capturing in:")
    for second in range(3, 0, -1):
        print(f"{second}...")
        time.sleep(1)

    app = QApplication(sys.argv)
    pixmap, monitor_left, monitor_top = capture_virtual_screen()
    window = SelectionWindow(pixmap, monitor_left, monitor_top)
    window.showFullScreen()
    app.exec()

    if window.selection is None:
        print("Selection cancelled.")
        return 1

    print(window.selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
