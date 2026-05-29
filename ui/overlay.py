"""Transparent always-on-top overlay window."""

from __future__ import annotations

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QFont, QGuiApplication
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from core.capture import ScreenRegion
from core.image_prep import ThresholdSettings
from core.llm_client import LLMClient
from core.ocr import RapidOcrEngine
from ui.workers import TranslationWorker


class OverlayWindow(QWidget):
    """Main transparent overlay that displays translation results."""

    def __init__(
        self,
        *,
        region: ScreenRegion,
        prep_settings: ThresholdSettings,
        ocr_engine: RapidOcrEngine,
        llm_client: LLMClient,
    ) -> None:
        super().__init__()
        self._region = region
        self._prep_settings = prep_settings
        self._ocr_engine = ocr_engine
        self._llm_client = llm_client
        self._worker: TranslationWorker | None = None
        self._drag_position: QPoint | None = None
        self._manual_position = False

        self._configure_window()
        self._build_label()
        self._resize_and_position()

    def request_translation(self) -> None:
        """Start a translation job if one is not already running."""

        if self._worker is not None and self._worker.isRunning():
            self.show_message("Обрабатываю предыдущий запрос...")
            return

        self.show_message("Обрабатываю...")
        self._worker = TranslationWorker(
            region=self._region,
            prep_settings=self._prep_settings,
            ocr_engine=self._ocr_engine,
            llm_client=self._llm_client,
        )
        self._worker.result_ready.connect(self.show_message)
        self._worker.error_occurred.connect(self.show_error)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    def show_message(self, text: str) -> None:
        self._label.setText(text)
        self._label.setStyleSheet(_label_stylesheet(border_color="rgba(0, 190, 255, 170)"))
        self._resize_and_position()
        self.show()
        self.raise_()

    def show_error(self, text: str) -> None:
        self._label.setText(f"Ошибка: {text}")
        self._label.setStyleSheet(_label_stylesheet(border_color="rgba(255, 90, 90, 180)"))
        self._resize_and_position()
        self.show()
        self.raise_()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(1000)
        self._llm_client.close()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drag_position is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            self._manual_position = True
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.RightButton:
            self.hide()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drag_position is not None:
            self._drag_position = None
            self._manual_position = True
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton:
            self.hide()
            return
        super().mouseDoubleClickEvent(event)

    def _worker_finished(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def _configure_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setWindowTitle("SubtitleSpy")

    def _build_label(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self._label = QLabel("")
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._label.setFont(QFont("Segoe UI", 13))
        self._label.setMinimumWidth(520)
        self._label.setMaximumWidth(860)
        self._label.setMinimumHeight(64)
        self._label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._label.setStyleSheet(_label_stylesheet(border_color="rgba(0, 190, 255, 170)"))
        layout.addWidget(self._label)

    def _resize_and_position(self) -> None:
        self.adjustSize()
        if not self._manual_position:
            self._place_near_bottom()

    def _place_near_bottom(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return

        geometry = screen.availableGeometry()
        width = max(self.sizeHint().width(), 560)
        height = max(self.sizeHint().height(), 96)
        x = geometry.left() + (geometry.width() - width) // 2
        y = geometry.bottom() - height - 64
        self.setGeometry(x, y, width, height)


def _label_stylesheet(*, border_color: str) -> str:
    return f"""
        QLabel {{
            color: white;
            background-color: rgba(0, 0, 0, 185);
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 16px 18px;
            line-height: 1.35;
        }}
    """
