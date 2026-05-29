"""Temporary integration check for the SubtitleSpy core pipeline.

Run this script after installing dependencies and adjust TEST_REGION for your
monitor/player layout. The script saves debug_prep.png so you can inspect OCR
preprocessing by eye.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2

from core.capture import ScreenRegion, capture_region
from core.image_prep import ThresholdSettings, preprocess_for_ocr
from core.llm_client import DEFAULT_GROQ_MODEL, LLMClient
from core.ocr import RapidOcrEngine


PROJECT_ROOT = Path(__file__).resolve().parent
DEBUG_PREP_PATH = PROJECT_ROOT / "debug_prep.png"
ENV_PATH = PROJECT_ROOT / ".env"

# Adjust these coordinates for your monitor and subtitle area.
TEST_REGION = ScreenRegion(
    left=400,
    top=800,
    width=1000,
    height=150,
)

# Start with Otsu because subtitle backgrounds vary a lot. If the saved
# debug_prep.png looks too noisy, try use_otsu=False and threshold=180..220.
PREP_SETTINGS = ThresholdSettings(
    threshold=180,
    use_otsu=True,
    blur_kernel_size=3,
)

def load_env_file(path: Path = ENV_PATH) -> None:
    """Load KEY=VALUE pairs from a local .env file if they are not set yet."""

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


load_env_file()
API_KEY = os.getenv("GROQ_API_KEY", "PASTE_YOUR_GROQ_API_KEY_HERE")


def main() -> None:
    import time

    print("=== SubtitleSpy core pipeline test ===")
    print(f"Capture region: {TEST_REGION}")
    print("Switch to the video/player window now. Capturing in:")
    for second in range(3, 0, -1):
        print(f"{second}...")
        time.sleep(1)

    print("\n[1/4] Capturing screen region...")
    frame = capture_region(TEST_REGION)
    print(f"Captured frame shape: {frame.shape}")

    print("\n[2/4] Preprocessing image for OCR...")
    prepared = preprocess_for_ocr(frame, PREP_SETTINGS, output_channels=3)
    if not cv2.imwrite(str(DEBUG_PREP_PATH), prepared):
        raise RuntimeError(f"Failed to write {DEBUG_PREP_PATH}")
    print(f"Saved preprocessed image: {DEBUG_PREP_PATH}")

    print("\n[3/4] Running RapidOCR...")
    ocr = RapidOcrEngine(min_confidence=0.3)
    result = ocr.recognize(prepared)

    if result.is_empty:
        print("OCR text: <empty>")
    else:
        print(f"OCR text: {result.text}")

    if result.elapsed is not None:
        print(f"OCR elapsed: {result.elapsed:.3f}s")

    print("\nOCR lines:")
    if not result.lines:
        print("  <no lines>")
    for index, line in enumerate(result.lines, start=1):
        confidence = "n/a" if line.confidence is None else f"{line.confidence:.3f}"
        print(f"  {index}. confidence={confidence} text={line.text!r}")

    print("\n[4/4] Calling LLM client...")
    if result.is_empty:
        print("Skipping LLM call because OCR returned no text.")
        return

    if API_KEY == "PASTE_YOUR_GROQ_API_KEY_HERE":
        print("Skipping LLM call: paste Groq API_KEY in test_pipeline.py first.")
        return

    client = LLMClient(api_key=API_KEY, model=DEFAULT_GROQ_MODEL)
    try:
        try:
            llm_response = client.translate_subtitle(
                result.text,
                target_language="Russian",
            )
        except Exception as exc:
            print(f"LLM call failed: {type(exc).__name__}: {exc}")
            return
    finally:
        client.close()

    print("\nLLM response:")
    print(llm_response.output_text)
    print(f"from_cache={llm_response.from_cache} input_hash={llm_response.input_hash}")


if __name__ == "__main__":
    main()
