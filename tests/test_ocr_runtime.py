"""Runtime tests for the PaddleOCR-based OCR reader.

These exercise the OCRReader logic that does not require loading the heavy
PaddleOCR model (stability gating, text cleaning, prioritisation, caching).
"""
import numpy as np

import src.perception.ocr as ocr_mod
from src.core.config import settings
from src.perception.ocr import OCRReader, TEXT_STABILITY_FRAMES


def test_use_gpu_respects_settings(monkeypatch):
    monkeypatch.setattr(settings, "OCR_USE_GPU", False)
    assert ocr_mod._use_gpu() is False


def test_stable_text_requires_two_matching_reads():
    reader = OCRReader()
    reader._text_history.extend(["Exit", "Exit", "Other"])
    assert reader._stable_text() == "Exit"

    reader2 = OCRReader()
    reader2._text_history.extend(["aa", "bb", "cc"])
    assert reader2._stable_text() == ""


def test_stable_text_needs_full_window():
    reader = OCRReader()
    reader._text_history.extend(["Exit", "Exit"])  # fewer than the window
    assert len(reader._text_history) < TEXT_STABILITY_FRAMES
    assert reader._stable_text() == ""


def test_clean_text_dedupes_and_filters():
    reader = OCRReader()
    # "!!" is non-alphanumeric, "a" is too short; "Exit" is duplicated.
    assert reader._clean_text(["Exit", "Exit", "!!", "a"]) == "Exit"
    assert reader._clean_text(["Open", "Door"]) == "Open Door"


def test_prioritise_orders_central_text_first():
    reader = OCRReader()
    detections = [
        {"text": "corner", "bbox": (0, 0, 80, 60)},
        {"text": "center", "bbox": (600, 370, 680, 430)},
    ]
    ordered = reader._prioritise(detections, 1280, 800)
    assert ordered[0]["text"] == "center"


def test_full_ocr_cache_reuses_non_empty_result():
    reader = OCRReader()
    reader._is_frame_sharp_enough = lambda _: True
    calls = []
    reader._run_full_ocr_on_frame = lambda _: calls.append(1) or [
        {"text": "Exit", "confidence": 0.99, "bbox": (10, 10, 80, 40)}
    ]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    first = reader._run_full_ocr_cached(frame)
    second = reader._run_full_ocr_cached(frame)

    assert first == second
    assert len(calls) == 1  # second call served from cache


def test_read_text_waits_for_stable_window_then_speaks_once():
    reader = OCRReader()
    reader._ready = True
    reader._run_full_ocr_cached = lambda _: [
        {"text": "Exit", "confidence": 0.99, "bbox": (10, 10, 80, 40)}
    ]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Not enough observations yet -> nothing spoken.
    assert reader.read_text(frame) == ""
    assert reader.read_text(frame) == ""
    # Window is now full and the text is stable -> spoken once.
    assert reader.read_text(frame) == "Exit"
    # Identical text is not repeated.
    assert reader.read_text(frame) == ""


def test_reset_clears_reader_state():
    reader = OCRReader()
    reader._text_history.append("Exit")
    reader._last_spoken_text = "Exit"
    reader._full_boxes = [{"text": "Exit"}]
    reader._det_boxes = [(0, 0, 1, 1)]

    reader.reset()

    assert list(reader._text_history) == []
    assert reader._last_spoken_text == ""
    assert reader._full_boxes == []
    assert reader._det_boxes == []
