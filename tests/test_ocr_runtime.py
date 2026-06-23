from pathlib import Path

import numpy as np

from src.perception.ocr import (
    OCRReader,
    _enabled_ocr_languages,
    _normalise_ocr_mode,
    _valid_paddle_inference_dir,
)


def test_arabic_script_detection_handles_real_unicode():
    assert OCRReader._is_predominantly_arabic("مرحبا بالعالم")
    assert OCRReader._is_predominantly_arabic("Shop شارع التحرير")
    assert not OCRReader._is_predominantly_arabic("hello world")


def test_paddle_inference_dir_validation(tmp_path: Path):
    assert not _valid_paddle_inference_dir("")
    assert not _valid_paddle_inference_dir(str(tmp_path))

    (tmp_path / "inference.pdmodel").write_bytes(b"model")
    assert not _valid_paddle_inference_dir(str(tmp_path))

    (tmp_path / "inference.pdiparams").write_bytes(b"params")
    assert _valid_paddle_inference_dir(str(tmp_path))


def test_ocr_mode_selection():
    assert _normalise_ocr_mode("arabic") == "ar"
    assert _normalise_ocr_mode("english") == "en"
    assert _enabled_ocr_languages("ar") == (True, False)
    assert _enabled_ocr_languages("en") == (False, True)
    assert _enabled_ocr_languages("both") == (True, True)


def test_gpu_load_falls_back_to_cpu():
    calls = []

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            calls.append(kwargs["use_gpu"])
            if kwargs["use_gpu"]:
                raise RuntimeError("gpu failed")

    reader = OCRReader()
    instance = reader._create_paddle_ocr(FakePaddleOCR, {"use_gpu": True}, "Fake")

    assert isinstance(instance, FakePaddleOCR)
    assert calls == [True, False]


def test_stable_text_uses_normalised_comparison_but_returns_latest_text():
    reader = OCRReader()
    reader._text_history.extend(["Good morning!", "good morning", "Good   morning."])

    assert reader._stable_text() == "Good   morning."


def test_empty_ocr_results_are_cached():
    reader = OCRReader()
    reader._ready = True
    reader._is_frame_sharp_enough = lambda _: True
    calls = []
    reader._run_ocr_on_frame = lambda _: calls.append(1) or []
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    assert reader._run_ocr_cached(frame) == []
    assert reader._run_ocr_cached(frame) == []
    assert len(calls) == 1


def test_cached_result_counts_as_one_stability_observation():
    reader = OCRReader()
    reader._ready = True
    reader._inference_generation = 1
    reader._run_ocr_cached = lambda _: [{
        "text": "Exit", "confidence": 0.99, "bbox": (10, 10, 80, 40)
    }]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    reader.read_text(frame)
    reader.read_text(frame)
    reader.read_text(frame)

    assert list(reader._text_history) == ["Exit"]


def test_prioritise_prefers_closer_text_when_other_scores_are_similar():
    reader = OCRReader()
    detections = [
        {"text": "far", "bbox": (500, 300, 620, 360)},
        {"text": "near", "bbox": (660, 300, 780, 360)},
    ]
    depth = np.full((800, 1280), 2500, dtype=np.uint16)
    depth[300:360, 660:780] = 500

    ordered = reader._prioritise(detections, 1280, 800, depth_map=depth)

    assert ordered[0]["text"] == "near"


def test_document_reading_orders_lines_top_to_bottom_then_left_to_right():
    detections = [
        {"text": "two", "bbox": (75, 12, 110, 35)},
        {"text": "three", "bbox": (10, 55, 70, 80)},
        {"text": "one", "bbox": (10, 10, 60, 34)},
        {"text": "four", "bbox": (80, 57, 130, 82)},
    ]

    ordered = OCRReader._order_for_reading(detections)

    assert [det["text"] for det in ordered] == ["one", "two", "three", "four"]
