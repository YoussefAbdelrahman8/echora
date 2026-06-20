import numpy as np

from src.perception.banknote import BanknoteDetector


def _detector_with_label(label: str) -> BanknoteDetector:
    detector = BanknoteDetector()
    detector._ready = True
    detector._run_yolo_cached = lambda _: [{"label": label, "confidence": 0.99}]
    return detector


def _classify_until_stable(detector: BanknoteDetector) -> str:
    result = ""
    for _ in range(3):
        result = detector.classify_denomination(np.zeros((1, 1, 3), dtype=np.uint8))
    return result


def test_banknote_denomination_is_announced_once_per_scan():
    detector = _detector_with_label("50 Egyptian pounds")

    assert _classify_until_stable(detector) == "50 Egyptian pounds"
    assert _classify_until_stable(detector) == ""

    detector._run_yolo_cached = lambda _: [{"label": "100 Egyptian pounds", "confidence": 0.99}]
    assert _classify_until_stable(detector) == "100 Egyptian pounds"

    detector._run_yolo_cached = lambda _: [{"label": "50 Egyptian pounds", "confidence": 0.99}]
    assert _classify_until_stable(detector) == ""


def test_banknote_reset_allows_a_new_scan_to_announce_the_denomination():
    detector = _detector_with_label("20 Egyptian pounds")

    assert _classify_until_stable(detector) == "20 Egyptian pounds"
    detector.reset()
    assert _classify_until_stable(detector) == "20 Egyptian pounds"
