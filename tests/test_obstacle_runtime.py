import builtins
import numpy as np

from src.core.config import settings
from src.perception.byte_tracker import ByteTracker
from src.perception.obstacle_detection import ObstacleDetector


def _det(label="chair", bbox=(500, 100, 780, 700), distance_mm=1700.0, urgency="WARNING"):
    return {
        "track_id": 1,
        "label": label,
        "bbox": bbox,
        "confidence": 0.9,
        "distance_mm": distance_mm,
        "angle_deg": 0.0,
        "urgency": urgency,
        "frame_shape": (800, 1280),
    }


def test_warning_obstacle_overlapping_corridor_promotes_to_danger():
    detector = ObstacleDetector()

    result = detector._filter_and_promote([_det(distance_mm=1700.0)])

    assert result[0]["urgency"] == "DANGER"


def test_invalid_depth_relevant_obstacle_in_corridor_is_kept_as_unknown():
    detector = ObstacleDetector()

    result = detector._filter_and_promote([_det(distance_mm=0.0, urgency="UNKNOWN")])

    assert len(result) == 1
    assert result[0]["urgency"] == "UNKNOWN"
    assert result[0]["distance_mm"] == 0.0


def test_invalid_depth_small_off_corridor_obstacle_is_dropped():
    detector = ObstacleDetector()
    small_side_obstacle = _det(
        bbox=(0, 100, 50, 180),
        distance_mm=0.0,
        urgency="UNKNOWN",
    )

    assert detector._filter_and_promote([small_side_obstacle]) == []


def test_interaction_only_classes_are_hidden_from_obstacle_output():
    tracks = [
        _det(label="cup", distance_mm=700.0, urgency="DANGER"),
        _det(label="chair", distance_mm=700.0, urgency="DANGER"),
    ]

    user_facing = ObstacleDetector._get_user_facing_obstacle_tracks(tracks)

    assert [track["label"] for track in user_facing] == ["chair"]
    assert tracks[0]["label"] == "cup"  # still available to interaction


def test_dual_model_detections_are_merged_before_tracking(monkeypatch):
    detector = ObstacleDetector()
    detector._yolo = object()
    detector._coco_yolo = object()
    detector._relevant_class_ids = [0]
    detector._coco_relevant_class_ids = [0]
    called_sources = []

    def fake_run(model, relevant_class_ids, source, track_id_offset, rgb_frame):
        called_sources.append(source)
        label = "door" if source == "accessibility" else "chair"
        return [{
            "track_id": track_id_offset + 1,
            "source": source,
            "label": label,
            "bbox": (500, 100, 780, 700),
            "confidence": 0.9,
        }]

    monkeypatch.setattr(detector, "_run_yolo", fake_run)
    frame = np.zeros((800, 1280, 3), dtype=np.uint8)
    depth = np.full((800, 1280), 2000, dtype=np.float32)

    detector.detect(frame, depth)  # first hit registers both tracks
    confirmed = detector.detect(frame, depth)

    assert called_sources == ["accessibility", "coco", "accessibility", "coco"]
    assert {track["label"] for track in confirmed} == {"door", "chair"}


def test_byte_tracker_confirms_after_two_hits_and_expires_after_misses():
    tracker = ByteTracker(frame_width=1280)
    det = _det(distance_mm=2000.0)

    assert tracker.update([dict(det)]) == []
    confirmed = tracker.update([dict(det)])
    assert len(confirmed) == 1
    assert confirmed[0]["state"] == "CONFIRMED"

    for _ in range(settings.KALMAN_MAX_MISSED_FRAMES + 1):
        tracker.update([])

    assert tracker.get_confirmed_tracks() == []


def test_vlm_disables_after_configured_failed_calls(monkeypatch):
    detector = ObstacleDetector()
    monkeypatch.setattr(settings, "VLM_ENABLED", True)
    monkeypatch.setattr(settings, "VLM_MAX_FAILURES", 5)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ollama":
            raise RuntimeError("ollama unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    for _ in range(5):
        detector._vlm_worker(frame)

    assert detector.get_stats()["vlm_disabled"] is True
    assert detector.get_stats()["vlm_failures"] == 5
