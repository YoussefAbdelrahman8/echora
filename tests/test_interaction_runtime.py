import numpy as np

from src.core.config import settings
from src.perception import interaction_detection as interaction
from src.perception.interaction_detection import InteractionDetector, InteractionPhase


class _NoopHaptic:
    def __init__(self):
        self.sent = []
        self.success_pulses = 0

    def send_grid(self, grid):
        self.sent.append(grid)
        return True

    def pulse_success(self):
        self.success_pulses += 1

    def all_off(self):
        return True


def _det(distance_mm=700.0):
    return {
        "label": "cup",
        "bbox": (10, 10, 80, 90),
        "distance_mm": distance_mm,
        "confidence": 0.9,
    }


def _success_vector():
    return {
        "dx_px": 5,
        "dy_px": 5,
        "dz_mm": 20.0,
        "dx_norm": 0.01,
        "dy_norm": 0.01,
        "dz_norm": 0.02,
        "xy_dist_px": 7.0,
        "finger_depth_mm": 680.0,
        "obj_depth_mm": 700.0,
    }


def test_interaction_uses_4_by_5_grid_config():
    assert settings.HAPTIC_ROWS == 4
    assert settings.HAPTIC_COLS == 5

    detector = InteractionDetector()

    assert detector._last_grid.shape == (4, 5)


def test_edge_phase_requires_xy_and_depth_alignment():
    detector = InteractionDetector()
    hand = {"index_tip": (50, 50)}
    target = {"center": (80, 60)}

    far_depth_vector = dict(_success_vector(), xy_dist_px=40.0, dz_mm=400.0)
    edge_vector = dict(_success_vector(), xy_dist_px=40.0, dz_mm=200.0)

    assert detector._compute_phase(hand, target, far_depth_vector) == InteractionPhase.GUIDANCE
    assert detector._compute_phase(hand, target, edge_vector) == InteractionPhase.EDGE


def test_success_requires_dwell_before_on_target(monkeypatch):
    detector = InteractionDetector()
    detector._ready = True
    detector._haptic = _NoopHaptic()
    detector._detect_right_hand = lambda frame: {"index_tip": (20, 20)}
    detector._filter_interactables = lambda detections, depth: [_det()]
    detector._compute_3d_vector = lambda hand, target, depth, frame: _success_vector()

    now = [100.0]
    monkeypatch.setattr(interaction.time, "time", lambda: now[0])

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.full((100, 100), 700, dtype=np.float32)

    first = detector.update(frame, depth, [_det()])
    assert first["phase"] == InteractionPhase.SUCCESS
    assert first["on_target"] is False

    now[0] += settings.INTERACTION_MIN_DWELL_SEC - 0.1
    almost = detector.update(frame, depth, [_det()])
    assert almost["on_target"] is False

    now[0] += 0.2
    reached = detector.update(frame, depth, [_det()])
    assert reached["on_target"] is True

    repeated = detector.update(frame, depth, [_det()])
    assert repeated["on_target"] is False
    assert detector._haptic.success_pulses == 1


def test_filter_interactables_copies_detection_and_drops_invalid_depth(monkeypatch):
    detector = InteractionDetector()
    depth = np.zeros((100, 100), dtype=np.float32)
    original = _det(distance_mm=0.0)

    assert detector._filter_interactables([original], depth) == []
    assert original["distance_mm"] == 0.0

    depth[10:90, 10:80] = 650
    filtered = detector._filter_interactables([original], depth)

    assert len(filtered) == 1
    assert filtered[0]["distance_mm"] > 0
    assert "center" in filtered[0]
    assert "center" not in original
