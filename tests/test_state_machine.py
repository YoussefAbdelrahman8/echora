import sys
import time
from pathlib import Path

# Add parent directory to path to import echora modules
sys.path.append(str(Path(__file__).parent.parent))

from src.core.state_machine import StateMachine
from src.core.config import settings, MODE
from src.core.utils import get_timestamp_ms

if __name__ == "__main__":
    print("=== ECHORA state_machine.py self-test ===\n")
    sm = StateMachine()

    def on_enter_ocr():
        print("  [CALLBACK] Entering OCR mode — stopping obstacle audio")

    def on_exit_ocr():
        print("  [CALLBACK] Exiting OCR mode — resuming obstacle audio")

    def on_enter_face():
        print("  [CALLBACK] Entering FACE_ID mode — starting face recognition")

    sm.register_callback(mode=MODE.OCR, on_enter=on_enter_ocr, on_exit=on_exit_ocr)
    sm.register_callback(mode=MODE.FACE_ID, on_enter=on_enter_face)

    def make_bundle(accel_x=0.0, accel_y=0.0, accel_z=9.81):
        return {
            "rgb": None, "depth": None,
            "imu": {
                "accel": {"x": accel_x, "y": accel_y, "z": accel_z},
                "gyro": {"x": 0.0, "y": 0.0, "z": 0.0},
                "timestamp_ms": get_timestamp_ms()
            },
            "timestamp_ms": get_timestamp_ms()
        }

    def make_obstacle_result(danger=False, warning=False):
        danger_track = [{
            "label": "chair", "distance_mm": 500, "urgency": "DANGER", "angle_deg": 0.0
        }] if danger else []
        warning_track = [{
            "label": "table", "distance_mm": 1500, "urgency": "WARNING", "angle_deg": 10.0
        }] if warning else []
        return {
            "tracks": danger_track + warning_track,
            "danger": danger_track, "warning": warning_track, "safe": []
        }

    print("Test 1: Initial mode")
    bundle = make_bundle()
    obs = make_obstacle_result()
    mode = sm.update(bundle, obs)
    print(f"  Mode: {mode} (expected: NAVIGATION)")
    assert mode == MODE.NAVIGATION
    print("  PASSED\n")

    print("Test 2: OCR trigger — 1 frame (should NOT switch yet)")
    mode = sm.update(make_bundle(), obs, ocr_text_distance=600.0)
    print(f"  Mode: {mode} (expected: NAVIGATION — need 2 frames)")
    assert mode == MODE.NAVIGATION
    print("  PASSED\n")

    print("Test 3: OCR trigger — 2 consecutive frames + dwell time passed")
    sm._mode_entered_at = time.time() - 2.0
    mode = sm.update(make_bundle(), obs, ocr_text_distance=600.0)
    print(f"  Mode: {mode} (expected: OCR)")
    assert mode == MODE.OCR
    print("  PASSED\n")

    print("Test 4: Emergency override — DANGER obstacle while in OCR mode")
    sm._mode_entered_at = time.time() - 5.0
    mode = sm.update(make_bundle(), make_obstacle_result(danger=True), ocr_text_distance=600.0)
    print(f"  Mode: {mode} (expected: NAVIGATION — emergency override)")
    assert mode == MODE.NAVIGATION
    print("  PASSED\n")

    print("Test 5: Fast motion — OCR should NOT trigger")
    sm._mode_entered_at = time.time() - 2.0
    sm._ocr_frames = sm._face_frames = sm._banknote_frames = sm._interaction_frames = 0
    if sm._current_mode != MODE.NAVIGATION:
        sm.force_mode(MODE.NAVIGATION, reason="test reset")
        sm._mode_entered_at = time.time() - 2.0

    walking_bundle = make_bundle(accel_x=8.0, accel_y=0.0, accel_z=9.81)
    for _ in range(5):
        mode = sm.update(walking_bundle, obs, ocr_text_distance=600.0)

    print(f"  Motion level: {sm._motion_level:.2f} m/s^2")
    print(f"  Mode: {mode} (expected: NAVIGATION — user moving too fast)")
    assert mode == MODE.NAVIGATION
    print("  PASSED\n")

    print("Test 6: Face ID trigger — stable face detection")
    sm._mode_entered_at = time.time() - 2.0
    for _ in range(3):
        mode = sm.update(make_bundle(), obs, face_confidence=0.92)
    print(f"  Final mode: {mode} (expected: FACE_ID)")
    assert mode == MODE.FACE_ID
    print("  PASSED\n")

    print("Test 7: Mode transition history")
    history = sm.get_history()
    print(f"  Last {len(history)} transitions:")
    for t in history:
        print(f"    {t.from_mode:12s} -> {t.to_mode:12s}  ({t.reason})")
    print("  PASSED\n")

    print("Test 8: Stats")
    for key, val in sm.get_stats().items():
        print(f"  {key}: {val}")
    print("  PASSED\n")

    print("=== All tests passed ===")
