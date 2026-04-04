import cv2
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path to import echora modules
sys.path.append(str(Path(__file__).parent.parent))

from src.hardware.camera import EchoraCamera
from src.core.config import HAPTIC_ROWS, HAPTIC_COLS
from src.perception.interaction_detection import ElectrodeGridBuilder, HapticBridge, InteractionDetector

if __name__ == "__main__":
    print("=== ECHORA interaction_detection.py self-test ===\n")

    print("Test 1: Electrode grid builder")
    builder = ElectrodeGridBuilder()

    grid_right = builder.build_guidance_grid(dx=100, dy=0)
    print(f"  Guidance RIGHT:\n{grid_right}")
    assert grid_right[0, 4] > 0 and grid_right[0, 5] > 0, "Right cols should be active"
    assert grid_right[0, 0] == 0 and grid_right[0, 1] == 0, "Left cols should be off"
    print("  PASSED\n")

    grid_up = builder.build_guidance_grid(dx=0, dy=-100)
    print(f"  Guidance UP:\n{grid_up}")
    assert grid_up[0, 0] > 0 and grid_up[1, 0] > 0, "Top rows should be active"
    assert grid_up[3, 0] == 0 and grid_up[4, 0] == 0, "Bottom rows should be off"
    print("  PASSED\n")

    grid_diag = builder.build_guidance_grid(dx=80, dy=-80)
    print(f"  Guidance UP-RIGHT:\n{grid_diag}")
    print("  PASSED\n")

    grid_success = builder.build_success_grid(pulse_count=0)
    print(f"  Success (even pulse):\n{grid_success}")
    assert np.all(grid_success == 1.0)
    print("  PASSED\n")

    grid_success_off = builder.build_success_grid(pulse_count=1)
    print(f"  Success (odd pulse):\n{grid_success_off}")
    assert np.all(grid_success_off == 0.0)
    print("  PASSED\n")


    print("Test 2: Edge grid from synthetic edge map")
    fake_edges = np.zeros((100, 120), dtype=np.uint8)
    fake_edges[:, 90:95] = 255
    edge_grid = builder.build_edge_grid(fake_edges)
    print(f"  Edge grid:\n{edge_grid}")
    assert edge_grid[:, -1].sum() >= edge_grid[:, 0].sum()
    print("  PASSED\n")


    print("Test 3: HapticBridge stub")
    bridge = HapticBridge()
    bridge.connect()
    test_grid = np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)
    test_grid[2, 3] = 1.0
    bridge.send(test_grid)
    bridge.send_all_on(0.5)
    bridge.send_all_off()
    assert bridge._send_count == 3
    print("  PASSED\n")


    print("Test 4: InteractionDetector with live camera")
    cam = EchoraCamera()
    detector = InteractionDetector()

    try:
        cam.init_pipeline()
        detector.load_model()
        print("  Camera and model ready. Press Q to stop.\n")
        frame_count = 0

        while True:
            bundle = cam.get_synced_bundle()
            if bundle is None:
                continue

            frame_count += 1
            rgb, depth = bundle["rgb"], bundle["depth"]
            
            result = detector.update(rgb, depth, detections=[])
            debug = detector.draw_debug_overlay(rgb.copy(), result)
            cv2.imshow("Interaction Detection Test", debug)

            if frame_count % 15 == 0:
                stats = detector.get_stats()
                print(f"  Frame {frame_count} | Phase: {stats['phase']:8s} | Target: {stats['target_label']}")

            if cv2.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    finally:
        cv2.destroyAllWindows()
        detector.release()
        cam.release()
        print(f"\n  Processed {frame_count} frames.")

    print("\n=== All tests complete ===")
