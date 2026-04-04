import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.hardware.camera import EchoraCamera
from src.perception.banknote import BanknoteDetector, DENOMINATION_MAP
from src.core.config import settings, MODE

if __name__ == "__main__":
    print("=== ECHORA banknote.py self-test ===\n")
    cam = EchoraCamera()
    detector = BanknoteDetector()

    try:
        print("Starting camera and loading model...")
        cam.init_pipeline()
        detector.load_model()

        if detector._stub_mode:
            print(
                "Running in STUB MODE — model file not found.\n"
                f"Expected: {settings.BANKNOTE_MODEL_PATH}\n"
                "Drop banknote_egp.pt into models/ folder and restart."
            )
        else:
            print("Model loaded. Class names from model:")
            for idx, name in detector._model.names.items():
                spoken = DENOMINATION_MAP.get(name, f"UNMAPPED: {name}")
                print(f"  Class {idx}: '{name}' → '{spoken}'")

        print("\nHold an Egyptian banknote in front of the camera.")
        print("Press Q to quit.\n")

        frame_count = 0
        while True:
            bundle = cam.get_synced_bundle()
            if bundle is None:
                continue

            frame_count += 1
            rgb = bundle["rgb"]
            depth = bundle["depth"]

            note_visible = detector.is_note_in_range(rgb, depth)
            denomination = ""
            if note_visible:
                denomination = detector.classify_denomination(rgb)

            if frame_count % 10 == 0:
                print(
                    f"Frame {frame_count:4d} | "
                    f"Note visible: {'YES' if note_visible else 'no ':3s} | "
                    f"Denomination: '{denomination}'"
                )

            debug = detector.draw_debug_overlay(rgb.copy())
            h, w = depth.shape
            center_depth = depth[h//2, w//2]
            cv2.putText(
                debug,
                f"Center depth: {center_depth}mm",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1
            )

            cv2.imshow("ECHORA Banknote Test", debug)
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cv2.destroyAllWindows()
        cam.release()
        stats = detector.get_stats()
        print(f"\nFinal stats: {stats}")
        print("\n=== Self-test complete ===")
