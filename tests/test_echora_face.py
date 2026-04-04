import cv2
import sys
from pathlib import Path

# Add parent directory to path to import echora modules
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.database import init_database
from src.hardware.camera import EchoraCamera
from src.perception.echora_face import FaceRecognizer

if __name__ == "__main__":
    print("=== ECHORA echora_face.py self-test ===\n")

    print("Initialising database...")
    init_database()

    cam = EchoraCamera()
    recogniser = FaceRecognizer()

    try:
        print("Starting camera...")
        cam.init_pipeline()
        recogniser.load_model()

        print(f"\nKnown faces: {recogniser.get_stats()['known_names']}")
        print("\nInstructions:")
        print("  R = register a new face (type name in terminal)")
        print("  Q = quit\n")

        while True:
            bundle = cam.get_synced_bundle()
            if bundle is None:
                continue

            rgb = bundle["rgb"]

            name, _ = recogniser.identify_face(rgb)
            if name:
                print(f"  IDENTIFIED: {name}")

            debug = recogniser.draw_debug_overlay(rgb.copy())
            cv2.imshow("ECHORA Face Recognition Test", debug)

            key = cv2.waitKey(1)

            if key == ord('q') or key == ord('Q'):
                break

            if key == ord('r') or key == ord('R'):
                cv2.destroyAllWindows()
                person_name = input("\nEnter person's name: ").strip()
                if person_name:
                    bundle = cam.get_synced_bundle()
                    if bundle:
                        success = recogniser.register_face(person_name, bundle["rgb"])
                        print(f"Registration {'succeeded' if success else 'failed'}.")
                    else:
                        print("No frame available.")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        cv2.destroyAllWindows()
        cam.release()
        print(f"\nStats: {recogniser.get_stats()}")
        print("\n=== Self-test complete ===")
