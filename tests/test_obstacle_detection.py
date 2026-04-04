import cv2
import sys
from pathlib import Path

# Add parent directory to path to import echora modules
sys.path.append(str(Path(__file__).parent.parent))

from src.hardware.camera import EchoraCamera
from src.perception.obstacle_detection import ObstacleDetector

if __name__ == "__main__":
    print("=== ECHORA obstacle_detection.py self-test ===\n")

    # Initialise camera 
    cam = EchoraCamera()
    cam.init_pipeline()

    # Initialise detector
    detector = ObstacleDetector()
    detector.load_model()

    print("Camera and detector ready. Running for 100 frames...\n")
    print(f"{'Frame':>6}  {'Tracks':>6}  {'Danger':>6}  {'Warning':>7}  {'Most urgent':<40}")
    print("-" * 80)

    frame_count = 0

    try:
        while cam.pipeline.isRunning() and frame_count < 100:
            bundle = cam.get_synced_bundle()
            if bundle is None:
                continue

            frame_count += 1
            result = detector.update(bundle)
            most_urgent = detector.get_most_urgent_obstacle()
            urgent_str  = (
                f"{most_urgent['label']} "
                f"{most_urgent['distance_mm']:.0f}mm "
                f"[{most_urgent['urgency']}]"
                if most_urgent else "none"
            )

            print(
                f"{result['frame_count']:>6}  "
                f"{len(result['tracks']):>6}  "
                f"{len(result['danger']):>6}  "
                f"{len(result['warning']):>7}  "
                f"{urgent_str:<40}"
            )

            if result["scene_desc"]:
                print(f"  VLM: {result['scene_desc']}")

            frame = bundle["rgb"].copy()
            for track in result["tracks"]:
                x1, y1, x2, y2 = track["bbox"]
                colour = {
                    "DANGER":  (0, 0, 220),
                    "WARNING": (0, 165, 255),
                    "SAFE":    (0, 200, 80),
                }.get(track["urgency"], (120, 120, 120))

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    frame,
                    f"{track['label']} {track['distance_mm']:.0f}mm",
                    (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, colour, 1
                )

            cv2.imshow("ECHORA obstacle detection", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        cv2.destroyAllWindows()
        cam.release()

        print("\n--- Final stats ---")
        stats = detector.get_stats()
        for key, val in stats.items():
            print(f"  {key}: {val}")

        print("\n=== Self-test complete ===")
