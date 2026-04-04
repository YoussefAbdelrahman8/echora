import sys
from pathlib import Path

# Add parent directory to path to import echora modules
sys.path.append(str(Path(__file__).parent.parent))

from src.perception.kalman_tracker import KalmanTracker

if __name__ == "__main__":
    print("=== ECHORA kalman_tracker.py self-test ===\n")
    tracker = KalmanTracker(frame_width=1280)

    simulated_frames = [
        [
            {"label": "person", "bbox": (100, 100, 200, 300), "confidence": 0.91, "distance_mm": 1500},
            {"label": "chair",  "bbox": (500, 200, 650, 400), "confidence": 0.85, "distance_mm": 2200},
        ],
        [
            {"label": "person", "bbox": (115, 100, 215, 300), "confidence": 0.89, "distance_mm": 1480},
            {"label": "chair",  "bbox": (502, 200, 652, 400), "confidence": 0.86, "distance_mm": 2200},
        ],
        [
            {"label": "chair",  "bbox": (501, 200, 651, 400), "confidence": 0.84, "distance_mm": 2200},
        ],
        [
            {"label": "person", "bbox": (140, 100, 240, 300), "confidence": 0.92, "distance_mm": 750},
            {"label": "chair",  "bbox": (500, 200, 650, 400), "confidence": 0.87, "distance_mm": 2200},
        ],
        [
            {"label": "person", "bbox": (155, 100, 255, 300), "confidence": 0.90, "distance_mm": 700},
            {"label": "chair",  "bbox": (500, 200, 650, 400), "confidence": 0.85, "distance_mm": 2200},
        ],
    ]

    for frame_num, detections in enumerate(simulated_frames, start=1):
        print(f"--- Frame {frame_num} ---")
        print(f"  YOLO detections: {len(detections)}")

        confirmed_tracks = tracker.update(detections)

        if confirmed_tracks:
            for t in confirmed_tracks:
                print(f"  CONFIRMED: [{t['id']}] {t['label']:8s} | "
                      f"dist={t['distance_mm']:5.0f}mm | "
                      f"angle={t['angle_deg']:+.1f}deg | "
                      f"urgency={t['urgency']:7s} | "
                      f"hits={t['hits']} missed={t['missed']}")
        else:
            print("  No confirmed tracks yet.")

        stats = tracker.get_stats()
        print(f"  Stats: {stats}\n")

    print("--- Most urgent track ---")
    most_urgent = tracker.get_most_urgent()
    if most_urgent:
        print(f"  -> {most_urgent['label']} at {most_urgent['distance_mm']}mm "
              f"[{most_urgent['urgency']}]")
    else:
        print("  -> No confirmed tracks.")

    print("\n--- Testing reset ---")
    tracker.reset()
    stats = tracker.get_stats()
    print(f"  After reset: {stats}")

    print("\n=== All tests complete ===")
