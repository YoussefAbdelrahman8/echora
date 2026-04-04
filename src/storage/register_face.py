
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.hardware.camera import EchoraCamera
from src.perception.echora_face import FaceRecognizer
from src.storage.database import init_database
from src.core.utils import logger

def register_person(name: str, cam: EchoraCamera, recogniser: FaceRecognizer):
    """
    Interactive face registration for one person.

    Shows a live camera preview with face detection overlay.
    User presses SPACE to capture when the face is clearly visible.
    Saves the embedding to the database on success.

    Arguments:
        name:       the person's name
        cam:        running EchoraCamera instance
        recogniser: loaded FaceRecognizer instance

    Returns:
        True if registered successfully, False if cancelled or failed.
    """

    print(f"\nRegistering: {name}")
    print("─" * 40)
    print("  Position face clearly in front of camera.")
    print("  Make sure lighting is good.")
    print("  Press SPACE to capture.")
    print("  Press Q to cancel.\n")

    try:
        import face_recognition as fr
        has_fr = True
    except ImportError:
        has_fr = False

    while True:
        bundle = cam.get_synced_bundle()
        if bundle is None:
            continue

        frame = bundle["rgb"]

        display = frame.copy()

        if has_fr:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            import cv2 as _cv2
            small_rgb  = _cv2.cvtColor(small, _cv2.COLOR_BGR2RGB)
            locations  = fr.face_locations(small_rgb, model="hog")

            h, w = frame.shape[:2]

            for (top, right, bottom, left) in locations:
                top    = top    * 4
                right  = right  * 4
                bottom = bottom * 4
                left   = left   * 4

                cv2.rectangle(
                    display,
                    (left, top), (right, bottom),
                    (0, 255, 0), 2
                )

                cv2.putText(
                    display, "Face detected",
                    (left, top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

            if not locations:
                cv2.putText(
                    display, "No face detected",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 220), 2
                )

        cv2.rectangle(display, (0, 0), (display.shape[1], 42), (0, 0, 0), -1)

        cv2.putText(
            display,
            f"Registering: {name}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1
        )

        cv2.putText(
            display,
            "SPACE = capture    Q = cancel",
            (8, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (180, 180, 180), 1
        )

        cv2.imshow("ECHORA — Face Registration", display)

        key = cv2.waitKey(1)

        if key == ord('q') or key == ord('Q'):
            print("  Registration cancelled.")
            return False

        if key == ord(' '):

            print("  Capturing face...")

            success = recogniser.register_face(name, frame)

            if success:
                success_frame = frame.copy()

                cv2.rectangle(
                    success_frame,
                    (0, 0),
                    (success_frame.shape[1], success_frame.shape[0]),
                    (0, 0, 0), -1
                )

                cv2.putText(
                    success_frame,
                    f"SAVED: {name}",
                    (
                        success_frame.shape[1] // 2 - 120,
                        success_frame.shape[0] // 2 - 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 220, 0), 3
                )

                cv2.putText(
                    success_frame,
                    "Face registered successfully.",
                    (
                        success_frame.shape[1] // 2 - 170,
                        success_frame.shape[0] // 2 + 30
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (180, 180, 180), 1
                )

                cv2.imshow("ECHORA — Face Registration", success_frame)
                cv2.waitKey(2000)

                print(f"  SUCCESS — {name} registered.")
                return True

            else:
                fail_frame = frame.copy()

                cv2.rectangle(
                    fail_frame,
                    (0, fail_frame.shape[0] // 2 - 40),
                    (fail_frame.shape[1], fail_frame.shape[0] // 2 + 40),
                    (0, 0, 0), -1
                )

                cv2.putText(
                    fail_frame,
                    "NO FACE DETECTED — try again",
                    (
                        fail_frame.shape[1] // 2 - 210,
                        fail_frame.shape[0] // 2 + 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 220), 2
                )

                cv2.imshow("ECHORA — Face Registration", fail_frame)
                cv2.waitKey(1500)

                print("  No face detected. Please try again.")

def main():

    print("=" * 50)
    print("  ECHORA Face Registration Tool")
    print("=" * 50)
    print()

    print("Initialising database...")
    db = init_database()

    existing = db.get_all_persons()
    if existing:
        print(f"Currently registered: {', '.join(p['name'] for p in existing)}")
    else:
        print("No faces registered yet.")

    print()

    name = input("Enter the person's name (or Q to quit): ").strip()

    if not name or name.lower() == 'q':
        print("Exiting.")
        return

    name = name.title()

    existing_person = db.get_person_by_name(name)
    if existing_person:
        confirm = input(
            f"  '{name}' is already registered "
            f"(seen {existing_person['seen_count']} times). "
            f"Re-register? (y/n): "
        ).strip().lower()

        if confirm != 'y':
            print("Cancelled.")
            return

    print("\nStarting camera...")
    cam        = EchoraCamera()
    recogniser = FaceRecognizer()

    try:
        cam.init_pipeline()
        recogniser.load_model()
        print("Camera ready.\n")

        success = register_person(name, cam, recogniser)

        if success:
            all_persons = db.get_all_persons()
            print(
                f"\nDatabase now contains "
                f"{len(all_persons)} registered face(s):"
            )
            for p in all_persons:
                last = p['last_seen'] or "never"
                print(
                    f"  {p['name']:20s} — "
                    f"registered: {p['added_at'][:10]} — "
                    f"last seen: {last}"
                )

            print()
            another = input(
                "Register another person? (y/n): "
            ).strip().lower()

            if another == 'y':
                cv2.destroyAllWindows()
                new_name = input(
                    "Enter next person's name: "
                ).strip().title()

                if new_name:
                    register_person(new_name, cam, recogniser)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        cv2.destroyAllWindows()
        cam.release()
        db.close()
        print("\nDone.")

