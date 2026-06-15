
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.hardware.camera import EchoraCamera
from src.perception.echora_face import FaceRecognizer
from src.storage.database import init_database, get_db
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

    while True:
        bundle = cam.get_synced_bundle()
        if bundle is None:
            continue

        frame   = bundle["rgb"]
        display = frame.copy()

        # Use InsightFace (same model as identification) for live preview overlay
        try:
            faces = recogniser._app.get(cv2.resize(frame, (320, 240))) if recogniser._app else []
            h_scale = frame.shape[0] / 240
            w_scale = frame.shape[1] / 320

            for face in faces:
                x1, y1, x2, y2 = (
                    int(face.bbox[0] * w_scale), int(face.bbox[1] * h_scale),
                    int(face.bbox[2] * w_scale), int(face.bbox[3] * h_scale),
                )
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display, f"det:{face.det_score:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1,
                )

            if not faces:
                cv2.putText(
                    display, "No face detected",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 220), 2,
                )
        except Exception:
            pass

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
    init_database()
    db = get_db()

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
        db.release()
        print("\nDone.")

