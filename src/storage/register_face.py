
import cv2
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.hardware.camera import EchoraCamera
from src.perception.echora_face import FaceRecognizer
from src.storage.database import init_database, get_db
from src.core.config import settings
from src.core.utils import logger


# ── Overlay helper ─────────────────────────────────────────────────────────────

def _show_message(
    frame,
    line1: str,
    line2: str = "",
    error: bool = False,
    delay_ms: int = 2000,
):
    overlay = frame.copy()
    h, w = overlay.shape[:2]
    mid_y = h // 2
    cv2.rectangle(overlay, (0, mid_y - 60), (w, mid_y + 60), (0, 0, 0), -1)
    color1 = (0, 0, 210) if error else (0, 210, 0)
    (tw, _), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(overlay, line1, ((w - tw) // 2, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color1, 2)
    if line2:
        (tw2, _), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
        cv2.putText(overlay, line2, ((w - tw2) // 2, mid_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 180), 1)
    cv2.imshow("ECHORA — Face Registration", overlay)
    cv2.waitKey(delay_ms)


# ── Main registration flow ─────────────────────────────────────────────────────

def register_person(name: str, cam: EchoraCamera, recogniser: FaceRecognizer) -> bool:
    """
    Interactive face registration with best-frame capture and quality gate.

    Flow:
      1. Live preview with colour-coded quality indicator (green/orange/red).
      2. SPACE starts a FACE_REGISTRATION_CAPTURE_SEC silent window.
      3. The frame with the highest det_score in that window is selected.
      4. Quality gate: if best det_score < FACE_REGISTRATION_MIN_SCORE → rejected
         with on-screen and printed guidance.
      5. register_face_hq() is called with the best frame:
         - loads buffalo_l (GPU-accelerated)
         - generates 5 photometric augmentations
         - averages ArcFace embeddings → one high-quality stored embedding
         - releases buffalo_l after registration.

    Returns True if registered successfully, False if cancelled or failed.
    """
    MIN_SCORE   = settings.FACE_REGISTRATION_MIN_SCORE
    CAPTURE_SEC = settings.FACE_REGISTRATION_CAPTURE_SEC

    print(f"\nRegistering: {name}")
    print("─" * 40)
    print(f"  GREEN border = good quality (score ≥ {MIN_SCORE:.0%}).")
    print("  Press SPACE when the face is clearly visible.")
    print("  Press Q to cancel.\n")

    # ── State ──────────────────────────────────────────────────────────────────
    capturing     = False
    capture_start = 0.0
    best_frame    = None
    best_score    = 0.0
    cancelled     = False

    while True:
        bundle = cam.get_synced_bundle()
        if bundle is None:
            continue

        frame   = bundle["rgb"]
        display = frame.copy()
        current_score = 0.0

        # ── Live face detection overlay ────────────────────────────────────────
        try:
            small = cv2.resize(frame, (320, 240))
            faces = recogniser._app.get(small) if recogniser._app else []
            h_s = frame.shape[0] / 240
            w_s = frame.shape[1] / 320

            if faces:
                current_score = max(f.det_score for f in faces)
                color = (
                    (0, 210,   0) if current_score >= MIN_SCORE  else
                    (0, 165, 255) if current_score >= 0.50       else
                    (0,   0, 210)
                )
                for face in faces:
                    x1 = int(face.bbox[0] * w_s)
                    y1 = int(face.bbox[1] * h_s)
                    x2 = int(face.bbox[2] * w_s)
                    y2 = int(face.bbox[3] * h_s)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        display, f"{face.det_score:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
                    )
        except Exception:
            pass

        # ── Capture-window state ───────────────────────────────────────────────
        if capturing:
            elapsed   = time.time() - capture_start
            remaining = max(0.0, CAPTURE_SEC - elapsed)

            if current_score > best_score:
                best_score = current_score
                best_frame = frame.copy()

            cv2.putText(
                display, f"Capturing best frame...  {remaining:.1f}s",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2,
            )

            if elapsed >= CAPTURE_SEC:
                break

        else:
            # ── Idle quality hint ──────────────────────────────────────────────
            if current_score >= MIN_SCORE:
                hint, hc = "GOOD — Press SPACE to register",      (0, 210,   0)
            elif current_score >= 0.50:
                hint, hc = "Move closer or improve lighting",     (0, 165, 255)
            elif current_score > 0.0:
                hint, hc = "Face too far or unclear",             (0,   0, 210)
            else:
                hint, hc = "No face detected",                    (100, 100, 100)

            cv2.putText(display, hint, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hc, 2)

        # ── Header bar ─────────────────────────────────────────────────────────
        cv2.rectangle(display, (0, 0), (display.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(display, f"Registering: {name}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "SPACE = capture    Q = cancel", (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("ECHORA — Face Registration", display)
        key = cv2.waitKey(1)

        if key in (ord('q'), ord('Q')):
            print("  Registration cancelled.")
            cancelled = True
            break

        if key == ord(' ') and not capturing:
            print(f"  {CAPTURE_SEC:.0f}s capture window started — hold still...")
            capturing     = True
            capture_start = time.time()
            best_frame    = frame.copy()
            best_score    = current_score

    if cancelled:
        return False

    # ── Post-capture quality gate ──────────────────────────────────────────────
    if best_frame is None or best_score < MIN_SCORE:
        if best_score > 0:
            msg = (f"Best score was {best_score:.2f} "
                   f"(minimum required: {MIN_SCORE:.2f}).")
        else:
            msg = "No face detected during the capture window."

        print(f"  Registration failed — {msg}")
        print("  Tips: better lighting / face the camera directly / move closer.")

        _show_message(
            best_frame if best_frame is not None else frame,
            "QUALITY TOO LOW — try again",
            "Better lighting  /  move closer  /  face camera",
            error=True,
            delay_ms=2500,
        )
        return False

    print(f"  Best frame score: {best_score:.3f}. Running HQ registration...")

    # ── HQ registration ────────────────────────────────────────────────────────
    success, reason = recogniser.register_face_hq(name, best_frame)

    if success:
        _show_message(
            best_frame,
            f"SAVED: {name}",
            f"Quality: {best_score:.2f}  —  Face registered successfully.",
            delay_ms=2000,
        )
        print(f"  SUCCESS — {name} registered (det_score={best_score:.3f})")
        return True

    reason_text = {
        "no_face":          "No face found in the captured frame.",
        "low_quality":      f"Score too low ({best_score:.2f}).",
        "embedding_failed": "Could not generate face embedding.",
        "db_unavailable":   "Database not available.",
        "db_error":         "Failed to save to database.",
    }.get(reason.split(":")[0], reason)

    _show_message(
        best_frame,
        f"FAILED: {reason_text}",
        "Please try again.",
        error=True,
        delay_ms=2500,
    )
    print(f"  FAILED — {reason_text}")
    return False


# ── CLI entry point ────────────────────────────────────────────────────────────

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
