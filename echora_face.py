# =============================================================================
# echora_face.py — ECHORA Face Detection and Identification
# =============================================================================
# NOTE: Named echora_face.py (not face_recognition.py) to avoid conflicting
# with the 'face_recognition' pip library which has the same name.
#
# Two jobs:
#   Job 1: detect_face()    — fast presence check, runs every 5 frames
#   Job 2: identify_face()  — full ID, runs only in FACE_ID mode
#
# Fully offline. Uses 128-dimensional face embeddings stored in SQLite.
# Embeddings loaded from database at startup — instant recognition.
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

# The actual face_recognition library from pip.
# Named differently from our file to avoid the naming collision.
import face_recognition as fr

# numpy for embedding vector operations.
import numpy as np

# cv2 for frame preprocessing and debug overlay.
import cv2

# time for performance timing.
import time

# collections.deque for identification stability tracking.
from collections import deque

# Type hints.
from typing import Optional, Dict, List, Tuple

# Our modules.
from config import (
    FACE_CONFIDENCE_THRESHOLD,
    FACE_STABILITY_FRAMES,
    FACE_RECOGNITION_TOLERANCE,
)
from utils import logger, get_timestamp_ms
from database import get_db


# =============================================================================
# FACE RECOGNISER CLASS
# =============================================================================

class FaceRecognizer:
    """
    Detects and identifies faces for ECHORA.

    Loads all registered face embeddings from the database at startup.
    Compares each detected face against the loaded embeddings.

    Two-phase design:
      Phase 1 — detect_face(): fast presence check every 5 frames
      Phase 2 — identify_face(): full ID only in FACE_ID mode

    Usage:
        recogniser = FaceRecognizer()
        recogniser.load_model()

        # Every 5 frames — feeds state machine:
        confidence = recogniser.detect_face(rgb_frame)

        # In FACE_ID mode only:
        name, details = recogniser.identify_face(rgb_frame)
    """

    def __init__(self):
        """
        Creates the recogniser. Does NOT load embeddings yet.
        Call load_model() after creating this object.
        """

        # ── In-memory face database ────────────────────────────────────────────
        # Loaded from SQLite at startup. Kept in memory for fast comparison.
        # These two lists are always parallel:
        #   _known_names[i] is the name for _known_embeddings[i]
        self._known_names:      List[str]        = []
        self._known_embeddings: List[np.ndarray] = []

        # Whether the model is loaded and ready.
        self._ready: bool = False

        # ── Stability tracking ─────────────────────────────────────────────────
        # Same pattern as OCR and banknote — confirm same result across N frames.
        self._name_history: deque = deque(maxlen=FACE_STABILITY_FRAMES)

        # The last name we announced — prevents repeating the same person.
        self._last_spoken: str = ""

        # ── Performance tracking ───────────────────────────────────────────────
        self._detect_count:   int   = 0
        self._identify_count: int   = 0
        self._success_count:  int   = 0
        self._avg_detect_ms:  float = 0.0

        logger.info("FaceRecognizer created. Call load_model() to start.")


    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def load_model(self):
        """
        Loads the face recognition model and all stored face embeddings.

        Steps:
          1. Test the face_recognition library loads correctly
          2. Load all registered persons from the database
          3. Cache their embeddings in memory for fast comparison
        """

        logger.info("Loading face recognition model...")

        try:
            # ── Step 1: Test library ───────────────────────────────────────────
            # Force dlib model loading now — avoids first-frame lag spike.
            # face_encodings() on a blank image returns [] but loads the model.
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            fr.face_encodings(test_image)
            logger.info("face_recognition library loaded successfully.")

        except Exception as e:
            logger.error(f"face_recognition failed to load: {e}")
            raise

        # ── Step 2: Load embeddings from database ──────────────────────────────
        self._load_embeddings_from_db()

        self._ready = True

        if self._known_names:
            logger.info(
                f"FaceRecognizer ready. "
                f"Known faces: {', '.join(self._known_names)}"
            )
        else:
            logger.info(
                "FaceRecognizer ready. "
                "No faces registered yet — run register_face.py to add people."
            )


    def _load_embeddings_from_db(self):
        """
        Loads all registered face embeddings from the SQLite database
        into memory for fast comparison.

        Called once at startup. If new faces are registered while ECHORA
        is running, call reload_embeddings() to refresh.
        """

        # Get the shared database instance.
        db = get_db()

        if db is None:
            logger.warning(
                "Database not initialised. "
                "Call init_database() before load_model()."
            )
            self._known_names      = []
            self._known_embeddings = []
            return

        # get_all_persons() returns list of dicts with name + embedding.
        persons = db.get_all_persons()

        # Extract names and embeddings into parallel lists.
        self._known_names      = [p["name"]      for p in persons]
        self._known_embeddings = [p["embedding"] for p in persons]

        logger.info(
            f"Loaded {len(self._known_names)} face embeddings from database."
        )


    def reload_embeddings(self):
        """
        Reloads face embeddings from the database.

        Call this after register_face.py adds a new person while
        ECHORA is running — so the new face is immediately recognisable
        without restarting the system.
        """

        logger.info("Reloading face embeddings from database...")
        self._load_embeddings_from_db()
        logger.info(
            f"Embeddings reloaded. "
            f"Known faces: {len(self._known_names)}"
        )


    def register_face(
        self,
        name:      str,
        rgb_frame: np.ndarray
    ) -> bool:
        """
        Registers a new person's face from a camera frame.

        Computes the face embedding and saves it to the database.
        Immediately available for recognition without restart.

        Called by register_face.py — the helper registration script.

        Arguments:
            name:      person's name
            rgb_frame: RGB frame containing their face

        Returns:
            True if registration succeeded, False if no face found.
        """

        logger.info(f"Registering face for: {name}")

        # ── Detect face locations ──────────────────────────────────────────────
        # face_locations() returns list of (top, right, bottom, left) tuples.
        # model="hog" = fast CPU-based detection.
        face_locations = fr.face_locations(rgb_frame, model="hog")

        if not face_locations:
            logger.warning(f"No face detected in frame for {name}.")
            return False

        # ── Select the largest face ────────────────────────────────────────────
        # If multiple faces, pick the largest (closest to camera).
        largest_face = max(
            face_locations,
            key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3])
        )

        # ── Compute embedding ──────────────────────────────────────────────────
        # num_jitters=10 — average over 10 crops for a more stable embedding.
        # Only used during registration so slow is OK.
        embeddings = fr.face_encodings(
            rgb_frame,
            known_face_locations = [largest_face],
            num_jitters          = 10,
        )

        if not embeddings:
            logger.error(f"Failed to compute embedding for {name}.")
            return False

        embedding = embeddings[0]

        # ── Save to database ───────────────────────────────────────────────────
        db = get_db()
        if db is None:
            logger.error("Database not available — cannot register face.")
            return False

        success = db.add_person(name, embedding)

        if success:
            # Reload embeddings so this person is immediately recognisable.
            self.reload_embeddings()
            # Log the registration event.
            db.log_event("face_registered", {"name": name})
            logger.info(f"Face registered: {name}")

        return success


    # =========================================================================
    # FACE DETECTION — FAST, EVERY 5 FRAMES
    # =========================================================================

    def detect_face(self, rgb_frame: np.ndarray) -> float:
        """
        Fast face presence check — runs every 5 frames.

        Returns confidence 0.0-1.0 based on detected face size.
        Does NOT identify who the face belongs to.

        This feeds the state machine to decide when to switch to FACE_ID mode.

        Arguments:
            rgb_frame: numpy array (H, W, 3)

        Returns:
            Confidence 0.0-1.0. Above FACE_CONFIDENCE_THRESHOLD
            = face detected = state machine may switch to FACE_ID.
        """

        if not self._ready:
            return 0.0

        self._detect_count += 1
        start_ms = get_timestamp_ms()

        try:
            h, w = rgb_frame.shape[:2]

            # ── Resize to 25% for speed ────────────────────────────────────────
            # Face detection at full 1280×800 takes ~200ms on CPU.
            # At 25% (320×200) it takes ~20ms — fast enough for every 5 frames.
            small = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

            # face_recognition expects RGB — OpenCV gives BGR.
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # Detect face locations.
            locations = fr.face_locations(rgb_small, model="hog")

            # Update rolling average detection time.
            elapsed = get_timestamp_ms() - start_ms
            self._avg_detect_ms = (
                self._avg_detect_ms * 0.9 + elapsed * 0.1
            )

            if not locations:
                return 0.0

            # ── Confidence from face size ──────────────────────────────────────
            # Larger face = closer = more confident.
            sh, sw = rgb_small.shape[:2]
            frame_area = sh * sw

            largest = max(
                locations,
                key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3])
            )
            top, right, bottom, left = largest
            face_area = (bottom - top) * (right - left)

            # Scale: face at 5% of frame → raw 0.5, face at 10% → raw 1.0.
            # Multiply by 10 so a reasonably sized face gives 0.5-0.9.
            raw_confidence = (face_area / frame_area) * 10.0
            confidence     = float(min(raw_confidence, 1.0))

            return confidence

        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return 0.0


    # =========================================================================
    # FACE IDENTIFICATION — FULL, ONLY IN FACE_ID MODE
    # =========================================================================

    def identify_face(
        self,
        rgb_frame: np.ndarray
    ) -> Tuple[str, str]:

        """
        Full face identification — runs only in FACE_ID mode.

        Steps:
          1. Detect face locations at full resolution
          2. Compute 128-d embedding for detected face
          3. Compare against all stored embeddings
          4. Find the closest match below FACE_RECOGNITION_TOLERANCE
          5. Stability check — same name in N consecutive frames
          6. Return stable identified name

        Arguments:
            rgb_frame: numpy array (H, W, 3)

        Returns:
            ("Ahmed", "") if recognised
            ("",      "") if no face or not in database
        """

        if not self._ready:
            return "", ""

        self._identify_count += 1

        try:
            # ── Convert colour ─────────────────────────────────────────────────
            rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

            # ── Detect faces at full resolution ────────────────────────────────
            # Full resolution needed for accurate embedding computation.
            locations = fr.face_locations(rgb, model="hog")

            if not locations:
                self._name_history.append("")
                return "", ""

            # ── Compute embeddings ─────────────────────────────────────────────
            # num_jitters=1 for speed — we run this every frame in FACE_ID mode.
            embeddings = fr.face_encodings(
                rgb,
                known_face_locations = locations,
                num_jitters          = 1,
            )

            if not embeddings:
                self._name_history.append("")
                return "", ""

            # ── Compare against known faces ────────────────────────────────────
            # Use the largest detected face for identification.
            # If multiple people in frame, identify the most prominent one.
            if len(locations) > 1:
                # Find index of largest face.
                largest_idx = max(
                    range(len(locations)),
                    key=lambda i: (
                        (locations[i][2] - locations[i][0]) *
                        (locations[i][1] - locations[i][3])
                    )
                )
                query_embedding = embeddings[largest_idx]
            else:
                query_embedding = embeddings[0]

            # If nobody registered, return unknown.
            if not self._known_embeddings:
                self._name_history.append("")
                return "", ""

            # compare_faces() returns True/False list — one per known person.
            # tolerance: lower = stricter. 0.5 is the recommended default.
            matches = fr.compare_faces(
                self._known_embeddings,
                query_embedding,
                tolerance = FACE_RECOGNITION_TOLERANCE
            )

            # face_distance() returns distance to each known embedding.
            # Lower = more similar = better match.
            distances = fr.face_distance(
                self._known_embeddings,
                query_embedding
            )

            # Find the best (lowest distance) match.
            best_idx      = int(np.argmin(distances))
            best_distance = float(distances[best_idx])
            best_match    = matches[best_idx]

            if not best_match:
                # Closest known face still too different — unknown person.
                logger.debug(
                    f"No match found. "
                    f"Best distance: {best_distance:.3f} "
                    f"(threshold: {FACE_RECOGNITION_TOLERANCE})"
                )
                self._name_history.append("")
                return "", ""

            identified_name = self._known_names[best_idx]

            logger.debug(
                f"Face matched: {identified_name} "
                f"(distance={best_distance:.3f})"
            )

            # ── Stability check ────────────────────────────────────────────────
            # Same name must appear in FACE_STABILITY_FRAMES consecutive frames.
            self._name_history.append(identified_name)

            if not self._is_stable():
                return "", ""

            # ── Check for repeat announcement ──────────────────────────────────
            if identified_name == self._last_spoken:
                return "", ""

            # ── Stable new identification ──────────────────────────────────────
            self._last_spoken = identified_name
            self._success_count += 1

            # Update database — last seen time + seen count.
            db = get_db()
            if db:
                db.update_last_seen(identified_name)
                db.log_event(
                    "face_identified",
                    {
                        "name":       identified_name,
                        "distance":   round(best_distance, 3),
                    }
                )

            logger.info(
                f"Face identified: {identified_name} "
                f"(distance={best_distance:.3f})"
            )
            logger.info(
                f"Face distances: { {self._known_names[i]: round(float(distances[i]), 3) for i in range(len(distances))} }")

            return identified_name, ""

        except Exception as e:
            logger.error(f"Face identification error: {e}")
            return "", ""


    # =========================================================================
    # STABILITY CHECKING
    # =========================================================================

    def _is_stable(self) -> bool:
        """
        Returns True if the same name appeared in all recent frames.
        Same pattern as OCR and banknote stability checks.
        """

        if len(self._name_history) < FACE_STABILITY_FRAMES:
            return False

        history_list = list(self._name_history)
        unique       = set(history_list)

        return (
            len(unique) == 1
            and list(unique)[0] != ""
        )


    # =========================================================================
    # DEBUG OVERLAY
    # =========================================================================

    def draw_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draws face detection boxes and identification label on frame."""

        try:
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = fr.face_locations(rgb, model="hog")

            for (top, right, bottom, left) in locations:
                cv2.rectangle(
                    frame, (left, top), (right, bottom),
                    (180, 0, 255), 2
                )

            if self._last_spoken:
                label = f"ID: {self._last_spoken}"
            elif locations:
                label = "Identifying..."
            else:
                label = "No face detected"

            cv2.putText(
                frame, label,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (180, 0, 255), 2
            )

            cv2.putText(
                frame,
                f"Stability: {len(self._name_history)}/{FACE_STABILITY_FRAMES}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 0, 255), 1
            )

        except Exception as e:
            logger.error(f"Face overlay error: {e}")

        return frame


    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def reset(self):
        """Resets identification state when exiting FACE_ID mode."""
        self._name_history.clear()
        self._last_spoken = ""
        logger.debug("FaceRecognizer state reset.")


    def get_stats(self) -> Dict:
        """Returns diagnostic statistics."""
        return {
            "ready":          self._ready,
            "known_faces":    len(self._known_names),
            "known_names":    self._known_names,
            "detect_count":   self._detect_count,
            "identify_count": self._identify_count,
            "success_count":  self._success_count,
            "avg_detect_ms":  round(self._avg_detect_ms, 1),
            "last_spoken":    self._last_spoken,
        }


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_recogniser: Optional[FaceRecognizer] = None


def init_face_recognition():
    """
    Initialises the module-level face recogniser.
    Call once at startup from control_unit.py.
    """

    global _recogniser

    if _recogniser is not None:
        logger.debug("Face recogniser already initialised.")
        return

    _recogniser = FaceRecognizer()
    _recogniser.load_model()
    logger.info("Module-level face recogniser ready.")


def detect_face(rgb_frame: np.ndarray) -> float:
    """
    Module-level function — returns face confidence 0.0-1.0.
    Called every 5 frames by control_unit.py.
    """
    if _recogniser is None:
        return 0.0
    return _recogniser.detect_face(rgb_frame)


def identify_face(rgb_frame: np.ndarray) -> Tuple[str, str]:
    """
    Module-level function — returns (name, details) tuple.
    Called only in FACE_ID mode by control_unit.py.
    """
    if _recogniser is None:
        return "", ""
    return _recogniser.identify_face(rgb_frame)



def reset_face():
    """Module-level reset — called when exiting FACE_ID mode."""
    if _recogniser is not None:
        _recogniser.reset()


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":

    print("=== ECHORA echora_face.py self-test ===\n")

    from database import init_database
    from camera import EchoraCamera

    # Initialise database first — face recogniser needs it.
    print("Initialising database...")
    init_database()

    cam        = EchoraCamera()
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

            # Run identification.
            name, _ = recogniser.identify_face(rgb)
            if name:
                print(f"  IDENTIFIED: {name}")

            # Draw debug overlay.
            debug = recogniser.draw_debug_overlay(rgb.copy())
            cv2.imshow("ECHORA Face Recognition Test", debug)

            key = cv2.waitKey(1)

            if key == ord('q') or key == ord('Q'):
                break

            if key == ord('r') or key == ord('R'):
                # Register a new face from terminal input.
                cv2.destroyAllWindows()
                person_name = input("\nEnter person's name: ").strip()
                if person_name:
                    bundle = cam.get_synced_bundle()
                    if bundle:
                        success = recogniser.register_face(
                            person_name, bundle["rgb"]
                        )
                        print(
                            f"Registration {'succeeded' if success else 'failed'}."
                        )
                    else:
                        print("No frame available.")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        cv2.destroyAllWindows()
        cam.release()
        print(f"\nStats: {recogniser.get_stats()}")
        print("\n=== Self-test complete ===")