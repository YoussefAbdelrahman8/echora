import face_recognition as fr
import numpy as np
import cv2
import time
from collections import deque
from typing import Optional, Dict, List, Tuple

from src.core.config import settings
from src.core.utils import logger, get_timestamp_ms
from src.storage.database import get_db


class FaceRecognizer:
    """Detects and identifies faces for ECHORA."""

    def __init__(self):
        self._known_names: List[str] = []
        self._known_embeddings: List[np.ndarray] = []
        self._ready: bool = False

        self._name_history: deque = deque(maxlen=settings.FACE_STABILITY_FRAMES)
        self._last_spoken: str = ""

        self._detect_count: int = 0
        self._identify_count: int = 0
        self._success_count: int = 0
        self._avg_detect_ms: float = 0.0

        logger.info("FaceRecognizer created. Call load_model() to start.")

    def load_model(self):
        logger.info("Loading face recognition model...")
        try:
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            fr.face_encodings(test_image)
            logger.info("face_recognition library loaded successfully.")
        except Exception as e:
            logger.error(f"face_recognition failed to load: {e}")
            raise

        self._load_embeddings_from_db()
        self._ready = True

        if self._known_names:
            logger.info(f"FaceRecognizer ready. Known faces: {', '.join(self._known_names)}")
        else:
            logger.info("FaceRecognizer ready. No faces registered yet.")

    def _load_embeddings_from_db(self):
        db = get_db()
        if db is None:
            logger.warning("Database not initialised. Call init_database() before load_model().")
            self._known_names = []
            self._known_embeddings = []
            return

        persons = db.get_all_persons()
        self._known_names = [p["name"] for p in persons]
        self._known_embeddings = [p["embedding"] for p in persons]
        logger.info(f"Loaded {len(self._known_names)} face embeddings from database.")

    def reload_embeddings(self):
        logger.info("Reloading face embeddings from database...")
        self._load_embeddings_from_db()
        logger.info(f"Embeddings reloaded. Known faces: {len(self._known_names)}")

    def register_face(self, name: str, rgb_frame: np.ndarray) -> bool:
        logger.info(f"Registering face for: {name}")

        face_locations = fr.face_locations(rgb_frame, model="hog")
        if not face_locations:
            logger.warning(f"No face detected in frame for {name}.")
            return False

        largest_face = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
        embeddings = fr.face_encodings(rgb_frame, known_face_locations=[largest_face], num_jitters=10)

        if not embeddings:
            logger.error(f"Failed to compute embedding for {name}.")
            return False

        embedding = embeddings[0]
        db = get_db()
        if db is None:
            logger.error("Database not available — cannot register face.")
            return False

        success = db.add_person(name, embedding)
        if success:
            self.reload_embeddings()
            db.log_event("face_registered", {"name": name})
            logger.info(f"Face registered: {name}")

        return success

    def detect_face(self, rgb_frame: np.ndarray) -> float:
        if not self._ready:
            return 0.0

        self._detect_count += 1
        start_ms = get_timestamp_ms()

        try:
            h, w = rgb_frame.shape[:2]
            small = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = fr.face_locations(rgb_small, model="hog")

            elapsed = get_timestamp_ms() - start_ms
            self._avg_detect_ms = (self._avg_detect_ms * 0.9 + elapsed * 0.1)

            if not locations:
                return 0.0

            sh, sw = rgb_small.shape[:2]
            frame_area = sh * sw

            largest = max(locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
            top, right, bottom, left = largest
            face_area = (bottom - top) * (right - left)

            raw_confidence = (face_area / frame_area) * 10.0
            return float(min(raw_confidence, 1.0))

        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return 0.0

    def identify_face(self, rgb_frame: np.ndarray) -> Tuple[str, str]:
        if not self._ready:
            return "", ""

        self._identify_count += 1

        try:
            rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            locations = fr.face_locations(rgb, model="hog")

            if not locations:
                self._name_history.append("")
                return "", ""

            embeddings = fr.face_encodings(rgb, known_face_locations=locations, num_jitters=1)
            if not embeddings:
                self._name_history.append("")
                return "", ""

            if len(locations) > 1:
                largest_idx = max(
                    range(len(locations)),
                    key=lambda i: (locations[i][2] - locations[i][0]) * (locations[i][1] - locations[i][3])
                )
                query_embedding = embeddings[largest_idx]
            else:
                query_embedding = embeddings[0]

            if not self._known_embeddings:
                self._name_history.append("")
                return "", ""

            matches = fr.compare_faces(self._known_embeddings, query_embedding, tolerance=settings.FACE_RECOGNITION_TOLERANCE)
            distances = fr.face_distance(self._known_embeddings, query_embedding)

            best_idx = int(np.argmin(distances))
            best_distance = float(distances[best_idx])
            best_match = matches[best_idx]

            if not best_match:
                logger.debug(f"No match found. Best distance: {best_distance:.3f} (threshold: {settings.FACE_RECOGNITION_TOLERANCE})")
                self._name_history.append("")
                return "", ""

            identified_name = self._known_names[best_idx]
            logger.debug(f"Face matched: {identified_name} (distance={best_distance:.3f})")

            self._name_history.append(identified_name)

            if not self._is_stable() or identified_name == self._last_spoken:
                return "", ""

            self._last_spoken = identified_name
            self._success_count += 1

            db = get_db()
            if db:
                db.update_last_seen(identified_name)
                db.log_event("face_identified", {"name": identified_name, "distance": round(best_distance, 3)})

            logger.info(f"Face identified: {identified_name} (distance={best_distance:.3f})")
            return identified_name, ""

        except Exception as e:
            logger.error(f"Face identification error: {e}")
            return "", ""

    def _is_stable(self) -> bool:
        if len(self._name_history) < settings.FACE_STABILITY_FRAMES:
            return False
        unique = set(self._name_history)
        return len(unique) == 1 and list(unique)[0] != ""

    def draw_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = fr.face_locations(rgb, model="hog")

            for (top, right, bottom, left) in locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (180, 0, 255), 2)

            if self._last_spoken:
                label = f"ID: {self._last_spoken}"
            elif locations:
                label = "Identifying..."
            else:
                label = "No face detected"

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 0, 255), 2)
            cv2.putText(
                frame, f"Stability: {len(self._name_history)}/{settings.FACE_STABILITY_FRAMES}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 255), 1
            )

        except Exception as e:
            logger.error(f"Face overlay error: {e}")

        return frame

    def reset(self):
        self._name_history.clear()
        self._last_spoken = ""

    def get_stats(self) -> Dict:
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

_recogniser: Optional[FaceRecognizer] = None

def init_face_recognition():
    global _recogniser
    if _recogniser is not None:
        return
    _recogniser = FaceRecognizer()
    _recogniser.load_model()
    logger.info("Module-level face recogniser ready.")

def detect_face(rgb_frame: np.ndarray) -> float:
    if _recogniser is None:
        return 0.0
    return _recogniser.detect_face(rgb_frame)

def identify_face(rgb_frame: np.ndarray) -> Tuple[str, str]:
    if _recogniser is None:
        return "", ""
    return _recogniser.identify_face(rgb_frame)

def reset_face():
    if _recogniser is not None:
        _recogniser.reset()