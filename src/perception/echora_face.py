import time
import numpy as np
import cv2
import torch
from collections import deque, Counter
from typing import Dict, List, Optional, Sequence, Tuple

from insightface.app import FaceAnalysis

from src.core.config import settings
from src.core.utils import logger, get_timestamp_ms
from src.storage.database import get_db

# ── Constants ──────────────────────────────────────────────────────────
EMBEDDING_SIZE        = 512    # buffalo_sc ArcFace output dimensions
RECOGNITION_THRESHOLD = settings.FACE_RECOGNITION_THRESHOLD  # cosine similarity
DET_SIZE_FULL         = (640, 640)   # detection input for identification
DET_SIZE_PROBE        = (320, 240)   # smaller input for the navigation probe
FACE_ABSENCE_SEC      = 600.0        # 10 min absence before re-announcing a known person


def _get_providers() -> List[str]:
    if torch.cuda.is_available():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class FaceRecognizer:
    """
    Detects and identifies faces using InsightFace (RetinaFace + ArcFace).

    RetinaFace replaces HOG — handles angles, partial occlusion, poor lighting.
    ArcFace 512-d embeddings replace dlib 128-d — far more discriminative and
    robust across expressions, lighting changes, and partial views.

    Similarity metric: cosine similarity (dot product of L2-normalised vectors).
    Threshold: RECOGNITION_THRESHOLD (default 0.35) — raise for stricter matching.
    """

    def __init__(self):
        self._app: Optional[FaceAnalysis] = None
        self._ready: bool = False

        self._known_names:      List[str]        = []
        self._known_embeddings: List[np.ndarray] = []  # L2-normalised float32

        self._name_history: deque = deque(maxlen=settings.FACE_STABILITY_FRAMES)
        self._last_spoken:  str   = ""
        self._session_seen: Dict[str, float] = {}  # name → wall-clock time last announced

        self._detect_count:   int   = 0
        self._identify_count: int   = 0
        self._success_count:  int   = 0
        self._avg_detect_ms:  float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────

    def load_model(self):
        logger.info("Loading InsightFace buffalo_sc (RetinaFace + ArcFace)...")
        logger.info("  First run downloads ~200 MB of model weights automatically.")
        t0 = time.time()
        try:
            self._app = FaceAnalysis(
                name="buffalo_sc",
                providers=_get_providers(),
            )
            self._app.prepare(ctx_id=0, det_size=DET_SIZE_FULL)
            logger.info(f"InsightFace ready in {(time.time()-t0)*1000:.0f}ms.")
        except Exception as e:
            logger.error(f"InsightFace failed to load: {e}")
            raise

        self._load_embeddings_from_db()
        self._ready = True

        if self._known_names:
            logger.info(f"FaceRecognizer ready. Known: {', '.join(self._known_names)}")
        else:
            logger.info("FaceRecognizer ready. No faces registered yet.")

    def _load_embeddings_from_db(self):
        db = get_db()
        if db is None:
            logger.warning("Database not ready — no embeddings loaded.")
            self._known_names, self._known_embeddings = [], []
            return

        # get_all_persons() already filters out old dlib embeddings (wrong size)
        persons = db.get_all_persons()
        self._known_names      = [p["name"]      for p in persons]
        self._known_embeddings = [p["embedding"] for p in persons]
        logger.info(f"Loaded {len(self._known_names)} InsightFace embedding(s).")

    def reload_embeddings(self):
        logger.info("Reloading face embeddings from database...")
        self._load_embeddings_from_db()

    # ── Navigation probe ──────────────────────────────────────────────

    def detect_face(self, frame: np.ndarray) -> float:
        """
        Quick probe called every 5 frames from the navigation loop.
        Returns the detection confidence (0.0–1.0) of the best face found.
        Uses a smaller input size to stay fast on CPU.
        """
        if not self._ready or self._app is None:
            return 0.0

        self._detect_count += 1
        t0 = get_timestamp_ms()

        try:
            small = cv2.resize(frame, DET_SIZE_PROBE)
            faces = self._app.get(small)
            elapsed = get_timestamp_ms() - t0
            self._avg_detect_ms = self._avg_detect_ms * 0.9 + elapsed * 0.1

            if not faces:
                return 0.0
            return float(max(f.det_score for f in faces))

        except Exception as e:
            logger.error(f"Face detect error: {e}")
            return 0.0

    # ── Full identification ────────────────────────────────────────────

    def identify_face(self, frame: np.ndarray) -> Tuple[str, str]:
        """
        Full identification pass using the complete frame resolution.

        Returns:
            (name, details)  — confirmed known person; details from DB
            ("unknown", "")  — face detected and stable, but not in database
            ("", "")         — no face, or not yet stable
        """
        if not self._ready or self._app is None:
            return "", ""

        self._identify_count += 1

        try:
            faces = self._app.get(frame)

            if not faces:
                self._name_history.append("")
                return "", ""

            # Pick the face with the highest detection confidence
            face      = max(faces, key=lambda f: f.det_score)
            embedding = face.embedding   # 512-d, already L2-normalised by InsightFace

            if not self._known_embeddings:
                self._name_history.append("unknown")
                stable = self._stable_result()
                if stable == "unknown" and self._last_spoken != "unknown":
                    self._last_spoken = "unknown"
                    return "unknown", ""
                return "", ""

            # Cosine similarity — dot product of L2-normalised vectors
            sims     = [float(np.dot(embedding, k)) for k in self._known_embeddings]
            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]
            second_sim = max((sim for i, sim in enumerate(sims) if i != best_idx), default=-1.0)

            logger.debug(
                f"Best match: {self._known_names[best_idx]} "
                f"sim={best_sim:.3f} threshold={RECOGNITION_THRESHOLD}"
            )

            if (best_sim < RECOGNITION_THRESHOLD
                    or best_sim - second_sim < settings.FACE_RECOGNITION_MARGIN):
                logger.debug(
                    f"Rejected face match: best={best_sim:.3f} second={second_sim:.3f} "
                    f"margin={best_sim - second_sim:.3f}"
                )
                self._name_history.append("unknown")
                stable = self._stable_result()
                if stable == "unknown" and self._last_spoken != "unknown":
                    self._last_spoken = "unknown"
                    return "unknown", ""
                return "", ""

            name = self._known_names[best_idx]
            self._name_history.append(name)

            stable = self._stable_result()
            if not stable:
                return "", ""

            now       = time.time()
            last_seen = self._session_seen.get(stable, 0.0)
            if now - last_seen < FACE_ABSENCE_SEC:
                # Person seen within the last 10 min — stay silent
                return "", ""

            # Genuine new sighting: first time this session or back after 10+ min
            self._session_seen[stable] = now
            self._last_spoken  = stable
            self._success_count += 1

            db = get_db()
            if db:
                db.update_last_seen(stable)   # increments seen_count only here
                db.log_event("face_identified", {
                    "name":       stable,
                    "similarity": round(best_sim, 3),
                })

            logger.info(f"Face identified: {stable} (sim={best_sim:.3f})")
            return stable, self._get_details(stable)

        except Exception as e:
            logger.error(f"Face identify error: {e}")
            return "", ""

    # ── Registration ──────────────────────────────────────────────────

    def register_face(self, name: str, frame: np.ndarray) -> bool:
        """Backward-compatible single-frame enrollment."""
        return self.register_face_samples(name, [frame])[0]

    def register_face_samples(
        self, name: str, frames: Sequence[np.ndarray]
    ) -> Tuple[bool, str]:
        """Enroll from several live frames in the same embedding space as recognition."""
        if not self._ready or self._app is None:
            return False, "not_ready"

        candidates = []
        for frame in frames:
            try:
                faces = self._app.get(frame)
                if not faces:
                    continue
                face = max(faces, key=lambda f: f.det_score)
                if face.det_score < settings.FACE_REGISTRATION_MIN_SCORE:
                    continue
                x1, y1, x2, y2 = np.asarray(face.bbox, dtype=int)
                h, w = frame.shape[:2]
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0 or min(crop.shape[:2]) < 60:
                    continue
                sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                if sharpness < 20.0:
                    continue
                emb = face.embedding.astype(np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    candidates.append((float(face.det_score) + min(sharpness, 200.0) / 1000.0, emb / norm))
            except Exception as exc:
                logger.debug(f"Skipping enrollment frame: {exc}")

        if not candidates:
            return False, "no_good_frames"
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[:settings.FACE_REGISTRATION_SAMPLE_COUNT]
        if len(selected) < min(settings.FACE_REGISTRATION_MIN_SAMPLES, len(frames)):
            return False, f"not_enough_good_frames:{len(selected)}"

        embedding = np.mean([item[1] for item in selected], axis=0).astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return False, "embedding_failed"
        embedding /= norm
        db = get_db()
        if db is None:
            return False, "db_unavailable"
        if not db.add_person(name, embedding):
            return False, "db_error"
        self.reload_embeddings()
        db.log_event("face_registered", {"name": name, "samples": len(selected), "model": "buffalo_sc"})
        logger.info(f"Face registered: '{name}' samples={len(selected)}")
        return True, ""

    def _generate_augmentations(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Returns 5 photometric variants of frame for embedding averaging.
        All variants are intensity-only — no geometry changes so face alignment is preserved.
        """
        variants = [frame]

        # Brightness +25
        variants.append(np.clip(frame.astype(np.int16) + 25, 0, 255).astype(np.uint8))

        # Brightness -25
        variants.append(np.clip(frame.astype(np.int16) - 25, 0, 255).astype(np.uint8))

        # CLAHE contrast enhancement
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            variants.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
        except Exception:
            pass

        # Unsharp mask sharpening
        try:
            blurred = cv2.GaussianBlur(frame, (0, 0), 3)
            variants.append(cv2.addWeighted(frame, 1.5, blurred, -0.5, 0))
        except Exception:
            pass

        return variants

    def register_face_hq(self, name: str, frame: np.ndarray) -> Tuple[bool, str]:
        """
        High-quality single-shot registration using buffalo_l + augmentation averaging.

        Loads buffalo_l temporarily (GPU-accelerated when available), generates
        photometric augmentations of the single enrollment frame, extracts an
        ArcFace embedding from each valid variant, then stores the L2-normalised
        average. buffalo_l is released after registration.

        Falls back to buffalo_sc (self._app) if buffalo_l cannot be loaded.

        Returns:
            (True, "")           on success
            (False, reason_str)  on failure — reason is one of:
                "no_face", "low_quality:<score>", "embedding_failed",
                "db_unavailable", "db_error"
        """
        # buffalo_l and buffalo_sc produce embeddings in different spaces.
        # Persisting a buffalo_l embedding while recognition uses buffalo_sc makes
        # cosine scores meaningless, so use the compatible path.
        return self.register_face_samples(name, [frame])

    # ── Debug overlay ─────────────────────────────────────────────────

    def draw_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        try:
            faces = self._app.get(frame) if self._app else []
            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 0, 255), 2)
                cv2.putText(
                    frame, f"{face.det_score:.2f}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 0, 255), 1,
                )

            label = (
                f"ID: {self._last_spoken}" if self._last_spoken
                else ("Identifying..." if faces else "No face detected")
            )
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 0, 255), 2)
            cv2.putText(
                frame,
                f"Stability: {len(self._name_history)}/{settings.FACE_STABILITY_FRAMES}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 255), 1,
            )
        except Exception as e:
            logger.error(f"Face overlay error: {e}")
        return frame

    # ── Helpers ───────────────────────────────────────────────────────

    def _stable_result(self) -> str:
        """
        Returns the most-common non-empty name in history when it appears
        in at least 2 of FACE_STABILITY_FRAMES frames.
        More robust than requiring all frames to match identically.
        """
        if len(self._name_history) < settings.FACE_STABILITY_FRAMES:
            return ""
        non_empty = [n for n in self._name_history if n]
        if not non_empty:
            return ""
        most_common, count = Counter(non_empty).most_common(1)[0]
        return most_common if count >= 2 else ""

    def _get_details(self, name: str) -> str:
        """Returns a human-readable string from the database for TTS."""
        try:
            db = get_db()
            if not db:
                return ""
            person = db.get_person_by_name(name)
            if not person:
                return ""
            count = person.get("seen_count", 0)
            if count == 0:
                return "First time meeting."
            if count == 1:
                return "Met once before."
            return f"Met {count} times before."
        except Exception:
            return ""

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
            "threshold":      RECOGNITION_THRESHOLD,
        }


# ── Module-level singleton ─────────────────────────────────────────────

_recogniser: Optional[FaceRecognizer] = None


def init_face_recognition():
    global _recogniser
    if _recogniser is not None:
        return
    _recogniser = FaceRecognizer()
    _recogniser.load_model()
    logger.info("Module-level face recogniser ready.")


def detect_face(frame: np.ndarray) -> float:
    return _recogniser.detect_face(frame) if _recogniser else 0.0


def identify_face(frame: np.ndarray) -> Tuple[str, str]:
    return _recogniser.identify_face(frame) if _recogniser else ("", "")


def reset_face():
    if _recogniser:
        _recogniser.reset()
