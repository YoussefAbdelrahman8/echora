import time
import numpy as np
import cv2
import torch
from collections import deque, Counter
from typing import Dict, List, Optional, Tuple

from insightface.app import FaceAnalysis

from src.core.config import settings
from src.core.utils import logger, get_timestamp_ms
from src.storage.database import get_db

# ── Constants ──────────────────────────────────────────────────────────
EMBEDDING_SIZE        = 512    # buffalo_sc ArcFace output dimensions
RECOGNITION_THRESHOLD = settings.FACE_RECOGNITION_THRESHOLD  # cosine similarity
DET_SIZE_FULL         = (640, 640)   # detection input for identification
DET_SIZE_PROBE        = (320, 240)   # smaller input for the navigation probe


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

            logger.debug(
                f"Best match: {self._known_names[best_idx]} "
                f"sim={best_sim:.3f} threshold={RECOGNITION_THRESHOLD}"
            )

            if best_sim < RECOGNITION_THRESHOLD:
                self._name_history.append("unknown")
                stable = self._stable_result()
                if stable == "unknown" and self._last_spoken != "unknown":
                    self._last_spoken = "unknown"
                    return "unknown", ""
                return "", ""

            name = self._known_names[best_idx]
            self._name_history.append(name)

            stable = self._stable_result()
            if not stable or stable == self._last_spoken:
                return "", ""

            self._last_spoken  = stable
            self._success_count += 1

            db = get_db()
            if db:
                db.update_last_seen(stable)
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
        if not self._ready or self._app is None:
            logger.error("FaceRecognizer not ready.")
            return False

        logger.info(f"Registering face for: {name}")
        faces = self._app.get(frame)

        if not faces:
            logger.warning(f"No face detected in frame for {name}.")
            return False

        face      = max(faces, key=lambda f: f.det_score)
        embedding = face.embedding   # 512-d float32, L2-normalised

        db = get_db()
        if db is None:
            logger.error("Database not available.")
            return False

        success = db.add_person(name, embedding)
        if success:
            self.reload_embeddings()
            db.log_event("face_registered", {"name": name})
            logger.info(f"Face registered: {name}  det_score={face.det_score:.2f}")

        return success

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
        logger.info(f"HQ registration for '{name}' — loading buffalo_l...")
        logger.info("  First run will download ~500 MB of model weights automatically.")

        hq_app = None
        _loaded_hq = False
        try:
            hq_app = FaceAnalysis(name="buffalo_l", providers=_get_providers())
            hq_app.prepare(ctx_id=0, det_size=DET_SIZE_FULL)
            _loaded_hq = True
            logger.info("buffalo_l loaded for registration.")
        except Exception as e:
            logger.warning(f"buffalo_l unavailable ({e}) — falling back to buffalo_sc.")
            hq_app = self._app

        try:
            # Quality check on original frame
            faces = hq_app.get(frame)
            if not faces:
                return False, "no_face"

            best_face = max(faces, key=lambda f: f.det_score)
            if best_face.det_score < settings.FACE_REGISTRATION_MIN_SCORE:
                return False, f"low_quality:{best_face.det_score:.2f}"

            # Collect embeddings from all photometric variants
            embeddings = []
            for variant in self._generate_augmentations(frame):
                try:
                    aug_faces = hq_app.get(variant)
                    if not aug_faces:
                        continue
                    f = max(aug_faces, key=lambda f: f.det_score)
                    if f.det_score < 0.50:
                        continue
                    emb = f.embedding.astype(np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        embeddings.append(emb / norm)
                except Exception:
                    continue

            if not embeddings:
                return False, "embedding_failed"

            # Average and re-normalise into one final embedding
            avg_emb = np.mean(embeddings, axis=0).astype(np.float32)
            norm = np.linalg.norm(avg_emb)
            if norm > 0:
                avg_emb = avg_emb / norm

            db = get_db()
            if db is None:
                return False, "db_unavailable"

            success = db.add_person(name, avg_emb)
            if success:
                self.reload_embeddings()
                db.log_event("face_registered_hq", {
                    "name":       name,
                    "model":      "buffalo_l" if _loaded_hq else "buffalo_sc",
                    "variants":   len(embeddings),
                    "det_score":  round(float(best_face.det_score), 3),
                })
                logger.info(
                    f"HQ face registered: '{name}'  "
                    f"model={'buffalo_l' if _loaded_hq else 'buffalo_sc'}  "
                    f"variants={len(embeddings)}  det={best_face.det_score:.3f}"
                )
                return True, ""

            return False, "db_error"

        finally:
            if _loaded_hq and hq_app is not None:
                del hq_app
                logger.info("buffalo_l released.")

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
