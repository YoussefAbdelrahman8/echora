# =============================================================================
# ocr.py — ECHORA Text Detection and Reading
# =============================================================================
# Single-pass pipeline (Arabic + English fix):
#   Instead of a two-stage detect-then-read approach, we now run a single
#   readtext() call on the full resized frame. This fixes Arabic because:
#     - Arabic detect() returns polygon coordinates, not simple [x,y,x,y] boxes
#     - The old coordinate parsing assumed Latin left-to-right layout
#     - Single-pass readtext() handles both scripts correctly internally
#
# Fixes applied vs previous version:
#   1. Single-pass readtext() replaces broken two-stage detect() + crop pipeline
#   2. OCR_LANGUAGE from config.py is used directly — no local OCR_LANGUAGES override
#   3. Crash in background thread now caught and _ocr_running always released
#   4. gpu= uses correct platform detection (MPS for Mac, CUDA for RTX, CPU fallback)
#   5. Arabic text is detected and returned correctly
#   6. Adaptive threshold removed from preprocessing — it destroys Arabic cursive
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import easyocr
import numpy as np
import cv2
import time
import torch
from collections import deque
from typing import List, Dict, Optional

from config import (
    OCR_CONFIDENCE_THRESHOLD,
    OCR_MIN_TEXT_HEIGHT_PX,
    OCR_MAX_CHARS,
    OCR_TRIGGER_DIST_MM,
    OCR_LANGUAGE,
    DEPTH_MIN_MM,
    DEPTH_MAX_MM,
)
from utils import logger, get_timestamp_ms, depth_in_region


# =============================================================================
# OCR CONFIGURATION
# =============================================================================

# Minimum EasyOCR confidence to accept a text block.
# 0.3 is more lenient than before — Arabic confidence scores tend to be lower.
MIN_WORD_CONFIDENCE = 0.3

# Minimum characters to bother with.
MIN_TEXT_LENGTH = 2

# Maximum characters in one spoken announcement.
MAX_SPEAK_CHARS = 150

# Consecutive frames with the same text before we speak.
# 3 = more stable, fewer false reads. Good for both Arabic and English.
TEXT_STABILITY_FRAMES = 3

# Resize factor before OCR. 0.5 = half resolution = much faster.
# Single-pass on 640x400 takes ~80ms on CPU, ~20ms on GPU.
OCR_SCALE_FACTOR = 0.5

# How many pixels of padding to add around each detected text bounding box.
# Helps EasyOCR read the full word without clipping.
BBOX_PADDING = 8


# =============================================================================
# GPU DETECTION
# =============================================================================

def _get_ocr_gpu() -> bool:
    """
    Returns True if a GPU is available for EasyOCR.

    EasyOCR on Mac does NOT support MPS — it only supports CUDA or CPU.
    Passing gpu=True on Mac with no CUDA causes a crash.
    This function returns the correct value for each platform.
    """

    # CUDA available = Windows/Linux with Nvidia GPU
    if torch.cuda.is_available():
        logger.info("OCR will use CUDA GPU.")
        return True

    # MPS = Apple Silicon — EasyOCR does NOT support MPS
    # Must use CPU on Mac even though MPS is available for YOLO
    if torch.backends.mps.is_available():
        logger.info("OCR will use CPU (MPS not supported by EasyOCR).")
        return False

    logger.info("OCR will use CPU.")
    return False


# =============================================================================
# OCR READER CLASS
# =============================================================================

class OCRReader:
    """
    Reads Arabic and English text from camera frames for ECHORA.

    Single-pass design:
      readtext() is called on the full resized frame directly.
      No separate detect() stage — fixes Arabic coordinate parsing crash.

    Results are filtered by confidence, stability-checked across N frames,
    and deduplicated before being passed to the audio system.
    """

    def __init__(self):

        self._reader:   Optional[easyocr.Reader] = None
        self._ready:    bool  = False

        # Stability tracking — same text must appear N times before speaking.
        self._text_history:    deque = deque(maxlen=TEXT_STABILITY_FRAMES)
        self._last_spoken_text: str  = ""

        # Distance cache — stores the last computed text distance.
        # get_text_distance() sets this; read_text() reads it.
        self._last_dist_mm:       float      = 0.0
        self._last_dist_frame_ts: float      = 0.0

        # Last raw detections — stored for debug overlay drawing.
        self._last_boxes: List[Dict] = []

        # Performance stats.
        self._read_count:   int   = 0
        self._success_count: int  = 0
        self._avg_ocr_ms:   float = 0.0

        logger.info("OCRReader created. Call load_model() to start.")


    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def load_model(self):
        """
        Loads EasyOCR with the languages configured in config.py.

        OCR_LANGUAGE = ["en"]         — English only
        OCR_LANGUAGE = ["en", "ar"]   — English + Arabic
        OCR_LANGUAGE = ["ar"]         — Arabic only

        GPU is used automatically when available (CUDA only — not MPS).
        """

        logger.info(f"Loading EasyOCR (languages: {OCR_LANGUAGE})...")

        start  = time.time()
        use_gpu = _get_ocr_gpu()

        try:
            self._reader = easyocr.Reader(
                OCR_LANGUAGE,
                gpu     = use_gpu,
                verbose = False
            )
            elapsed = round((time.time() - start) * 1000)
            logger.info(f"EasyOCR loaded in {elapsed}ms. GPU={use_gpu}")

        except Exception as e:
            logger.error(f"EasyOCR failed to load: {e}")
            raise

        self._ready = True
        logger.info("OCRReader ready.")


    # =========================================================================
    # CORE OCR — SINGLE PASS
    # =========================================================================

    def _run_ocr_on_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Runs a single readtext() call on the full (resized) frame.

        Returns a list of detection dicts:
        [
          {
            "text":       str,      clean text string
            "confidence": float,    0.0-1.0
            "bbox":       (x1,y1,x2,y2)  in ORIGINAL frame coordinates
          },
          ...
        ]

        WHY single-pass instead of detect() + crop():
          The old pipeline called detect() to get region boxes, then cropped
          each region and called readtext() on each crop separately.
          This broke Arabic because:
            1. detect() returns polygon points, not simple [x,y,x,y] boxes
            2. Our coordinate parser assumed [x_min, x_max, y_min, y_max]
            3. Arabic polygons have a different coordinate layout
            4. The resulting crops were zero-size or garbage → silent crash
          Single readtext() on the full frame lets EasyOCR handle all
          coordinate parsing internally — works correctly for both scripts.
        """

        if not self._ready or self._reader is None:
            return []

        try:
            h, w = frame.shape[:2]

            # Resize to OCR_SCALE_FACTOR for speed.
            # 1280x800 × 0.5 = 640x400 — fast enough for background thread.
            new_w = int(w * OCR_SCALE_FACTOR)
            new_h = int(h * OCR_SCALE_FACTOR)
            small = cv2.resize(frame, (new_w, new_h))

            # Light preprocessing — enhance contrast without destroying
            # Arabic cursive connections.
            # We do NOT use adaptive threshold — it breaks Arabic script by
            # cutting the connections between letters.
            preprocessed = self._preprocess(small)

            # Single readtext() call on the full frame.
            # detail=1 returns (bbox, text, confidence) tuples.
            # width_ths=0.7: merge nearby boxes — good for Arabic words
            # text_threshold=0.6: minimum score to confirm a text region
            # low_text=0.3: minimum score for a text candidate
            try:
                results = self._reader.readtext(
                    preprocessed,
                    detail          = 1,
                    width_ths       = 0.7,
                    text_threshold  = 0.6,
                    low_text        = 0.3,
                )
            except TypeError:
                # Older EasyOCR versions do not have all these parameters.
                results = self._reader.readtext(
                    preprocessed,
                    detail = 1,
                )

            # Scale factor to convert coordinates back to original frame size.
            scale = 1.0 / OCR_SCALE_FACTOR

            detections = []

            for (bbox_points, text, confidence) in results:

                # Skip low confidence.
                if confidence < MIN_WORD_CONFIDENCE:
                    continue

                text = text.strip()

                # Skip too-short text.
                if len(text) < MIN_TEXT_LENGTH:
                    continue

                # Skip text that is mostly symbols/noise.
                alpha_count = sum(1 for c in text if c.isalnum())
                if len(text) > 0 and alpha_count / len(text) < 0.4:
                    # 0.4 threshold (was 0.5) — Arabic characters are
                    # alphanumeric but some scripts have lower density.
                    continue

                # bbox_points is a list of 4 corner points:
                # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                # Convert to (x1, y1, x2, y2) axis-aligned bounding box.
                try:
                    xs = [int(pt[0] * scale) for pt in bbox_points]
                    ys = [int(pt[1] * scale) for pt in bbox_points]

                    x1 = max(0, min(xs) - BBOX_PADDING)
                    y1 = max(0, min(ys) - BBOX_PADDING)
                    x2 = min(w, max(xs) + BBOX_PADDING)
                    y2 = min(h, max(ys) + BBOX_PADDING)

                    # Skip degenerate boxes.
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Skip boxes that are too short.
                    if (y2 - y1) < OCR_MIN_TEXT_HEIGHT_PX:
                        continue

                except (IndexError, TypeError, ValueError):
                    continue

                detections.append({
                    "text":       text,
                    "confidence": confidence,
                    "bbox":       (x1, y1, x2, y2),
                })

            return detections

        except Exception as e:
            logger.error(f"OCR run error: {e}")
            return []


    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_text_distance(
        self,
        frame:     np.ndarray,
        depth_map: np.ndarray
    ) -> float:
        """
        Returns the distance in mm to the nearest detected text.

        Called by control_unit.py background thread every 20 frames
        to feed the state machine for mode switching.

        Also caches the detections so read_text() can reuse them
        without running OCR twice on the same frame.
        """

        detections = self._run_ocr_on_frame(frame)

        # Cache for read_text() reuse.
        self._last_boxes          = detections
        self._last_dist_frame_ts  = get_timestamp_ms()

        if not detections:
            self._last_dist_mm = 0.0
            return 0.0

        # Find the closest text region using the depth map.
        min_dist = float('inf')

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            dist = depth_in_region(depth_map, x1, y1, x2, y2)
            if dist > 0 and dist < min_dist:
                min_dist = dist

        result = 0.0 if min_dist == float('inf') else min_dist
        self._last_dist_mm = result
        return result


    def read_text(self, frame: np.ndarray) -> str:
        """
        Reads and returns stable text from the frame.

        Reuses cached detections from get_text_distance() if called
        within 100ms of it — avoids running OCR twice per cycle.

        Returns the text string to speak, or "" if:
          - No text detected
          - Text not yet stable across N frames
          - Same text as last announcement
        """

        if not self._ready:
            return ""

        self._read_count += 1
        start_ms = get_timestamp_ms()

        # Reuse cached detections if fresh (within 100ms).
        cache_age = get_timestamp_ms() - self._last_dist_frame_ts
        if cache_age < 100 and self._last_boxes:
            detections = self._last_boxes
        else:
            detections = self._run_ocr_on_frame(frame)
            self._last_boxes = detections

        if not detections:
            self._text_history.append("")
            return ""

        # Sort by priority — center of frame first, largest text first.
        h, w = frame.shape[:2]
        detections = self._prioritise(detections, w, h)

        # Combine all text blocks into one string.
        texts   = [d["text"] for d in detections]
        combined = self._clean_text(texts)

        # Stability check — same text must appear in all recent frames.
        self._text_history.append(combined)

        if not self._is_stable():
            logger.debug(f"OCR not stable: '{combined[:40]}'")
            return ""

        # Skip if same as last spoken.
        if combined == self._last_spoken_text:
            return ""

        if combined:
            self._last_spoken_text = combined
            self._success_count   += 1

            elapsed            = get_timestamp_ms() - start_ms
            self._avg_ocr_ms   = self._avg_ocr_ms * 0.8 + elapsed * 0.2

            logger.info(f"OCR: '{combined[:80]}'")

        return combined


    # =========================================================================
    # PREPROCESSING
    # =========================================================================

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Light preprocessing that works for both Arabic and English.

        Deliberately avoids adaptive threshold — it breaks Arabic cursive
        by slicing through the connections between letters.

        Steps:
          1. Convert to grayscale
          2. CLAHE contrast enhancement — helps in uneven lighting
          3. Mild Gaussian blur — reduces camera noise without breaking letters
        """

        try:
            # Grayscale — OCR does not need colour.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # CLAHE — improve contrast locally.
            # clipLimit=2.0: moderate enhancement.
            # tileGridSize=(8,8): local neighbourhood size.
            clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Mild blur — reduce salt-and-pepper noise.
            # (3,3) kernel = very gentle, preserves letter shapes.
            blurred  = cv2.GaussianBlur(enhanced, (3, 3), 0)

            return blurred

        except Exception as e:
            logger.warning(f"Preprocessing failed, using original: {e}")
            return frame


    # =========================================================================
    # TEXT PROCESSING
    # =========================================================================

    def _clean_text(self, text_blocks: List[str]) -> str:
        """Cleans and combines raw OCR text blocks into one spoken string."""

        if not text_blocks:
            return ""

        cleaned = []

        for text in text_blocks:
            text = text.strip()
            if not text:
                continue
            if len(text) < MIN_TEXT_LENGTH:
                continue

            alpha_count = sum(1 for c in text if c.isalnum())
            if len(text) > 0 and alpha_count / len(text) < 0.4:
                continue

            cleaned.append(text)

        if not cleaned:
            return ""

        # Remove duplicates while preserving order.
        unique   = list(dict.fromkeys(cleaned))
        combined = " ".join(unique)

        # Truncate at word boundary.
        if len(combined) > MAX_SPEAK_CHARS:
            truncated  = combined[:MAX_SPEAK_CHARS]
            last_space = truncated.rfind(" ")
            combined   = (
                truncated[:last_space]
                if last_space > MAX_SPEAK_CHARS // 2
                else truncated
            )

        return combined.strip()


    def _is_stable(self) -> bool:
        """True if same non-empty text appeared in all recent frames."""

        if len(self._text_history) < TEXT_STABILITY_FRAMES:
            return False

        unique = set(self._text_history)
        return len(unique) == 1 and list(unique)[0] != ""


    def _prioritise(
        self,
        detections:   List[Dict],
        frame_width:  int,
        frame_height: int
    ) -> List[Dict]:
        """Sorts detections — center of frame and largest text first."""

        cx = frame_width  // 2
        cy = frame_height // 2
        fd = (frame_width**2 + frame_height**2) ** 0.5

        def score(d: Dict) -> float:
            x1, y1, x2, y2 = d["bbox"]
            rx = (x1 + x2) // 2
            ry = (y1 + y2) // 2
            dist  = ((rx - cx)**2 + (ry - cy)**2) ** 0.5
            ndist = dist / fd
            area  = (x2 - x1) * (y2 - y1)
            narea = 1.0 - (area / (frame_width * frame_height))
            return (ndist * 0.7) + (narea * 0.3)

        return sorted(detections, key=score)


    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def reset(self):
        """Resets state when exiting OCR mode."""
        self._text_history.clear()
        self._last_spoken_text = ""
        self._last_boxes       = []
        logger.debug("OCRReader state reset.")


    def get_stats(self) -> Dict:
        return {
            "ready":         self._ready,
            "read_count":    self._read_count,
            "success_count": self._success_count,
            "avg_ocr_ms":    round(self._avg_ocr_ms, 1),
            "history_len":   len(self._text_history),
            "last_text":     self._last_spoken_text[:50],
            "last_dist_mm":  round(self._last_dist_mm, 0),
        }


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_ocr_reader: Optional[OCRReader] = None


def init_ocr():
    """Initialise once at startup. Called from control_unit.py."""

    global _ocr_reader

    if _ocr_reader is not None:
        logger.debug("OCR already initialised.")
        return

    _ocr_reader = OCRReader()
    _ocr_reader.load_model()
    logger.info("Module-level OCR reader ready.")


def read_text(frame: np.ndarray) -> str:
    """Read text from frame. Called in OCR mode by control_unit.py."""

    if _ocr_reader is None:
        logger.warning("OCR not initialised.")
        return ""
    return _ocr_reader.read_text(frame)


def get_text_distance(frame: np.ndarray, depth_map: np.ndarray) -> float:
    """Distance to nearest text in mm. Called every 20 frames."""

    if _ocr_reader is None:
        return 0.0
    return _ocr_reader.get_text_distance(frame, depth_map)


def reset_ocr():
    """Reset state when exiting OCR mode."""

    if _ocr_reader is not None:
        _ocr_reader.reset()


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":

    print("=== ECHORA ocr.py self-test (Arabic + English) ===\n")
    print(f"Languages configured: {OCR_LANGUAGE}")
    print(f"GPU: {_get_ocr_gpu()}\n")

    from camera import EchoraCamera

    cam    = EchoraCamera()
    reader = OCRReader()

    try:
        print("Loading camera and OCR model...")
        cam.init_pipeline()
        reader.load_model()
        print("Ready. Point the camera at Arabic or English text.")
        print("Press Q to quit.\n")

        frame_count = 0
        last_text   = ""

        while True:
            bundle = cam.get_synced_bundle()
            if bundle is None:
                continue

            frame_count += 1
            rgb   = bundle["rgb"]
            depth = bundle["depth"]

            # Run distance check every 5 frames for testing.
            text_dist = 0.0
            if frame_count % 5 == 0:
                text_dist = reader.get_text_distance(rgb, depth)

            # Try to read.
            text = ""
            if text_dist > 0 and text_dist < OCR_TRIGGER_DIST_MM:
                text = reader.read_text(rgb)

            if frame_count % 10 == 0:
                print(
                    f"Frame {frame_count:4d} | "
                    f"Dist: {text_dist:6.0f}mm | "
                    f"Detections: {len(reader._last_boxes):2d}",
                    end=""
                )
                if text:
                    print(f" | READ: '{text}'")
                else:
                    print()

            # Debug overlay.
            debug = rgb.copy()
            for det in reader._last_boxes:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    debug,
                    f"{det['text'][:20]} ({det['confidence']:.2f})",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1
                )

            dist_str = f"Text: {text_dist:.0f}mm" if text_dist > 0 else "No text"
            cv2.putText(debug, dist_str, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if text:
                last_text = text
            if last_text:
                cv2.putText(
                    debug, f"'{last_text[:60]}'",
                    (10, debug.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            cv2.imshow("ECHORA OCR Test", debug)
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        cv2.destroyAllWindows()
        cam.release()
        print(f"\nStats: {reader.get_stats()}")
        print("=== Done ===")