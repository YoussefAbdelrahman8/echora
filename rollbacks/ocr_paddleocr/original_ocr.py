import easyocr
import numpy as np
import cv2
import time
import threading
import torch
from collections import deque, Counter
from typing import List, Dict, Optional

from src.core.config import settings
from src.core.utils import logger, get_timestamp_ms, depth_in_region

MIN_WORD_CONFIDENCE  = 0.3
MIN_TEXT_LENGTH      = 2
MAX_SPEAK_CHARS      = 150
TEXT_STABILITY_FRAMES = 3
OCR_SCALE_FACTOR     = 0.75   # raised from 0.5 — better accuracy; latency hidden by threading
BBOX_PADDING         = 8
INFERENCE_CACHE_MS   = 200    # reuse last inference result if fresher than this


def _get_ocr_gpu() -> bool:
    if torch.cuda.is_available():
        logger.info("OCR: using CUDA GPU.")
        return True
    if torch.backends.mps.is_available():
        logger.info("OCR: MPS detected but not supported by EasyOCR — using CPU.")
        return False
    logger.info("OCR: using CPU.")
    return False


class OCRReader:
    """
    Reads Arabic and English text from camera frames using EasyOCR.

    Designed to be called from a background thread — all inference is
    blocking but the caller (ControlUnit) handles the threading so the
    main camera loop is never stalled.
    """

    def __init__(self):
        self._reader: Optional[easyocr.Reader] = None
        self._ready: bool = False

        # Stability: keep last N results; require majority match
        self._text_history: deque = deque(maxlen=TEXT_STABILITY_FRAMES)
        self._last_spoken_text: str = ""

        # Shared inference cache — avoids running OCR twice in the same frame
        self._last_boxes:          List[Dict] = []
        self._last_inference_ts:   float      = 0.0
        self._last_dist_mm:        float      = 0.0

        self._read_count:   int   = 0
        self._success_count: int  = 0
        self._avg_ocr_ms:   float = 0.0

        # EasyOCR is NOT thread-safe — this lock ensures only one
        # inference runs at a time across both the probe thread and
        # the OCR-mode text thread.
        self._inference_lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def load_model(self):
        logger.info(f"Loading EasyOCR (languages: {settings.OCR_LANGUAGE})...")
        t0 = time.time()
        use_gpu = _get_ocr_gpu()
        try:
            self._reader = easyocr.Reader(
                settings.OCR_LANGUAGE, gpu=use_gpu, verbose=False
            )
            logger.info(f"EasyOCR loaded in {(time.time()-t0)*1000:.0f}ms. gpu={use_gpu}")
        except Exception as e:
            logger.error(f"EasyOCR load failed: {e}")
            raise
        self._ready = True

    def reset(self):
        self._text_history.clear()
        self._last_spoken_text = ""
        self._last_boxes = []
        self._last_inference_ts = 0.0

    # ── Public API (called by module functions) ────────────────────────

    def get_text_distance(self, frame: np.ndarray, depth_map: np.ndarray) -> float:
        """
        Lightweight probe: runs OCR (or returns cached result) and returns
        the depth of the nearest detected text region in mm.
        Used by the state machine to decide whether to enter OCR mode.
        """
        detections = self._run_ocr_cached(frame)
        self._last_dist_mm = 0.0

        if not detections:
            return 0.0

        min_dist = float("inf")
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            d = depth_in_region(depth_map, x1, y1, x2, y2)
            if 0 < d < min_dist:
                min_dist = d

        self._last_dist_mm = 0.0 if min_dist == float("inf") else min_dist
        return self._last_dist_mm

    def read_text(self, frame: np.ndarray) -> str:
        """
        Full OCR pass: run inference (or use cache), apply stability filter,
        and return the cleaned text string ready for TTS.
        Returns "" if text is not stable yet or unchanged since last call.
        """
        if not self._ready:
            return ""

        self._read_count += 1
        t0 = get_timestamp_ms()

        detections = self._run_ocr_cached(frame)

        if not detections:
            self._text_history.append("")
            return ""

        h, w = frame.shape[:2]
        detections = self._prioritise(detections, w, h)
        combined   = self._clean_text([d["text"] for d in detections])

        self._text_history.append(combined)

        stable_text = self._stable_text()
        if not stable_text:
            return ""
        if stable_text == self._last_spoken_text:
            return ""

        self._last_spoken_text = stable_text
        self._success_count   += 1
        elapsed = get_timestamp_ms() - t0
        self._avg_ocr_ms = self._avg_ocr_ms * 0.8 + elapsed * 0.2
        logger.info(f"OCR confirmed: '{stable_text[:80]}'")
        return stable_text

    # ── Core inference ─────────────────────────────────────────────────

    def _run_ocr_cached(self, frame: np.ndarray) -> List[Dict]:
        """
        Returns cached inference result if younger than INFERENCE_CACHE_MS,
        otherwise acquires the inference lock and runs a fresh EasyOCR pass.

        The lock guarantees that the probe thread and the OCR-mode text
        thread never call EasyOCR simultaneously (it is not thread-safe).
        If the lock is already held the caller blocks until inference
        finishes, then gets the freshly cached result.
        """
        age = get_timestamp_ms() - self._last_inference_ts
        if age < INFERENCE_CACHE_MS and self._last_boxes:
            return self._last_boxes

        with self._inference_lock:
            # Re-check cache after acquiring lock — another thread may
            # have just completed inference while we were waiting.
            age = get_timestamp_ms() - self._last_inference_ts
            if age < INFERENCE_CACHE_MS and self._last_boxes:
                return self._last_boxes

            detections = self._run_ocr_on_frame(frame)
            self._last_boxes        = detections
            self._last_inference_ts = get_timestamp_ms()
            return detections

    def _run_ocr_on_frame(self, frame: np.ndarray) -> List[Dict]:
        if not self._ready or self._reader is None:
            return []
        try:
            h, w = frame.shape[:2]
            small = cv2.resize(
                frame,
                (int(w * OCR_SCALE_FACTOR), int(h * OCR_SCALE_FACTOR))
            )
            preprocessed = self._preprocess(small)

            try:
                results = self._reader.readtext(
                    preprocessed,
                    detail=1,
                    width_ths=0.7,
                    text_threshold=0.6,
                    low_text=0.3,
                )
            except TypeError:
                results = self._reader.readtext(preprocessed, detail=1)

            scale      = 1.0 / OCR_SCALE_FACTOR
            detections = []

            for (bbox_points, text, confidence) in results:
                if confidence < MIN_WORD_CONFIDENCE:
                    continue
                text = text.strip()
                if len(text) < MIN_TEXT_LENGTH:
                    continue
                alpha = sum(1 for c in text if c.isalnum())
                if alpha / max(len(text), 1) < 0.4:
                    continue

                try:
                    xs = [int(pt[0] * scale) for pt in bbox_points]
                    ys = [int(pt[1] * scale) for pt in bbox_points]
                    x1 = max(0, min(xs) - BBOX_PADDING)
                    y1 = max(0, min(ys) - BBOX_PADDING)
                    x2 = min(w, max(xs) + BBOX_PADDING)
                    y2 = min(h, max(ys) + BBOX_PADDING)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if (y2 - y1) < settings.OCR_MIN_TEXT_HEIGHT_PX:
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
            logger.error(f"OCR inference error: {e}")
            return []

    # ── Preprocessing ──────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing pipeline tuned for real-world conditions:
          1. Grayscale conversion
          2. Brightness-adaptive contrast (histogram eq. for dark images,
             CLAHE with higher clipLimit for normal/bright ones)
          3. Sharpening kernel to make character edges crisper
          4. Light Gaussian smoothing to reduce sensor noise

        All steps degrade gracefully — on any failure the original frame
        is returned so OCR can still attempt inference.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Brightness-adaptive contrast enhancement
            mean_brightness = float(np.mean(gray))
            if mean_brightness < 60:
                # Very dark — aggressive histogram equalisation first
                gray = cv2.equalizeHist(gray)

            clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced  = clahe.apply(gray)

            # Sharpening: unsharp mask style
            #   sharpened = original + (original - blurred) * strength
            blurred   = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
            sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

            # Mild smoothing to remove high-frequency sensor noise
            denoised  = cv2.GaussianBlur(sharpened, (3, 3), 0)

            return denoised

        except Exception as e:
            logger.warning(f"OCR preprocessing failed — using original: {e}")
            return frame

    # ── Text processing ────────────────────────────────────────────────

    def _stable_text(self) -> str:
        """
        Returns the most-common non-empty text in the history deque,
        provided it appears in at least 2 out of TEXT_STABILITY_FRAMES frames.

        This is more robust than requiring all frames to be identical —
        EasyOCR is slightly non-deterministic and minor differences between
        frames (a letter swapped, punctuation added) would block confirmation.
        """
        if len(self._text_history) < TEXT_STABILITY_FRAMES:
            return ""
        non_empty = [t for t in self._text_history if t]
        if not non_empty:
            return ""
        most_common, count = Counter(non_empty).most_common(1)[0]
        return most_common if count >= 2 else ""

    def _clean_text(self, text_blocks: List[str]) -> str:
        cleaned = []
        for text in text_blocks:
            text = text.strip()
            if not text or len(text) < MIN_TEXT_LENGTH:
                continue
            alpha = sum(1 for c in text if c.isalnum())
            if alpha / max(len(text), 1) < 0.4:
                continue
            cleaned.append(text)

        if not cleaned:
            return ""

        # Arabic is right-to-left — if the content is predominantly Arabic
        # the blocks must be reversed so they read in the correct spoken order.
        if self._is_predominantly_arabic(" ".join(cleaned)):
            cleaned = list(reversed(cleaned))

        unique   = list(dict.fromkeys(cleaned))
        combined = " ".join(unique)

        if len(combined) > MAX_SPEAK_CHARS:
            truncated  = combined[:MAX_SPEAK_CHARS]
            last_space = truncated.rfind(" ")
            combined   = (
                truncated[:last_space]
                if last_space > MAX_SPEAK_CHARS // 2
                else truncated
            )

        return combined.strip()

    @staticmethod
    def _is_predominantly_arabic(text: str) -> bool:
        """Returns True when more than 40 % of characters are Arabic Unicode."""
        if not text:
            return False
        arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
        return arabic / max(len(text), 1) > 0.4

    def _prioritise(
        self, detections: List[Dict], frame_width: int, frame_height: int
    ) -> List[Dict]:
        """Sort detections by proximity to frame centre (most likely what user aims at)."""
        cx = frame_width  // 2
        cy = frame_height // 2
        fd = (frame_width**2 + frame_height**2) ** 0.5

        def score(d: Dict) -> float:
            x1, y1, x2, y2 = d["bbox"]
            rx, ry = (x1 + x2) // 2, (y1 + y2) // 2
            ndist  = ((rx - cx)**2 + (ry - cy)**2) ** 0.5 / fd
            narea  = 1.0 - ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)
            return ndist * 0.7 + narea * 0.3

        return sorted(detections, key=score)

    # ── Stats ──────────────────────────────────────────────────────────

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


# ── Module-level singleton ─────────────────────────────────────────────

_ocr_reader: Optional[OCRReader] = None


def init_ocr():
    global _ocr_reader
    if _ocr_reader is not None:
        return
    _ocr_reader = OCRReader()
    _ocr_reader.load_model()
    logger.info("OCR module ready.")


def read_text(frame: np.ndarray) -> str:
    return _ocr_reader.read_text(frame) if _ocr_reader else ""


def get_text_distance(frame: np.ndarray, depth_map: np.ndarray) -> float:
    return _ocr_reader.get_text_distance(frame, depth_map) if _ocr_reader else 0.0


def reset_ocr():
    if _ocr_reader:
        _ocr_reader.reset()
