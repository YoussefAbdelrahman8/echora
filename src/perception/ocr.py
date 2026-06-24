import time
import threading
from collections import deque, Counter
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.core.config import settings
from src.core.utils import logger, get_timestamp_ms, depth_in_region

# ── Tuning ──────────────────────────────────────────────────────────────
MIN_WORD_CONFIDENCE   = 0.4
MIN_TEXT_LENGTH       = 2
MAX_SPEAK_CHARS       = 150
TEXT_STABILITY_FRAMES = 3
BBOX_PADDING          = 8

DET_CACHE_MS      = 100   # probe cache TTL
DET_STALE_MS      = 2000  # keep stale probe result instead of empty
FULL_OCR_CACHE_MS = 300   # full-read cache TTL

# Preprocessing
OVEREXPOSURE_GAMMA = 0.7
BILATERAL_D        = 5
BILATERAL_SIGMA    = 15


def _use_gpu() -> bool:
    if not settings.OCR_USE_GPU:
        return False
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("OCR: CUDA GPU available — using GPU.")
            return True
    except Exception as e:
        logger.warning(f"OCR: GPU check failed ({e}); falling back to CPU.")
    return False


class OCRReader:
    """
    English-only OCR using PaddleOCR PP-OCRv4 with a fine-tuned English
    recognition model (TextOCR dataset).

    One model instance, two call modes:
      - Probe  (get_text_distance): rec=False — ~20 ms, for state-machine trigger.
      - Read   (read_text):         rec=True  — full det+cls+rec, for TTS output.
    """

    def __init__(self):
        self._ocr:     Optional[Any] = None
        self._ready:   bool = False
        self._use_gpu: bool = False

        self._text_history:     deque = deque(maxlen=TEXT_STABILITY_FRAMES)
        self._last_spoken_text: str   = ""

        # Probe cache (det-only)
        self._det_boxes:        List  = []
        self._det_inference_ts: float = 0.0
        self._det_last_dist_mm: float = 0.0

        # Full-read cache
        self._full_boxes:        List[Dict] = []
        self._full_inference_ts: float      = 0.0

        self._read_count:    int   = 0
        self._success_count: int   = 0
        self._avg_ocr_ms:    float = 0.0

        self._lock = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────

    def load_model(self):
        logger.info("Loading PaddleOCR PP-OCRv4 (English, fine-tuned rec model)...")
        t0 = time.time()
        self._use_gpu = _use_gpu()

        rec_dir = settings.BASE_DIR / "assets" / "models" / "ocr" / "textocr_en_rec"
        if not rec_dir.exists():
            logger.warning(f"OCR: fine-tuned rec model not found at {rec_dir} — using default EN rec.")
            rec_dir_str = None
        else:
            rec_dir_str = str(rec_dir)
            logger.info(f"OCR rec model: {rec_dir}")

        try:
            from paddleocr import PaddleOCR
            kwargs: Dict[str, Any] = dict(
                use_angle_cls=True,
                lang="en",
                use_gpu=self._use_gpu,
                show_log=False,
                det_db_unclip_ratio=1.6,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                drop_score=MIN_WORD_CONFIDENCE,
                use_mp=False,
            )
            if rec_dir_str is not None:
                kwargs["rec_model_dir"] = rec_dir_str

            self._ocr = PaddleOCR(**kwargs)
            self._ready = True
            logger.info(
                f"PaddleOCR ready in {(time.time() - t0) * 1000:.0f} ms  "
                f"gpu={self._use_gpu}  "
                f"rec={'fine-tuned' if rec_dir_str else 'default'}"
            )
        except Exception as e:
            logger.error(f"PaddleOCR failed to load: {e}")
            raise

    def reset(self):
        self._text_history.clear()
        self._last_spoken_text  = ""
        self._det_boxes         = []
        self._det_inference_ts  = 0.0
        self._full_boxes        = []
        self._full_inference_ts = 0.0

    # ── Public API ─────────────────────────────────────────────────────

    def get_text_distance(self, frame: np.ndarray, depth_map: np.ndarray) -> float:
        """
        Fast probe: depth (mm) of the nearest detected text region.
        Uses det-only PaddleOCR (rec=False) — ~20 ms.
        Returns 0.0 when no text found.
        """
        boxes = self._run_det_cached(frame)
        if not boxes:
            self._det_last_dist_mm = 0.0
            return 0.0

        min_dist = float("inf")
        for x1, y1, x2, y2 in boxes:
            d = depth_in_region(depth_map, x1, y1, x2, y2)
            if 0 < d < min_dist:
                min_dist = d

        self._det_last_dist_mm = 0.0 if min_dist == float("inf") else min_dist
        return self._det_last_dist_mm

    def read_text(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> str:
        """
        Full OCR using the fine-tuned English recognition model.
        Returns stable text once TEXT_STABILITY_FRAMES consecutive reads agree;
        returns "" when text is unstable or identical to the last spoken text.
        """
        if not self._ready:
            return ""

        self._read_count += 1
        t0 = get_timestamp_ms()

        detections = self._run_full_ocr_cached(frame)

        if not detections:
            self._text_history.append("")
            return ""

        h, w       = frame.shape[:2]
        detections = self._prioritise(detections, w, h)
        combined   = self._clean_text([d["text"] for d in detections])

        self._text_history.append(combined)

        stable = self._stable_text()
        if not stable:
            return ""
        if stable == self._last_spoken_text:
            return ""

        self._last_spoken_text = stable
        self._success_count   += 1
        elapsed = get_timestamp_ms() - t0
        self._avg_ocr_ms = self._avg_ocr_ms * 0.8 + elapsed * 0.2
        logger.info(f"OCR confirmed: '{stable[:80]}'")
        return stable

    # ── Probe: detection-only ─────────────────────────────────────────

    def _run_det_cached(self, frame: np.ndarray) -> List[tuple]:
        age = get_timestamp_ms() - self._det_inference_ts
        if age < DET_CACHE_MS:
            return self._det_boxes

        if not self._is_frame_sharp_enough(frame):
            return self._det_boxes if age < DET_STALE_MS else []

        with self._lock:
            age = get_timestamp_ms() - self._det_inference_ts
            if age < DET_CACHE_MS:
                return self._det_boxes

            boxes = self._run_det_on_frame(frame)
            self._det_boxes        = boxes
            self._det_inference_ts = get_timestamp_ms()
            return boxes

    def _run_det_on_frame(self, frame: np.ndarray) -> List[tuple]:
        if self._ocr is None:
            return []
        try:
            result = self._ocr.ocr(frame, rec=False, cls=False)
            if not result:
                return []
            items = result[0]
            if items is None or (hasattr(items, "__len__") and len(items) == 0):
                return []

            h, w = frame.shape[:2]
            boxes = []
            for item in items:
                quad = item[0] if (isinstance(item, (list, tuple)) and len(item) == 1) else item
                xs = [int(pt[0]) for pt in quad]
                ys = [int(pt[1]) for pt in quad]
                x1 = max(0, min(xs) - BBOX_PADDING)
                y1 = max(0, min(ys) - BBOX_PADDING)
                x2 = min(w, max(xs) + BBOX_PADDING)
                y2 = min(h, max(ys) + BBOX_PADDING)
                if x2 > x1 and y2 > y1 and (y2 - y1) >= settings.OCR_MIN_TEXT_HEIGHT_PX:
                    boxes.append((x1, y1, x2, y2))
            return boxes

        except Exception as e:
            logger.error(f"OCR det pass error: {e}")
            return []

    # ── Full OCR: det + cls + rec ─────────────────────────────────────

    def _run_full_ocr_cached(self, frame: np.ndarray) -> List[Dict]:
        age = get_timestamp_ms() - self._full_inference_ts
        if age < FULL_OCR_CACHE_MS and self._full_boxes:
            return self._full_boxes

        if not self._is_frame_sharp_enough(frame):
            return self._full_boxes

        with self._lock:
            age = get_timestamp_ms() - self._full_inference_ts
            if age < FULL_OCR_CACHE_MS and self._full_boxes:
                return self._full_boxes

            detections = self._run_full_ocr_on_frame(frame)
            self._full_boxes        = detections
            self._full_inference_ts = get_timestamp_ms()
            return detections

    def _run_full_ocr_on_frame(self, frame: np.ndarray) -> List[Dict]:
        if self._ocr is None:
            return []
        try:
            preprocessed = self._preprocess(frame)
            result = self._ocr.ocr(preprocessed, rec=True, cls=True)
            if not result or result[0] is None:
                return []

            h, w = frame.shape[:2]
            detections = []
            for line in result[0]:
                if line is None:
                    continue

                bbox_points, (text, confidence) = line
                text = text.strip()
                if not text or len(text) < MIN_TEXT_LENGTH:
                    continue
                if confidence < MIN_WORD_CONFIDENCE:
                    continue

                alpha = sum(1 for c in text if c.isalnum())
                if alpha / max(len(text), 1) < 0.4:
                    continue

                xs = [int(pt[0]) for pt in bbox_points]
                ys = [int(pt[1]) for pt in bbox_points]
                x1 = max(0, min(xs) - BBOX_PADDING)
                y1 = max(0, min(ys) - BBOX_PADDING)
                x2 = min(w, max(xs) + BBOX_PADDING)
                y2 = min(h, max(ys) + BBOX_PADDING)

                if x2 <= x1 or y2 <= y1:
                    continue
                if (y2 - y1) < settings.OCR_MIN_TEXT_HEIGHT_PX:
                    continue

                detections.append({
                    "text":       text,
                    "confidence": float(confidence),
                    "bbox":       (x1, y1, x2, y2),
                })

            return detections

        except Exception as e:
            logger.error(f"OCR full pass error: {e}")
            return []

    # ── Preprocessing ──────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        try:
            gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray))

            if mean_brightness < 60:
                lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            elif mean_brightness < 100:
                lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            elif mean_brightness > 200:
                lut   = np.array(
                    [int(((i / 255.0) ** (1.0 / OVEREXPOSURE_GAMMA)) * 255)
                     for i in range(256)],
                    dtype=np.uint8,
                )
                frame = cv2.LUT(frame, lut)

            blurred   = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.5)
            sharpened = cv2.addWeighted(frame, 1.3, blurred, -0.3, 0)
            denoised  = cv2.bilateralFilter(sharpened, BILATERAL_D, BILATERAL_SIGMA, BILATERAL_SIGMA)
            return denoised

        except Exception as e:
            logger.warning(f"OCR preprocessing failed — using original: {e}")
            return frame

    def _is_frame_sharp_enough(self, frame: np.ndarray) -> bool:
        try:
            gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if variance < settings.OCR_BLUR_THRESHOLD:
                logger.debug(f"OCR: blurry frame (var={variance:.1f}) — skipped.")
                return False
            return True
        except Exception:
            return True

    # ── Text processing ────────────────────────────────────────────────

    def _stable_text(self) -> str:
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

    def _prioritise(
        self, detections: List[Dict], frame_width: int, frame_height: int
    ) -> List[Dict]:
        cx = frame_width  // 2
        cy = frame_height // 2
        fd = (frame_width**2 + frame_height**2) ** 0.5

        def score(d: Dict) -> float:
            x1, y1, x2, y2 = d["bbox"]
            rx = (x1 + x2) // 2
            ry = (y1 + y2) // 2
            ndist = ((rx - cx)**2 + (ry - cy)**2) ** 0.5 / fd
            narea = 1.0 - ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)
            return ndist * 0.7 + narea * 0.3

        return sorted(detections, key=score)

    # ── Debug overlay ──────────────────────────────────────────────────

    def draw_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        if not self._ready:
            cv2.putText(frame, "OCR: not loaded",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
            return frame

        for x1, y1, x2, y2 in self._det_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        dist_str = (
            f"Text: {self._det_last_dist_mm:.0f} mm"
            if self._det_last_dist_mm > 0 else "No text nearby"
        )
        cv2.putText(frame, dist_str, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if self._last_spoken_text:
            cv2.putText(
                frame,
                f"'{self._last_spoken_text[:60]}'",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
        return frame

    # ── Stats ──────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            "ready":         self._ready,
            "engine":        "PaddleOCR PP-OCRv4 (fine-tuned EN rec)",
            "read_count":    self._read_count,
            "success_count": self._success_count,
            "avg_ocr_ms":    round(self._avg_ocr_ms, 1),
            "history_len":   len(self._text_history),
            "last_text":     self._last_spoken_text[:50],
            "last_dist_mm":  round(self._det_last_dist_mm, 0),
        }


# ── Module-level singleton ───────────────────────────────────────────────

_ocr_reader: Optional[OCRReader] = None


def init_ocr():
    global _ocr_reader
    if _ocr_reader is not None:
        return
    _ocr_reader = OCRReader()
    _ocr_reader.load_model()
    logger.info("OCR module ready.")


def read_text(frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> str:
    return _ocr_reader.read_text(frame, depth_map=depth_map) if _ocr_reader else ""


def get_text_distance(frame: np.ndarray, depth_map: np.ndarray) -> float:
    return _ocr_reader.get_text_distance(frame, depth_map) if _ocr_reader else 0.0


def reset_ocr():
    if _ocr_reader:
        _ocr_reader.reset()
