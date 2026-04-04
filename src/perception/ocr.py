import easyocr
import numpy as np
import cv2
import time
import torch
from collections import deque
from typing import List, Dict, Optional

from src.core.config import (
    OCR_CONFIDENCE_THRESHOLD,
    OCR_MIN_TEXT_HEIGHT_PX,
    OCR_MAX_CHARS,
    OCR_TRIGGER_DIST_MM,
    OCR_LANGUAGE,
    DEPTH_MIN_MM,
    DEPTH_MAX_MM,
)
from src.core.utils import logger, get_timestamp_ms, depth_in_region

MIN_WORD_CONFIDENCE = 0.3
MIN_TEXT_LENGTH = 2
MAX_SPEAK_CHARS = 150
TEXT_STABILITY_FRAMES = 3
OCR_SCALE_FACTOR = 0.5
BBOX_PADDING = 8

def _get_ocr_gpu() -> bool:
    if torch.cuda.is_available():
        logger.info("OCR will use CUDA GPU.")
        return True
    if torch.backends.mps.is_available():
        logger.info("OCR will use CPU (MPS not supported by EasyOCR).")
        return False
    logger.info("OCR will use CPU.")
    return False

class OCRReader:
    """Reads Arabic and English text from camera frames using a single pass EasyOCR approach."""

    def __init__(self):
        self._reader: Optional[easyocr.Reader] = None
        self._ready: bool = False
        self._text_history: deque = deque(maxlen=TEXT_STABILITY_FRAMES)
        self._last_spoken_text: str = ""
        self._last_dist_mm: float = 0.0
        self._last_dist_frame_ts: float = 0.0
        self._last_boxes: List[Dict] = []

        self._read_count: int = 0
        self._success_count: int = 0
        self._avg_ocr_ms: float = 0.0

    def load_model(self):
        logger.info(f"Loading EasyOCR (languages: {OCR_LANGUAGE})...")
        start = time.time()
        use_gpu = _get_ocr_gpu()

        try:
            self._reader = easyocr.Reader(OCR_LANGUAGE, gpu=use_gpu, verbose=False)
            elapsed = round((time.time() - start) * 1000)
            logger.info(f"EasyOCR loaded in {elapsed}ms. GPU={use_gpu}")
        except Exception as e:
            logger.error(f"EasyOCR failed to load: {e}")
            raise

        self._ready = True
        logger.info("OCRReader ready.")

    def _run_ocr_on_frame(self, frame: np.ndarray) -> List[Dict]:
        if not self._ready or self._reader is None:
            return []

        try:
            h, w = frame.shape[:2]
            new_w = int(w * OCR_SCALE_FACTOR)
            new_h = int(h * OCR_SCALE_FACTOR)
            small = cv2.resize(frame, (new_w, new_h))

            preprocessed = self._preprocess(small)

            try:
                results = self._reader.readtext(
                    preprocessed, detail=1, width_ths=0.7, text_threshold=0.6, low_text=0.3
                )
            except TypeError:
                results = self._reader.readtext(preprocessed, detail=1)

            scale = 1.0 / OCR_SCALE_FACTOR
            detections = []

            for (bbox_points, text, confidence) in results:
                if confidence < MIN_WORD_CONFIDENCE:
                    continue

                text = text.strip()
                if len(text) < MIN_TEXT_LENGTH:
                    continue

                alpha_count = sum(1 for c in text if c.isalnum())
                if len(text) > 0 and alpha_count / len(text) < 0.4:
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

                    if (y2 - y1) < OCR_MIN_TEXT_HEIGHT_PX:
                        continue

                except (IndexError, TypeError, ValueError):
                    continue

                detections.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                })

            return detections
        except Exception as e:
            logger.error(f"OCR run error: {e}")
            return []

    def get_text_distance(self, frame: np.ndarray, depth_map: np.ndarray) -> float:
        detections = self._run_ocr_on_frame(frame)

        self._last_boxes = detections
        self._last_dist_frame_ts = get_timestamp_ms()

        if not detections:
            self._last_dist_mm = 0.0
            return 0.0

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
        if not self._ready:
            return ""

        self._read_count += 1
        start_ms = get_timestamp_ms()

        cache_age = get_timestamp_ms() - self._last_dist_frame_ts
        if cache_age < 100 and self._last_boxes:
            detections = self._last_boxes
        else:
            detections = self._run_ocr_on_frame(frame)
            self._last_boxes = detections

        if not detections:
            self._text_history.append("")
            return ""

        h, w = frame.shape[:2]
        detections = self._prioritise(detections, w, h)

        texts = [d["text"] for d in detections]
        combined = self._clean_text(texts)

        self._text_history.append(combined)

        if not self._is_stable():
            return ""
        if combined == self._last_spoken_text:
            return ""

        if combined:
            self._last_spoken_text = combined
            self._success_count += 1
            elapsed = get_timestamp_ms() - start_ms
            self._avg_ocr_ms = self._avg_ocr_ms * 0.8 + elapsed * 0.2
            logger.info(f"OCR: '{combined[:80]}'")

        return combined

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            return blurred
        except Exception as e:
            logger.warning(f"Preprocessing failed, using original: {e}")
            return frame

    def _clean_text(self, text_blocks: List[str]) -> str:
        if not text_blocks:
            return ""

        cleaned = []
        for text in text_blocks:
            text = text.strip()
            if not text or len(text) < MIN_TEXT_LENGTH:
                continue

            alpha_count = sum(1 for c in text if c.isalnum())
            if len(text) > 0 and alpha_count / len(text) < 0.4:
                continue

            cleaned.append(text)

        if not cleaned:
            return ""

        unique = list(dict.fromkeys(cleaned))
        combined = " ".join(unique)

        if len(combined) > MAX_SPEAK_CHARS:
            truncated = combined[:MAX_SPEAK_CHARS]
            last_space = truncated.rfind(" ")
            combined = truncated[:last_space] if last_space > MAX_SPEAK_CHARS // 2 else truncated

        return combined.strip()

    def _is_stable(self) -> bool:
        if len(self._text_history) < TEXT_STABILITY_FRAMES:
            return False
        unique = set(self._text_history)
        return len(unique) == 1 and list(unique)[0] != ""

    def _prioritise(self, detections: List[Dict], frame_width: int, frame_height: int) -> List[Dict]:
        cx, cy = frame_width // 2, frame_height // 2
        fd = (frame_width**2 + frame_height**2) ** 0.5

        def score(d: Dict) -> float:
            x1, y1, x2, y2 = d["bbox"]
            rx, ry = (x1 + x2) // 2, (y1 + y2) // 2
            dist = ((rx - cx)**2 + (ry - cy)**2) ** 0.5
            ndist = dist / fd
            area = (x2 - x1) * (y2 - y1)
            narea = 1.0 - (area / (frame_width * frame_height))
            return (ndist * 0.7) + (narea * 0.3)

        return sorted(detections, key=score)

    def reset(self):
        self._text_history.clear()
        self._last_spoken_text = ""
        self._last_boxes = []

    def get_stats(self) -> Dict:
        return {
            "ready": self._ready,
            "read_count": self._read_count,
            "success_count": self._success_count,
            "avg_ocr_ms": round(self._avg_ocr_ms, 1),
            "history_len": len(self._text_history),
            "last_text": self._last_spoken_text[:50],
            "last_dist_mm": round(self._last_dist_mm, 0),
        }

_ocr_reader: Optional[OCRReader] = None

def init_ocr():
    global _ocr_reader
    if _ocr_reader is not None:
        return
    _ocr_reader = OCRReader()
    _ocr_reader.load_model()
    logger.info("Module-level OCR reader ready.")

def read_text(frame: np.ndarray) -> str:
    if _ocr_reader is None:
        return ""
    return _ocr_reader.read_text(frame)

def get_text_distance(frame: np.ndarray, depth_map: np.ndarray) -> float:
    if _ocr_reader is None:
        return 0.0
    return _ocr_reader.get_text_distance(frame, depth_map)

def reset_ocr():
    if _ocr_reader is not None:
        _ocr_reader.reset()