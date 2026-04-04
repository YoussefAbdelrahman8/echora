from ultralytics import YOLO
import numpy as np
import cv2
import time
from collections import deque
from typing import Optional, Dict, List

from src.core.config import (
    BANKNOTE_MODEL_PATH,
    BANKNOTE_CONFIDENCE_THRESHOLD,
    BANKNOTE_STABILITY_FRAMES,
    BANKNOTE_MAX_DIST_MM,
)
from src.core.utils import logger, get_timestamp_ms, depth_in_region

DENOMINATION_MAP = {
    # Numeric string format
    "5":           "5 pounds", "10":          "10 pounds", "20":          "20 pounds",
    "50":          "50 pounds", "100":         "100 pounds", "200":         "200 pounds",
    # EGP suffix format
    "5_EGP":       "5 pounds", "10_EGP":      "10 pounds", "20_EGP":      "20 pounds",
    "50_EGP":      "50 pounds", "100_EGP":     "100 pounds", "200_EGP":     "200 pounds",
    # Underscore format
    "5_pounds":    "5 pounds", "10_pounds":   "10 pounds", "20_pounds":   "20 pounds",
    "50_pounds":   "50 pounds", "100_pounds":  "100 pounds", "200_pounds":  "200 pounds",
    # Word format
    "five":        "5 pounds", "ten":         "10 pounds", "twenty":      "20 pounds",
    "fifty":       "50 pounds", "hundred":     "100 pounds", "two_hundred": "200 pounds",
    # Hyphen format
    "5-pounds":    "5 pounds", "10-pounds":   "10 pounds", "20-pounds":   "20 pounds",
    "50-pounds":   "50 pounds", "100-pounds":  "100 pounds", "200-pounds":  "200 pounds",
    # Roboflow sometimes adds spaces
    "5 pounds":    "5 pounds", "10 pounds":   "10 pounds", "20 pounds":   "20 pounds",
    "50 pounds":   "50 pounds", "100 pounds":  "100 pounds", "200 pounds":  "200 pounds",
}

class BanknoteDetector:
    """Detects and classifies Egyptian Pound banknotes using YOLOv8."""

    def __init__(self):
        self._model: Optional[YOLO] = None
        self._ready: bool = False
        self._stub_mode: bool = False
        self._device: str = "cpu"
        
        self._denomination_history: deque = deque(maxlen=BANKNOTE_STABILITY_FRAMES)
        self._last_spoken: str = ""
        
        self._detect_count: int = 0
        self._success_count: int = 0
        self._avg_infer_ms: float = 0.0

    def load_model(self):
        if not BANKNOTE_MODEL_PATH.exists():
            logger.warning(f"Banknote model not found at: {BANKNOTE_MODEL_PATH}. Running in STUB MODE.")
            self._stub_mode = True
            self._ready = True
            return

        import torch
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
            
        logger.info(f"Loading banknote model on {self._device}...")
        
        try:
            self._model = YOLO(str(BANKNOTE_MODEL_PATH))
            for class_name in self._model.names.values():
                if class_name not in DENOMINATION_MAP:
                    logger.warning(f"Unmapped banknote class: '{class_name}'. Add it to DENOMINATION_MAP.")
        except Exception as e:
            logger.error(f"Failed to load banknote model: {e}")
            self._stub_mode = True

        self._ready = True
        logger.info(f"BanknoteDetector ready ({'stub mode' if self._stub_mode else 'real detection'}).")

    def detect_banknote(self, rgb_frame: np.ndarray) -> bool:
        """Fast check to see if a banknote is visible."""
        if self._stub_mode or self._model is None:
            return False

        try:
            results = self._model(
                rgb_frame, verbose=False, conf=BANKNOTE_CONFIDENCE_THRESHOLD,
                imgsz=320, device=self._device, half=(self._device == "cuda")
            )
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                return True
            return False
        except Exception as e:
            logger.error(f"Banknote detection error: {e}")
            return False

    def classify_denomination(self, rgb_frame: np.ndarray) -> str:
        """Fully classifies the denomination of the banknote."""
        if self._stub_mode or self._model is None:
            return ""

        self._detect_count += 1
        start_ms = get_timestamp_ms()

        try:
            results = self._model(
                rgb_frame, verbose=False, conf=BANKNOTE_CONFIDENCE_THRESHOLD,
                imgsz=640, device=self._device, half=(self._device == "cuda")
            )
            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                self._denomination_history.append("")
                return ""

            best_idx = int(result.boxes.conf.argmax())
            best_box = result.boxes[best_idx]
            class_idx = int(best_box.cls.item())
            confidence = float(best_box.conf.item())
            raw_class = self._model.names[class_idx]
            denomination = DENOMINATION_MAP.get(raw_class)

            if denomination is None:
                logger.warning(f"Unknown banknote class: '{raw_class}'.")
                self._denomination_history.append("")
                return ""

            self._denomination_history.append(denomination)

            if not self._is_stable() or denomination == self._last_spoken:
                return ""

            self._last_spoken = denomination
            self._success_count += 1
            
            elapsed = get_timestamp_ms() - start_ms
            self._avg_infer_ms = (self._avg_infer_ms * 0.8) + (elapsed * 0.2)
            
            logger.info(f"Banknote classified: {denomination} (conf={confidence:.2f})")
            return denomination

        except Exception as e:
            logger.error(f"Banknote classification error: {e}")
            return ""

    def is_note_in_range(self, rgb_frame: np.ndarray, depth_map: np.ndarray) -> bool:
        if self._stub_mode or self._model is None:
            return False

        try:
            results = self._model(
                rgb_frame, verbose=False, conf=BANKNOTE_CONFIDENCE_THRESHOLD,
                imgsz=320, device=self._device, half=(self._device == "cuda")
            )
            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                return False

            best_idx = int(result.boxes.conf.argmax())
            best_box = result.boxes[best_idx]
            xyxy = best_box.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            
            dist_mm = depth_in_region(depth_map, x1, y1, x2, y2)
            if dist_mm <= 0:
                return True

            return dist_mm <= BANKNOTE_MAX_DIST_MM

        except Exception as e:
            logger.error(f"Banknote range check error: {e}")
            return False

    def _is_stable(self) -> bool:
        if len(self._denomination_history) < BANKNOTE_STABILITY_FRAMES:
            return False

        unique = set(self._denomination_history)
        return len(unique) == 1 and list(unique)[0] != ""

    def draw_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        if self._stub_mode or self._model is None:
            cv2.putText(frame, "BANKNOTE: stub mode", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
            return frame

        try:
            results = self._model(
                frame, verbose=False, conf=BANKNOTE_CONFIDENCE_THRESHOLD,
                imgsz=640, device=self._device, half=(self._device=="cuda")
            )
            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                cv2.putText(frame, "BANKNOTE: no note detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
                return frame

            for box in result.boxes:
                xyxy = box.xyxy.cpu().numpy()[0]
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                x2, y2 = int(xyxy[2]), int(xyxy[3])
                confidence = float(box.conf.item())
                class_idx = int(box.cls.item())
                raw_class = self._model.names[class_idx]
                denomination = DENOMINATION_MAP.get(raw_class, raw_class)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 2)
                label = f"{denomination} ({confidence:.0%})"
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

            stability_str = f"Stable: {len(self._denomination_history)}/{BANKNOTE_STABILITY_FRAMES}"
            cv2.putText(frame, stability_str, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)

        except Exception as e:
            logger.error(f"Banknote overlay error: {e}")

        return frame

    def reset(self):
        self._denomination_history.clear()
        self._last_spoken = ""

    def get_stats(self) -> Dict:
        return {
            "ready":        self._ready,
            "stub_mode":    self._stub_mode,
            "detect_count": self._detect_count,
            "success_count": self._success_count,
            "avg_infer_ms": round(self._avg_infer_ms, 1),
            "last_spoken":  self._last_spoken,
            "stability":    len(self._denomination_history),
        }

_detector: Optional[BanknoteDetector] = None

def init_banknote():
    global _detector
    if _detector is not None:
        return
    _detector = BanknoteDetector()
    _detector.load_model()
    logger.info("Module-level banknote detector ready.")

def detect_banknote(rgb_frame: np.ndarray) -> bool:
    if _detector is None:
        return False
    return _detector.detect_banknote(rgb_frame)

def classify_denomination(rgb_frame: np.ndarray) -> str:
    if _detector is None:
        return ""
    return _detector.classify_denomination(rgb_frame)

def reset_banknote():
    if _detector is not None:
        _detector.reset()