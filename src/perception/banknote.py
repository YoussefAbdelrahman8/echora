import time
from collections import deque, Counter
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.core.config import settings
from src.core.utils import logger, get_timestamp_ms, depth_in_region

# Reuse the last YOLO result if it is younger than one frame (~33 ms).
# This prevents running inference twice when detect_banknote() and
# classify_denomination() are both called in the same frame.
_CACHE_TTL_MS = 40


class BanknoteDetector:
    """
    Detects and classifies Egyptian Pound banknotes using a custom YOLOv8
    detection model (banknote_egp.pt).  Class names and the number of
    classes are read directly from the model — no hard-coded label map needed.
    """

    def __init__(self):
        self._yolo:   Optional[YOLO] = None
        self._device: str            = "cpu"
        self._ready:  bool           = False

        # Stability: keep last N results; require majority match
        self._history:     deque = deque(maxlen=settings.BANKNOTE_STABILITY_FRAMES)
        self._last_spoken: str   = ""

        # Per-frame inference cache
        self._cache_dets: List[Dict] = []
        self._cache_ts:   float      = 0.0   # get_timestamp_ms() of last run

        self._infer_count:   int   = 0
        self._success_count: int   = 0
        self._avg_ms:        float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────

    def load_model(self):
        if not settings.BANKNOTE_MODEL_PATH.exists():
            logger.error(f"Banknote model not found: {settings.BANKNOTE_MODEL_PATH}")
            return

        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        logger.info(f"Loading banknote YOLOv8 on {self._device}…")
        try:
            self._yolo = YOLO(str(settings.BANKNOTE_MODEL_PATH))

            blank = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(2):
                self._yolo(blank, verbose=False, device=self._device,
                           half=(self._device == "cuda"))

            logger.info(
                f"BanknoteDetector ready — "
                f"{len(self._yolo.names)} classes: {list(self._yolo.names.values())}"
            )
            self._ready = True
        except Exception as e:
            logger.error(f"Failed to load banknote model: {e}")

    # ── Public API ────────────────────────────────────────────────────

    def detect_banknote(self, frame: np.ndarray) -> bool:
        """Quick probe: True if any banknote is above the confidence threshold."""
        if not self._ready:
            return False
        return len(self._run_yolo_cached(frame)) > 0

    def classify_denomination(self, frame: np.ndarray) -> str:
        """
        Full classification.  Returns the spoken denomination string when
        stable (majority of BANKNOTE_STABILITY_FRAMES agree) and not the
        same as the last announcement.  Returns "" otherwise.
        """
        if not self._ready:
            return ""

        self._infer_count += 1
        t0 = get_timestamp_ms()

        dets = self._run_yolo_cached(frame)

        if not dets:
            self._history.append("")
            return ""

        # Pick the highest-confidence detection
        best  = max(dets, key=lambda d: d["confidence"])
        label = best["label"]

        # If a clearly different denomination is starting to appear, clear
        # the stale history so the switch registers within 1-2 frames
        # instead of waiting for the full window to rotate out.
        if self._history and label != "" and all(
            h != "" and h != label for h in self._history
        ):
            self._history.clear()

        self._history.append(label)

        stable = self._stable_label()
        if not stable or stable == self._last_spoken:
            return ""

        self._last_spoken   = stable
        self._success_count += 1
        elapsed = get_timestamp_ms() - t0
        self._avg_ms = self._avg_ms * 0.8 + elapsed * 0.2
        logger.info(
            f"Banknote: {stable}  "
            f"(conf={best['confidence']:.0%}  {elapsed:.0f}ms)"
        )
        return stable

    def is_note_in_range(self, frame: np.ndarray, depth_map: np.ndarray) -> bool:
        """
        True if a banknote is detected and the depth at its bounding box
        is within BANKNOTE_MAX_DIST_MM.
        """
        if not self._ready:
            return False
        dets = self._run_yolo_cached(frame)
        if not dets:
            return False

        best = max(dets, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = best["bbox"]
        d = depth_in_region(depth_map, x1, y1, x2, y2)
        if d <= 0:
            return True   # no depth reading — assume in range
        return d <= settings.BANKNOTE_MAX_DIST_MM

    # ── Core inference ────────────────────────────────────────────────

    def _run_yolo_cached(self, frame: np.ndarray) -> List[Dict]:
        """
        Returns the cached result if it is younger than _CACHE_TTL_MS,
        otherwise runs a fresh YOLO inference.  This ensures detect_banknote()
        and classify_denomination() only trigger one GPU/CPU call per frame.
        """
        if get_timestamp_ms() - self._cache_ts < _CACHE_TTL_MS:
            return self._cache_dets

        dets = self._run_yolo(frame)
        self._cache_dets = dets
        self._cache_ts   = get_timestamp_ms()
        return dets

    def _run_yolo(self, frame: np.ndarray) -> List[Dict]:
        if self._yolo is None:
            return []
        try:
            results = self._yolo(
                frame,
                verbose=False,
                conf=settings.BANKNOTE_CONFIDENCE_THRESHOLD,
                device=self._device,
                half=(self._device == "cuda"),
            )
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return []

            dets = []
            for box in result.boxes:
                xyxy      = box.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                class_idx = int(box.cls.item())
                label     = self._yolo.names.get(class_idx, f"Class {class_idx}")
                conf      = float(box.conf.item())
                dets.append({
                    "bbox":       (x1, y1, x2, y2),
                    "class_idx":  class_idx,
                    "label":      label,
                    "confidence": conf,
                })

            # Log every raw detection at INFO so you can see real confidence values
            if dets:
                summary = "  |  ".join(
                    f"{d['label']} {d['confidence']:.0%}" for d in dets
                )
                logger.info(f"Banknote raw: {summary}")

            return dets
        except Exception as e:
            logger.error(f"Banknote inference error: {e}")
            return []

    # ── Stability ─────────────────────────────────────────────────────

    def _stable_label(self) -> str:
        if len(self._history) < settings.BANKNOTE_STABILITY_FRAMES:
            return ""
        non_empty = [x for x in self._history if x]
        if not non_empty:
            return ""
        label, count = Counter(non_empty).most_common(1)[0]
        return label if count >= 2 else ""

    # ── Debug overlay ─────────────────────────────────────────────────

    def draw_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        if not self._ready:
            cv2.putText(frame, "BANKNOTE: model not loaded",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
            return frame

        # Use cached detections — avoids a third YOLO call just for the overlay
        dets = self._cache_dets

        if not dets:
            cv2.putText(frame, "No banknote detected",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        else:
            for det in dets:
                x1, y1, x2, y2 = det["bbox"]
                conf  = det["confidence"]
                label = det["label"]
                color = (0, 215, 255) if conf >= settings.BANKNOTE_CONFIDENCE_THRESHOLD \
                        else (100, 100, 100)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}  {conf:.0%}",
                            (x1, max(y1 - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # Stability bar and last spoken
        stab = f"Stable: {len(self._history)}/{settings.BANKNOTE_STABILITY_FRAMES}  Last: {self._last_spoken or '—'}"
        cv2.putText(frame, stab, (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1, cv2.LINE_AA)

        return frame

    # ── Misc ──────────────────────────────────────────────────────────

    def reset(self):
        self._history.clear()
        self._last_spoken = ""
        self._cache_dets  = []
        self._cache_ts    = 0.0

    def get_stats(self) -> Dict:
        return {
            "ready":          self._ready,
            "device":         self._device,
            "infer_count":    self._infer_count,
            "success_count":  self._success_count,
            "avg_ms":         round(self._avg_ms, 1),
            "last_spoken":    self._last_spoken,
            "stability":      len(self._history),
            "conf_threshold": settings.BANKNOTE_CONFIDENCE_THRESHOLD,
        }


# ── Module-level singleton ─────────────────────────────────────────────

_detector: Optional[BanknoteDetector] = None


def init_banknote():
    global _detector
    if _detector is not None:
        return
    _detector = BanknoteDetector()
    _detector.load_model()


def detect_banknote(frame: np.ndarray) -> bool:
    return _detector.detect_banknote(frame) if _detector else False


def classify_denomination(frame: np.ndarray) -> str:
    return _detector.classify_denomination(frame) if _detector else ""


def reset_banknote():
    if _detector:
        _detector.reset()
