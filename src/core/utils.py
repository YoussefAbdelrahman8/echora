import cv2
import numpy as np
import time
import logging
from typing import Dict, Tuple, Optional, List

from src.core.config import (
    DANGER_DIST_MM,
    WARNING_DIST_MM,
    CAMERA_HFOV_DEG,
    ALERT_COOLDOWN_SEC,
    LOG_LEVEL,
    LOG_PATH,
)

logger = logging.getLogger("ECHORA")
logger.setLevel(getattr(logging, LOG_LEVEL))

stream_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

try:
    file_handler = logging.FileHandler(LOG_PATH, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
except Exception:
    pass


def bbox_center(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    return (x1 + x2) // 2, (y1 + y2) // 2


def angle_from_x(cx: int, frame_width: int) -> float:
    normalised = cx / frame_width
    shifted = normalised - 0.5
    angle = shifted * CAMERA_HFOV_DEG
    return round(angle, 1)


def normalise_bbox(x1: int, y1: int, x2: int, y2: int, frame_width: int, frame_height: int) -> Tuple[float, float, float, float]:
    return x1 / frame_width, y1 / frame_height, x2 / frame_width, y2 / frame_height


def denormalise_bbox(nx1: float, ny1: float, nx2: float, ny2: float, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
    return int(nx1 * frame_width), int(ny1 * frame_height), int(nx2 * frame_width), int(ny2 * frame_height)


def bbox_area(x1: int, y1: int, x2: int, y2: int) -> int:
    return (x2 - x1) * (y2 - y1)


def boxes_overlap(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    if ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1:
        return False
    return True


def depth_in_region(depth_map: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    region = depth_map[y1:y2, x1:x2]
    flat = region.flatten()
    valid = flat[flat > 0]
    if len(valid) == 0:
        return 0.0
    return float(np.median(valid))


def mm_to_spoken(distance_mm: float) -> str:
    if distance_mm <= 0:
        return "unknown distance"
    if distance_mm < 1000:
        return f"{round(distance_mm / 10)} centimetres"
    return f"{round(distance_mm / 1000, 1)} metres"


def classify_urgency(distance_mm: float) -> str:
    if distance_mm <= 0: return "UNKNOWN"
    if distance_mm < DANGER_DIST_MM: return "DANGER"
    if distance_mm < WARNING_DIST_MM: return "WARNING"
    return "SAFE"


def crop_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1 = int(np.clip(x1, 0, w)), int(np.clip(y1, 0, h))
    x2, y2 = int(np.clip(x2, 0, w)), int(np.clip(y2, 0, h))
    return frame[y1:y2, x1:x2]


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def draw_overlay(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    COLOURS = {
        "DANGER":  (0, 0, 220),
        "WARNING": (0, 165, 255),
        "SAFE":    (0, 200, 80),
        "UNKNOWN": (120, 120, 120)
    }

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        colour = COLOURS.get(det.get("urgency", "UNKNOWN"), (120, 120, 120))
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        
        text = f"{det.get('label', '?')}  {mm_to_spoken(det.get('distance_mm', 0))}  {det.get('angle_deg', 0):+.1f}deg  [{det.get('urgency', '?')}]"
        cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    return frame


def get_timestamp_ms() -> float:
    return time.time() * 1000


class RateLimiter:
    def __init__(self, run_every: int):
        self.run_every = run_every
        self._counter = 0

    def should_run(self) -> bool:
        self._counter += 1
        return self._counter % self.run_every == 0

    def reset(self):
        self._counter = 0


class AlertCooldown:
    def __init__(self):
        self._last_alerted: Dict[str, float] = {}

    def can_alert(self, label: str) -> bool:
        now = time.time()
        if now - self._last_alerted.get(label, 0) >= ALERT_COOLDOWN_SEC:
            self._last_alerted[label] = now
            return True
        return False

    def reset(self, label: Optional[str] = None):
        if label:
            self._last_alerted.pop(label, None)
        else:
            self._last_alerted.clear()