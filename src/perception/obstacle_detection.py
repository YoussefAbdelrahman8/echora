from ultralytics import YOLO
import numpy as np
import cv2
import threading
import time
import torch
from typing import List, Dict, Optional
import ollama

from src.core.config import settings, MODE
from src.core.utils import (
    logger,
    bbox_center,
    angle_from_x,
    classify_urgency,
    depth_in_region,
    RateLimiter,
)
from src.perception.kalman_tracker import KalmanTracker

VLM_PROMPT = (
    "You are an assistant for a blind person. "
    "Look at this image and describe in ONE short sentence "
    "the most important obstacle or hazard they should know about. "
    "Focus on objects in their direct path. Be very brief."
)
VLM_MODEL_NAME = "llava"

class ObstacleDetector:
    """
    Detects and tracks obstacles using YOLOv8 (Layer 1) and VLM for scene context (Layer 2).
    """

    def __init__(self):
        self._yolo: Optional[YOLO] = None
        self._tracker = KalmanTracker(frame_width=settings.CAMERA_RGB_WIDTH)
        self._det_limiter = RateLimiter(run_every=settings.OBSTACLE_RUN_EVERY)
        self._vlm_limiter = RateLimiter(run_every=settings.VLM_RUN_EVERY)
        self.latest_vlm_description: str = ""
        self._vlm_running: bool = False
        self._vlm_lock = threading.Lock()
        self._frame_count: int = 0
        self._last_result: Optional[Dict] = None
        self._device: str = "cpu"

    def load_model(self):
        if not settings.YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found at: {settings.YOLO_MODEL_PATH}")

        self._yolo = YOLO(str(settings.YOLO_MODEL_PATH))

        if torch.backends.mps.is_available():
            self._device = "mps"
            logger.info("YOLO will run on Apple MPS GPU.")
        elif torch.cuda.is_available():
            self._device = "cuda"
            logger.info("YOLO will run on CUDA GPU.")
        else:
            self._device = "cpu"
            logger.info("YOLO will run on CPU.")

        blank = np.zeros((settings.YOLO_INPUT_HEIGHT, settings.YOLO_INPUT_WIDTH, 3), dtype=np.uint8)
        for _ in range(3):
            self._yolo(blank, verbose=False, device=self._device, half=(self._device == "cuda"))
        logger.info("YOLO warmup complete.")

    def update(self, bundle: Dict) -> Dict:
        self._frame_count += 1
        rgb_frame = bundle["rgb"]
        depth_map = bundle["depth"]
        timestamp = bundle["timestamp_ms"]

        if self._det_limiter.should_run():
            confirmed_tracks = self.detect(rgb_frame, depth_map)
        else:
            confirmed_tracks = self._tracker.get_confirmed_tracks()

        if self._vlm_limiter.should_run() and not self._vlm_running:
            self._start_vlm_thread(rgb_frame.copy())

        danger_tracks  = [t for t in confirmed_tracks if t["urgency"] == "DANGER"]
        warning_tracks = [t for t in confirmed_tracks if t["urgency"] == "WARNING"]
        safe_tracks    = [t for t in confirmed_tracks if t["urgency"] == "SAFE"]

        with self._vlm_lock:
            scene_desc = self.latest_vlm_description

        result = {
            "tracks":       confirmed_tracks,
            "danger":       danger_tracks,
            "warning":      warning_tracks,
            "safe":         safe_tracks,
            "scene_desc":   scene_desc,
            "frame_count":  self._frame_count,
            "timestamp_ms": timestamp,
        }

        self._last_result = result
        return result

    def detect(self, rgb_frame: np.ndarray, depth_map: np.ndarray) -> List[Dict]:
        raw_detections = self._run_yolo(rgb_frame)
        if not raw_detections:
            return self._tracker.update([])

        detections_with_depth = []
        for det in raw_detections:
            x1, y1, x2, y2 = det["bbox"]
            distance_mm = self._get_depth_for_detection(depth_map, x1, y1, x2, y2)
            det["distance_mm"] = distance_mm

            cx, _ = bbox_center(x1, y1, x2, y2)
            det["angle_deg"] = angle_from_x(cx, rgb_frame.shape[1])
            det["urgency"] = classify_urgency(distance_mm)
            detections_with_depth.append(det)

        filtered = self._filter_detections(detections_with_depth)
        return self._tracker.update(filtered)

    def _run_yolo(self, rgb_frame: np.ndarray) -> List[Dict]:
        if self._yolo is None:
            return []

        try:
            results = self._yolo(
                rgb_frame,
                verbose=False,
                conf=settings.DETECTION_CONFIDENCE_THRESHOLD,
                device=self._device,
                half=(self._device == "cuda"),
                imgsz=settings.YOLO_INPUT_WIDTH,
            )

            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                return []

            detections = []
            for box in result.boxes:
                xyxy = box.xyxy.cpu().numpy()[0]
                x1 = int(xyxy[0])
                y1 = int(xyxy[1])
                x2 = int(xyxy[2])
                y2 = int(xyxy[3])
                
                class_idx = int(box.cls.item())
                label = self._yolo.names[class_idx]
                detections.append({
                    "label": label,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(box.conf.item()),
                })

            return detections
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return []

    def _get_depth_for_detection(self, depth_map: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        w, h = x2 - x1, y2 - y1
        margin_x = int(w * (1 - settings.DEPTH_BBOX_SCALE) / 2)
        margin_y = int(h * (1 - settings.DEPTH_BBOX_SCALE) / 2)

        sample_x1 = max(0, x1 + margin_x)
        sample_y1 = max(0, y1 + margin_y)
        sample_x2 = min(depth_map.shape[1] - 1, x2 - margin_x)
        sample_y2 = min(depth_map.shape[0] - 1, y2 - margin_y)

        return depth_in_region(depth_map, sample_x1, sample_y1, sample_x2, sample_y2)

    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        filtered = []
        for det in detections:
            if det["label"] not in settings.RELEVANT_CLASSES:
                continue

            if det["distance_mm"] <= settings.DEPTH_MIN_MM or det["distance_mm"] > settings.DEPTH_MAX_MM:
                continue

            if (det["urgency"] == "WARNING"
                    and abs(det.get("angle_deg", 0)) <= settings.COLLISION_CORRIDOR_DEG
                    and det["distance_mm"] < settings.DANGER_DIST_MM * 1.2):
                det["urgency"] = "DANGER"

            filtered.append(det)

        return filtered

    def _start_vlm_thread(self, rgb_frame: np.ndarray):
        self._vlm_running = True
        threading.Thread(target=self._vlm_worker, args=(rgb_frame,), daemon=True).start()

    def _vlm_worker(self, rgb_frame: np.ndarray):
        try:
            small_frame = cv2.resize(rgb_frame, (512, 320))
            success, encoded = cv2.imencode(".jpg", small_frame)
            if not success:
                return

            response = ollama.chat(
                model=VLM_MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": VLM_PROMPT,
                    "images": [encoded.tobytes()],
                }]
            )

            description = response["message"]["content"].strip()
            with self._vlm_lock:
                self.latest_vlm_description = description

        except Exception as e:
            logger.warning(f"VLM error (non-fatal): {e}")
        finally:
            self._vlm_running = False

    def get_scene_description(self) -> str:
        with self._vlm_lock:
            return self.latest_vlm_description

    def get_danger_zone_objects(self) -> List[Dict]:
        return self._last_result.get("danger", []) if self._last_result else []

    def get_warning_zone_objects(self) -> List[Dict]:
        return self._last_result.get("warning", []) if self._last_result else []

    def get_all_tracks(self) -> List[Dict]:
        return self._tracker.get_confirmed_tracks()

    def get_most_urgent_obstacle(self) -> Optional[Dict]:
        return self._tracker.get_most_urgent()

    def reset_tracker(self):
        self._tracker.reset()

    def get_stats(self) -> Dict:
        return {
            "frame_count":    self._frame_count,
            "vlm_running":    self._vlm_running,
            "tracker":        self._tracker.get_stats(),
            "last_scene_desc": self.latest_vlm_description[:80] + "..." if len(self.latest_vlm_description) > 80 else self.latest_vlm_description,
        }