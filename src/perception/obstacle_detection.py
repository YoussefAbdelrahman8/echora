import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.core.config import settings
from src.core.utils import (
    angle_from_x,
    bbox_center,
    classify_urgency,
    depth_in_region,
    logger,
    RateLimiter,
)
from src.perception.byte_tracker import ByteTracker

VLM_PROMPT = (
    "You provide spoken scene awareness for a blind user. "
    "Describe only what is clearly visible directly ahead, using one or two "
    "short, natural sentences (maximum 25 words). "
    "Prioritize useful navigation context: an open path, doorway, stairs, "
    "nearby people, large furniture, or a notable landmark. "
    "Use simple everyday words and say left, right, or ahead only when clear. "
    "Do not mention the image, the camera, uncertainty, or your role. "
    "Do not guess or invent details."
)
VLM_MODEL_NAME = "gemma4:e2b"


class ObstacleDetector:
    """
    Two-layer obstacle pipeline:
      Layer 1 — YOLOv8 + ByteTrack:  fast per-frame detection and tracking.
      Layer 2 — VLM (Gemma / Ollama): periodic scene description for context.
    """

    def __init__(self):
        self._yolo: Optional[YOLO] = None
        self._coco_yolo: Optional[YOLO] = None
        self._tracker = ByteTracker(frame_width=settings.CAMERA_RGB_WIDTH)
        self._det_limiter = RateLimiter(run_every=settings.OBSTACLE_RUN_EVERY)
        self._vlm_limiter = RateLimiter(run_every=settings.VLM_RUN_EVERY)

        self.latest_vlm_description: str = ""
        self._vlm_running: bool = False
        self._vlm_lock = threading.Lock()

        self._frame_count: int = 0
        self._last_result: Optional[Dict] = None
        self._last_confirmed: List[Dict] = []
        self._device: str = "cpu"
        self._relevant_class_ids: Optional[List[int]] = None
        self._coco_relevant_class_ids: Optional[List[int]] = None
        self._last_inference_ms: Dict[str, float] = {}
        self._vlm_failures: int = 0
        self._vlm_disabled: bool = not settings.VLM_ENABLED

    # ── Lifecycle ─────────────────────────────────────────────────────

    def load_model(self):
        if not settings.YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found: {settings.YOLO_MODEL_PATH}")

        if torch.backends.mps.is_available():
            self._device = "mps"
            logger.info("YOLO running on Apple MPS.")
        elif torch.cuda.is_available():
            self._device = "cuda"
            logger.info("YOLO running on CUDA.")
        else:
            self._device = "cpu"
            logger.info("YOLO running on CPU.")

        self._yolo = YOLO(str(settings.YOLO_MODEL_PATH))
        self._relevant_class_ids = self._get_relevant_class_ids(self._yolo)

        if settings.COCO_YOLO_ENABLED:
            if not settings.COCO_YOLO_MODEL_PATH.exists():
                logger.warning(
                    "COCO YOLO model not found at %s; continuing with the "
                    "accessibility model only.", settings.COCO_YOLO_MODEL_PATH
                )
            else:
                self._coco_yolo = YOLO(str(settings.COCO_YOLO_MODEL_PATH))
                self._coco_relevant_class_ids = self._get_relevant_class_ids(
                    self._coco_yolo
                )

        supported_labels = set(self._yolo.names.values())
        if self._coco_yolo is not None:
            supported_labels.update(self._coco_yolo.names.values())
        unsupported_relevant = sorted(set(settings.RELEVANT_CLASSES) - supported_labels)
        if unsupported_relevant:
            logger.warning(
                "No loaded YOLO model supports these configured relevant "
                "classes: %s", ", ".join(unsupported_relevant)
            )

        blank = np.zeros(
            (settings.YOLO_INPUT_HEIGHT, settings.YOLO_INPUT_WIDTH, 3), dtype=np.uint8
        )
        models = [("accessibility", self._yolo)]
        if self._coco_yolo is not None:
            models.append(("coco", self._coco_yolo))
        for name, model in models:
            for _ in range(3):
                model(blank, verbose=False, device=self._device,
                      half=(self._device == "cuda"))
            logger.info("YOLO %s model warmup complete.", name)
        logger.info(
            "Dual YOLO ready. Accessibility relevant classes: %s; "
            "COCO relevant classes: %s",
            len(self._relevant_class_ids or []),
            len(self._coco_relevant_class_ids or []) if self._coco_yolo else 0,
        )

    # ── Main entry point ──────────────────────────────────────────────

    def update(self, bundle: Dict, allow_vlm: bool = True) -> Dict:
        """
        Run the obstacle pipeline for one frame.

        allow_vlm: when False, the periodic Gemma/Ollama scene-description call
        is suppressed. The scene description is only used in NAVIGATION, so the
        caller passes False in other modes (e.g. INTERACTION) to free the GPU.
        """
        self._frame_count += 1
        rgb_frame = bundle.get("rgb")
        depth_map = bundle.get("depth")
        timestamp = bundle.get("timestamp_ms", 0)
        if rgb_frame is None or depth_map is None:
            logger.warning("ObstacleDetector.update: incomplete bundle, skipping frame.")
            return self._last_result or {
                "tracks": [], "obstacle_tracks": [], "danger": [], "warning": [],
                "safe": [], "unknown": [], "scene_desc": "", "frame_count": self._frame_count,
                "timestamp_ms": timestamp,
            }

        if self._det_limiter.should_run():
            confirmed_tracks = self.detect(rgb_frame, depth_map)
        else:
            confirmed_tracks = self._tracker.get_confirmed_tracks()

        # Keep every confirmed detection for interaction guidance, but do not
        # expose interaction-only classes to the navigation/obstacle layer.
        # They must not create obstacle alerts, haptics, mode overrides, or
        # navigation bounding boxes.
        obstacle_tracks = self._get_user_facing_obstacle_tracks(confirmed_tracks)
        danger_tracks  = [t for t in obstacle_tracks if t["urgency"] == "DANGER"]
        warning_tracks = [t for t in obstacle_tracks if t["urgency"] == "WARNING"]
        safe_tracks    = [t for t in obstacle_tracks if t["urgency"] == "SAFE"]
        unknown_tracks = [t for t in obstacle_tracks if t["urgency"] == "UNKNOWN"]

        if (
            allow_vlm
            and not danger_tracks
            and not warning_tracks
            and self._should_run_vlm()
        ):
            self._start_vlm_thread(rgb_frame.copy())

        with self._vlm_lock:
            scene_desc = self.latest_vlm_description

        result = {
            # Full detector output, including interaction-only classes. This
            # is intentionally consumed by InteractionDetector.
            "tracks":       confirmed_tracks,
            # User-facing navigation output. Never contains interaction-only
            # classes and is the only set used for obstacle feedback/UI.
            "obstacle_tracks": obstacle_tracks,
            "danger":       danger_tracks,
            "warning":      warning_tracks,
            "safe":         safe_tracks,
            "unknown":      unknown_tracks,
            "scene_desc":   scene_desc,
            "frame_count":  self._frame_count,
            "timestamp_ms": timestamp,
        }
        self._last_result = result
        return result

    @staticmethod
    def _get_user_facing_obstacle_tracks(tracks: List[Dict]) -> List[Dict]:
        """Exclude classes reserved solely for the interaction feature."""
        interaction_only = set(settings.INTERACTABLE_CLASSES)
        return [track for track in tracks if track.get("label") not in interaction_only]

    # ── Detection pipeline ────────────────────────────────────────────

    def detect(self, rgb_frame: np.ndarray, depth_map: np.ndarray) -> List[Dict]:
        raw = self._run_yolo(
            self._yolo,
            self._relevant_class_ids,
            source="accessibility",
            track_id_offset=1_000_000,
            rgb_frame=rgb_frame,
        )
        if self._coco_yolo is not None:
            raw.extend(self._run_yolo(
                self._coco_yolo,
                self._coco_relevant_class_ids,
                source="coco",
                track_id_offset=2_000_000,
                rgb_frame=rgb_frame,
            ))
        if not raw:
            return self._tracker.update([])

        enriched = []
        for det in raw:
            x1, y1, x2, y2 = det["bbox"]
            distance_mm = self._get_depth(depth_map, x1, y1, x2, y2)
            cx, _       = bbox_center(x1, y1, x2, y2)

            det["distance_mm"] = distance_mm
            det["angle_deg"]   = angle_from_x(cx, rgb_frame.shape[1])
            det["urgency"]     = classify_urgency(distance_mm)
            det["frame_shape"]  = rgb_frame.shape[:2]
            enriched.append(det)

        filtered = self._filter_and_promote(enriched)
        confirmed = self._tracker.update(filtered)
        self._last_confirmed = confirmed
        return confirmed

    @staticmethod
    def _get_relevant_class_ids(model: Optional[YOLO]) -> List[int]:
        if model is None:
            return []
        return [
            idx
            for idx, name in model.names.items()
            if name in settings.RELEVANT_CLASSES
        ]

    def _run_yolo(
        self,
        model: Optional[YOLO],
        relevant_class_ids: Optional[List[int]],
        source: str,
        track_id_offset: int,
        rgb_frame: np.ndarray,
    ) -> List[Dict]:
        """
        Runs YOLO with ByteTrack enabled (persist=True keeps tracker state
        between calls). Returns raw detections with stable track IDs.
        """
        if model is None:
            return []
        started_at = time.perf_counter()
        try:
            results = model.track(
                rgb_frame,
                verbose=False,
                persist=True,
                tracker="bytetrack.yaml",
                conf=settings.DETECTION_CONFIDENCE_THRESHOLD,
                classes=relevant_class_ids or None,
                device=self._device,
                half=(self._device == "cuda"),
                imgsz=(settings.YOLO_INPUT_HEIGHT, settings.YOLO_INPUT_WIDTH),
            )

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return []

            ids = result.boxes.id  # tensor or None if tracker not yet warm
            detections = []

            for i, box in enumerate(result.boxes):
                xyxy = box.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = (
                    int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                )
                class_idx = int(box.cls.item())
                label     = model.names[class_idx]
                # Negative IDs mark unassigned detections (tracker still cold)
                raw_track_id = int(ids[i].item()) if ids is not None else -(i + 1)
                track_id = (
                    track_id_offset + raw_track_id
                    if raw_track_id >= 0
                    else -(track_id_offset + abs(raw_track_id))
                )

                detections.append({
                    "track_id":   track_id,
                    "source":     source,
                    "label":      label,
                    "bbox":       (x1, y1, x2, y2),
                    "confidence": float(box.conf.item()),
                })

            return detections

        except Exception as e:
            logger.error("%s YOLO inference error: %s", source, e)
            return []
        finally:
            self._last_inference_ms[source] = (
                time.perf_counter() - started_at
            ) * 1000.0

    def _get_depth(
        self,
        depth_map: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> float:
        w, h = x2 - x1, y2 - y1
        mx   = int(w * (1 - settings.DEPTH_BBOX_SCALE) / 2)
        my   = int(h * (1 - settings.DEPTH_BBOX_SCALE) / 2)
        return depth_in_region(
            depth_map,
            max(0, x1 + mx), max(0, y1 + my),
            min(depth_map.shape[1] - 1, x2 - mx),
            min(depth_map.shape[0] - 1, y2 - my),
        )

    def _bbox_area_ratio(self, det: Dict) -> float:
        frame_h, frame_w = det.get("frame_shape", (settings.CAMERA_RGB_HEIGHT, settings.CAMERA_RGB_WIDTH))
        x1, y1, x2, y2 = det["bbox"]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        return area / max(frame_w * frame_h, 1)

    def _overlaps_collision_corridor(self, det: Dict) -> bool:
        _, frame_w = det.get("frame_shape", (settings.CAMERA_RGB_HEIGHT, settings.CAMERA_RGB_WIDTH))
        half_fov = settings.CAMERA_HFOV_DEG / 2.0
        half_corridor_px = int((settings.COLLISION_CORRIDOR_DEG / max(half_fov, 1.0)) * (frame_w / 2.0))
        corridor_cx = frame_w // 2
        corridor_x1 = max(0, corridor_cx - half_corridor_px)
        corridor_x2 = min(frame_w, corridor_cx + half_corridor_px)
        x1, _, x2, _ = det["bbox"]
        return x1 <= corridor_x2 and x2 >= corridor_x1

    def _should_keep_unknown_depth(self, det: Dict) -> bool:
        return (
            self._overlaps_collision_corridor(det)
            or self._bbox_area_ratio(det) >= settings.UNKNOWN_OBSTACLE_MIN_AREA_RATIO
        )

    def _filter_and_promote(self, detections: List[Dict]) -> List[Dict]:
        """
        1. Drop irrelevant classes.
        2. Keep relevant detections with invalid depth as UNKNOWN when they
           overlap the walking corridor or are large in the image.
        3. Corridor promotion: a WARNING obstacle that overlaps the walking
           corridor and is nearly at danger distance is
           escalated to DANGER so the user gets an immediate alert.

        Because ByteTracker stores the full dict (including urgency), this
        promotion is preserved all the way through to the final output —
        fixing the silent bug present in the old Kalman pipeline.
        """
        out = []
        for det in detections:
            if det["label"] not in settings.RELEVANT_CLASSES:
                continue

            d = det["distance_mm"]
            if d <= settings.DEPTH_MIN_MM:
                if self._should_keep_unknown_depth(det):
                    det["urgency"] = "UNKNOWN"
                    det["distance_mm"] = 0.0
                    out.append(det)
                continue

            if d > settings.DEPTH_MAX_MM:
                continue

            if (
                det["urgency"] == "WARNING"
                and self._overlaps_collision_corridor(det)
                and d < settings.DANGER_DIST_MM * 1.2
            ):
                det["urgency"] = "DANGER"

            out.append(det)

        return out

    # ── VLM scene description ─────────────────────────────────────────

    def _should_run_vlm(self) -> bool:
        return (
            settings.VLM_ENABLED
            and not self._vlm_disabled
            and not self._vlm_running
            and self._vlm_limiter.should_run()
        )

    def _start_vlm_thread(self, rgb_frame: np.ndarray):
        self._vlm_running = True
        threading.Thread(
            target=self._vlm_worker, args=(rgb_frame,), daemon=True
        ).start()

    def _vlm_worker(self, rgb_frame: np.ndarray):
        try:
            import ollama
            small = cv2.resize(rgb_frame, (512, 320))
            ok, buf = cv2.imencode(".jpg", small)
            if not ok:
                return

            response = ollama.chat(
                model=settings.VLM_MODEL_NAME or VLM_MODEL_NAME,
                messages=[{
                    "role":    "user",
                    "content": VLM_PROMPT,
                    "images":  [buf.tobytes()],
                }],
            )

            # Support both new (attribute) and old (dict) ollama response formats
            try:
                description = response.message.content.strip()
            except AttributeError:
                description = response["message"]["content"].strip()

            with self._vlm_lock:
                self.latest_vlm_description = description
            self._vlm_failures = 0

        except Exception as e:
            self._vlm_failures += 1
            if self._vlm_failures >= settings.VLM_MAX_FAILURES:
                self._vlm_disabled = True
                logger.warning(
                    f"VLM disabled after {self._vlm_failures} failed call(s): {e}"
                )
            else:
                logger.warning(
                    f"VLM error (non-fatal, {self._vlm_failures}/"
                    f"{settings.VLM_MAX_FAILURES}): {e}"
                )
        finally:
            self._vlm_running = False

    # ── Convenience accessors ─────────────────────────────────────────

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
        confirmed = self._get_user_facing_obstacle_tracks(
            self._tracker.get_confirmed_tracks()
        )
        if not confirmed:
            return None
        priority = {"DANGER": 0, "WARNING": 1, "SAFE": 2, "UNKNOWN": 3}
        return min(
            confirmed,
            key=lambda t: (priority.get(t["urgency"], 3), t["distance_mm"]),
        )

    def reset_tracker(self):
        self._tracker.reset()

    def get_stats(self) -> Dict:
        desc = self.latest_vlm_description
        return {
            "frame_count":     self._frame_count,
            "vlm_running":     self._vlm_running,
            "vlm_disabled":    self._vlm_disabled,
            "vlm_failures":    self._vlm_failures,
            "inference_ms":    dict(self._last_inference_ms),
            "tracker":         self._tracker.get_stats(),
            "last_scene_desc": desc[:80] + "..." if len(desc) > 80 else desc,
        }
