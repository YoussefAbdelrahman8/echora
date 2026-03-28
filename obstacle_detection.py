# =============================================================================
# obstacle_detection.py — ECHORA Obstacle Detection + Scene Understanding
# =============================================================================
# Two layers:
#   Layer 1 — YOLO (runs every frame, fast, finds objects + distances)
#   Layer 2 — VLM  (runs every 30 frames, slow, understands the scene)
#
# Produces a clean result package every frame consumed by control_unit.py.
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

# YOLO from the ultralytics library.
# ultralytics is the official Python library for YOLOv8 models.
# Install with: pip install ultralytics
from ultralytics import YOLO

# numpy for array operations on frames and depth maps.
import numpy as np

# cv2 for image resizing before feeding into YOLO.
import cv2

# threading to run the VLM in the background without blocking the main loop.
import threading

# time for timestamps and rate limiting.
import time

# Type hints for cleaner, more readable code.
from typing import List, Dict, Optional, Tuple

# Our own modules — everything built so far.
from config import (
COLLISION_CORRIDOR_DEG,
    CAMERA_RGB_WIDTH,
    CAMERA_RGB_HEIGHT,
    DETECTION_CONFIDENCE_THRESHOLD,
    RELEVANT_CLASSES,
    DEPTH_BBOX_SCALE,
    DEPTH_MIN_MM,
    DEPTH_MAX_MM,
    DANGER_DIST_MM,
    WARNING_DIST_MM,
    YOLO_MODEL_PATH,
    VLM_RUN_EVERY,
    OBSTACLE_RUN_EVERY,
    YOLO_INPUT_WIDTH,
    YOLO_INPUT_HEIGHT,
)
from utils import (
    logger,
    bbox_center,
    angle_from_x,
    classify_urgency,
    depth_in_region,
    denormalise_bbox,
    get_timestamp_ms,
    RateLimiter,
    crop_region,
)
from kalman_tracker import KalmanTracker


# =============================================================================
# VLM CONFIGURATION
# =============================================================================
# We use a small vision-language model to understand the scene contextually.
# The VLM receives the RGB frame and a text prompt, and returns a plain
# English description of what the blind user should know.
#
# We use the 'ollama' library to run a local VLM on the laptop.
# This means NO internet connection required — everything runs offline.
# Install ollama: https://ollama.com  then run: ollama pull llava
#
# If ollama is not available, the system falls back to a simple
# "scene description not available" message and keeps running.
# This means the VLM is OPTIONAL — the core safety system always works.

# The text prompt sent to the VLM every time it runs.
# Keep it short and specific — longer prompts = slower responses.
VLM_PROMPT = (
    "You are an assistant for a blind person. "
    "Look at this image and describe in ONE short sentence "
    "the most important obstacle or hazard they should know about. "
    "Focus on objects in their direct path. Be very brief."
)

# The name of the local VLM model to use via ollama.
# 'llava' is a lightweight vision-language model that runs on a laptop CPU.
# Alternative: 'llava:13b' for better quality but slower speed.
VLM_MODEL_NAME = "llava"


# =============================================================================
# OBSTACLE DETECTOR CLASS
# =============================================================================

class ObstacleDetector:
    """
    Detects and tracks obstacles for ECHORA.

    Layer 1 — YOLO: runs every frame.
      - Detects objects in the RGB frame.
      - Combines bounding boxes with depth map to get 3D distances.
      - Filters to only RELEVANT_CLASSES from config.py.
      - Passes filtered detections to KalmanTracker for smooth tracking.

    Layer 2 — VLM: runs every VLM_RUN_EVERY frames in a background thread.
      - Sends the RGB frame to a local vision-language model.
      - Receives a plain English scene description.
      - Stores it for control_unit.py to speak aloud.

    Usage:
        detector = ObstacleDetector()
        detector.load_model()

        while True:
            bundle = cam.get_synced_bundle()
            result = detector.update(bundle)

            for track in result["danger"]:
                print(f"DANGER: {track['label']} at {track['distance_mm']}mm")
    """

    def __init__(self):
        """
        Creates the detector. Does NOT load the model yet.
        Call load_model() separately after creating this object.

        Why separate? Loading the YOLO model takes a few seconds.
        Separating it allows better error handling and startup logging.
        """

        # The YOLO model object — None until load_model() runs.
        self._yolo: Optional[YOLO] = None

        # KalmanTracker — manages one Kalman filter per detected object.
        # Initialised with the camera frame width for angle calculations.
        self._tracker = KalmanTracker(frame_width=CAMERA_RGB_WIDTH)

        # RateLimiter for the main detection loop.
        # OBSTACLE_RUN_EVERY = 1 means run every single frame.
        self._det_limiter = RateLimiter(run_every=OBSTACLE_RUN_EVERY)

        # RateLimiter for the VLM — runs much less frequently.
        # VLM_RUN_EVERY = 30 means run once every 30 frames (~1 per second).
        self._vlm_limiter = RateLimiter(run_every=VLM_RUN_EVERY)

        # Stores the latest VLM scene description string.
        # Updated by the background VLM thread.
        # Initialised to empty string — no description yet.
        self.latest_vlm_description: str = ""

        # Flag that tracks whether the VLM is currently running.
        # Prevents two VLM threads from running at the same time.
        self._vlm_running: bool = False

        # threading.Lock for safe access to latest_vlm_description.
        # The VLM thread writes to it; the main thread reads from it.
        self._vlm_lock = threading.Lock()

        # Frame counter — incremented every time update() is called.
        # Used for rate limiting and diagnostic logging.
        self._frame_count: int = 0

        # Whether the VLM (ollama) is available on this machine.
        # Set to False if ollama is not installed — VLM is optional.
        self._vlm_available: bool = False

        # The last result package returned by update().
        # Stored so get_confirmed_tracks() can return it between updates.
        self._last_result: Optional[Dict] = None

        logger.info("ObstacleDetector created. Call load_model() to start.")


    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def load_model(self):

        logger.info(f"Loading YOLO model from: {YOLO_MODEL_PATH}")

        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO model not found at: {YOLO_MODEL_PATH}"
            )

        self._yolo = YOLO(str(YOLO_MODEL_PATH))

        # ── NEW: Force YOLO to use the best available device ──────────────────
        import torch

        if torch.backends.mps.is_available():
            # Apple Silicon GPU — much faster than CPU
            self._device = "mps"
            logger.info("YOLO will run on Apple MPS GPU.")
        elif torch.cuda.is_available():
            # Nvidia GPU — fastest option
            self._device = "cuda"
            logger.info("YOLO will run on CUDA GPU.")
        else:
            # Fallback to CPU
            self._device = "cpu"
            logger.info("YOLO will run on CPU (no GPU available).")
        # ─────────────────────────────────────────────────────────────────────

        logger.info("YOLO model loaded successfully.")
        logger.info(f"Model classes: {len(self._yolo.names)} total")
        # ... rest of load_model stays the same


    # =========================================================================
    # MAIN UPDATE FUNCTION
    # =========================================================================

    def update(self, bundle: Dict) -> Dict:
        """
        Top-level function — call this every frame from control_unit.py.

        Takes the camera bundle from camera.py and returns a full result
        package containing tracked obstacles, urgency levels, and the
        latest VLM scene description.

        Arguments:
            bundle: the dictionary from cam.get_synced_bundle()
                    Must contain: "rgb", "depth", "timestamp_ms"

        Returns:
            {
              "tracks":       list of all confirmed track dicts,
              "danger":       subset of tracks in DANGER zone,
              "warning":      subset of tracks in WARNING zone,
              "safe":         subset of tracks in SAFE zone,
              "scene_desc":   latest VLM description string,
              "frame_count":  int,
              "timestamp_ms": float,
            }
        """

        # Increment our internal frame counter.
        self._frame_count += 1

        # Extract the RGB frame and depth map from the bundle.
        rgb_frame = bundle["rgb"]
        depth_map = bundle["depth"]
        timestamp = bundle["timestamp_ms"]

        # ── Run YOLO detection ─────────────────────────────────────────────────
        # _det_limiter.should_run() returns True every OBSTACLE_RUN_EVERY frames.
        # Since OBSTACLE_RUN_EVERY = 1, this is True every frame.
        # We keep the limiter here anyway for flexibility — you could change
        # OBSTACLE_RUN_EVERY to 2 to halve the CPU load if needed.
        if self._det_limiter.should_run():
            confirmed_tracks = self.detect(rgb_frame, depth_map)
        else:
            # If we're not running YOLO this frame, return the last known tracks.
            confirmed_tracks = self._tracker.get_confirmed_tracks()

        # ── Trigger VLM in background ──────────────────────────────────────────
        # _vlm_limiter.should_run() returns True every VLM_RUN_EVERY frames.
        # _vlm_running is False when no VLM thread is currently active.
        # Both must be true to start a new VLM analysis.
        if self._vlm_limiter.should_run() and not self._vlm_running:
            self._start_vlm_thread(rgb_frame.copy())
            # .copy() is important — we make a copy of the frame so that
            # by the time the slow VLM thread reads it, the main loop
            # hasn't already overwritten it with a newer frame.

        # ── Split tracks by urgency ────────────────────────────────────────────
        # Sort confirmed tracks into three urgency buckets.
        # This makes it easy for control_unit.py to handle each level.
        danger_tracks  = [t for t in confirmed_tracks if t["urgency"] == "DANGER"]
        warning_tracks = [t for t in confirmed_tracks if t["urgency"] == "WARNING"]
        safe_tracks    = [t for t in confirmed_tracks if t["urgency"] == "SAFE"]

        # ── Get latest VLM description ─────────────────────────────────────────
        # Read the latest VLM result safely using the lock.
        with self._vlm_lock:
            scene_desc = self.latest_vlm_description

        # ── Build and return the result package ───────────────────────────────
        result = {
            "tracks":       confirmed_tracks,
            "danger":       danger_tracks,
            "warning":      warning_tracks,
            "safe":         safe_tracks,
            "scene_desc":   scene_desc,
            "frame_count":  self._frame_count,
            "timestamp_ms": timestamp,
        }

        # Cache the result for external callers.
        self._last_result = result

        return result


    # =========================================================================
    # DETECTION PIPELINE
    # =========================================================================

    def detect(self, rgb_frame: np.ndarray, depth_map: np.ndarray) -> List[Dict]:
        """
        Runs the full detection pipeline for one frame.

        Steps:
          1. Run YOLO on the RGB frame → raw bounding boxes + labels
          2. For each detection: get depth from depth map
          3. Filter: remove irrelevant classes and low-confidence detections
          4. Update KalmanTracker → get smoothed, confirmed tracks
          5. Return confirmed track dictionaries

        Arguments:
            rgb_frame:  numpy array (H, W, 3) uint8 — the colour frame
            depth_map:  numpy array (H, W) uint16 — depth in mm per pixel

        Returns:
            List of confirmed track dictionaries from KalmanTracker.
        """

        # ── Step 1: Run YOLO ───────────────────────────────────────────────────
        raw_detections = self._run_yolo(rgb_frame)

        # If YOLO found nothing, still update the tracker with an empty list.
        # This tells the tracker that ALL existing tracks were missed this frame.
        if not raw_detections:
            return self._tracker.update([])

        # ── Step 2: Add depth to each detection ───────────────────────────────
        detections_with_depth = []

        for det in raw_detections:
            # Get the bounding box coordinates.
            x1, y1, x2, y2 = det["bbox"]

            # Get the median depth value inside this bounding box.
            # We use the center region of the box (DEPTH_BBOX_SCALE = 0.5)
            # to avoid edge pixels which often have noisy depth readings.
            distance_mm = self._get_depth_for_detection(
                depth_map, x1, y1, x2, y2
            )

            # Add the depth to this detection dictionary.
            det["distance_mm"] = distance_mm

            # Calculate the horizontal angle of this object.
            cx, _ = bbox_center(x1, y1, x2, y2)
            det["angle_deg"] = angle_from_x(cx, rgb_frame.shape[1])

            # Classify urgency based on distance.
            det["urgency"] = classify_urgency(distance_mm)

            detections_with_depth.append(det)

        # ── Step 3: Filter detections ──────────────────────────────────────────
        filtered = self._filter_detections(detections_with_depth)

        # ── Step 4: Update KalmanTracker ──────────────────────────────────────
        # Pass the filtered detections to the tracker.
        # The tracker matches them to existing tracks, updates Kalman filters,
        # creates new tracks, and removes old ones.
        # Returns only CONFIRMED tracks.
        confirmed_tracks = self._tracker.update(filtered)

        return confirmed_tracks


    # =========================================================================
    # YOLO INFERENCE
    # =========================================================================

    def _run_yolo(self, rgb_frame: np.ndarray) -> List[Dict]:

        if self._yolo is None:
            return []

        try:
            # ── Resize for faster inference ────────────────────────────────────
            yolo_frame = cv2.resize(
                rgb_frame,
                (YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT)
            )
            # ──────────────────────────────────────────────────────────────────

            results = self._yolo(
                yolo_frame,  # ← use resized frame
                verbose=False,
                conf=DETECTION_CONFIDENCE_THRESHOLD,
                device=self._device
            )

            # After detection, scale bounding boxes back to original frame size.
            # YOLO returns coordinates relative to the input size (640×400).
            # We need them relative to the display size (1280×800).
            scale_x = rgb_frame.shape[1] / YOLO_INPUT_WIDTH
            scale_y = rgb_frame.shape[0] / YOLO_INPUT_HEIGHT

            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                return []

            detections = []

            for box in result.boxes:
                xyxy = box.xyxy.cpu().numpy()[0]

                # Scale coordinates back to original frame size
                x1 = int(xyxy[0] * scale_x)
                y1 = int(xyxy[1] * scale_y)
                x2 = int(xyxy[2] * scale_x)
                y2 = int(xyxy[3] * scale_y)

                confidence = float(box.conf.item())
                class_idx = int(box.cls.item())
                label = self._yolo.names[class_idx]

                detections.append({
                    "label": label,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                })

            return detections

        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return []

    # =========================================================================
    # DEPTH EXTRACTION
    # =========================================================================

    def _get_depth_for_detection(
        self,
        depth_map: np.ndarray,
        x1: int, y1: int, x2: int, y2: int
    ) -> float:
        """
        Gets the reliable depth reading for a detected object's bounding box.

        We don't use the full bounding box for depth — the edges of bounding
        boxes often overlap with background pixels which have very different
        depth values. Instead we shrink the box by DEPTH_BBOX_SCALE (0.5)
        and only sample the central region.

        Example with DEPTH_BBOX_SCALE = 0.5:
          Original box: x1=100, y1=100, x2=300, y2=300 (200×200 pixels)
          Shrunk box:   x1=150, y1=150, x2=250, y2=250 (100×100 pixels, center)

        Arguments:
            depth_map: numpy array (H, W) uint16, values in mm
            x1, y1, x2, y2: bounding box pixel coordinates

        Returns:
            Median depth in mm, or 0.0 if no valid data.
        """

        # Calculate the width and height of the bounding box.
        w = x2 - x1
        h = y2 - y1

        # Calculate how much to shrink the box on each side.
        # DEPTH_BBOX_SCALE = 0.5 means we keep the center 50%.
        # margin_x = how many pixels to shrink on left AND right.
        # margin_y = how many pixels to shrink on top AND bottom.
        margin_x = int(w * (1 - DEPTH_BBOX_SCALE) / 2)
        margin_y = int(h * (1 - DEPTH_BBOX_SCALE) / 2)

        # Apply the margins to get the shrunk bounding box.
        # max(0, ...) prevents going below pixel 0 (outside the frame).
        sample_x1 = max(0, x1 + margin_x)
        sample_y1 = max(0, y1 + margin_y)
        sample_x2 = min(depth_map.shape[1] - 1, x2 - margin_x)
        sample_y2 = min(depth_map.shape[0] - 1, y2 - margin_y)

        # Use depth_in_region from utils.py to get the median depth.
        # This function already handles zero pixels and returns 0.0
        # if no valid depth data exists in the region.
        distance = depth_in_region(
            depth_map, sample_x1, sample_y1, sample_x2, sample_y2
        )

        return distance


    # =========================================================================
    # DETECTION FILTERING
    # =========================================================================

    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Filters raw detections to keep only the ones ECHORA cares about.

        Removes:
          1. Objects not in RELEVANT_CLASSES (irrelevant for navigation)
          2. Objects below DETECTION_CONFIDENCE_THRESHOLD (too uncertain)
          3. Objects with 0 depth (no valid depth reading — can't judge danger)
          4. Objects beyond DEPTH_MAX_MM (too far to be relevant)

        Arguments:
            detections: list of detection dicts with depth already added

        Returns:
            Filtered list — only relevant, reliable detections.
        """

        filtered = []

        for det in detections:

            # ── Filter 1: Relevance ────────────────────────────────────────────
            # Only keep objects in our RELEVANT_CLASSES list from config.py.
            # This removes frisbees, kites, surfboards, snowboards, etc.
            if det["label"] not in RELEVANT_CLASSES:
                logger.debug(f"Filtered out irrelevant class: {det['label']}")
                continue   # 'continue' skips to the next detection

            # ── Filter 2: Confidence ───────────────────────────────────────────
            # Only keep detections where YOLO is confident enough.
            # Below threshold = probably a false positive.
            if det["confidence"] < DETECTION_CONFIDENCE_THRESHOLD:
                logger.debug(
                    f"Filtered out low-confidence detection: "
                    f"{det['label']} ({det['confidence']:.2f})"
                )
                continue

            # ── Filter 3: Valid depth ──────────────────────────────────────────
            # depth_mm = 0 means the depth sensor got no valid reading here.
            # This usually means the object is too shiny, too dark, or too
            # close/far for the stereo sensor. We can't judge danger without
            # knowing the distance, so we skip these.
            if det["distance_mm"] <= 0:
                logger.debug(
                    f"Filtered out no-depth detection: {det['label']}"
                )
                continue

            # ── Filter 4: Range ────────────────────────────────────────────────
            # Objects beyond DEPTH_MAX_MM are too far to be an immediate concern.
            # We report them but at much lower urgency — actually they would
            # already be classified SAFE by classify_urgency(), but we also
            # filter out extreme values that are sensor noise.
            if det["distance_mm"] > DEPTH_MAX_MM:
                logger.debug(
                    f"Filtered out out-of-range detection: "
                    f"{det['label']} at {det['distance_mm']:.0f}mm"
                )
                continue

            # Boost urgency for objects in the direct collision corridor.
            # At face level, objects straight ahead are the real danger.
            if (det["urgency"] == "WARNING"
                    and abs(det.get("angle_deg", 0)) <= COLLISION_CORRIDOR_DEG
                    and det["distance_mm"] < DANGER_DIST_MM * 1.2):
                # Object is close AND straight ahead — escalate to DANGER.
                det["urgency"] = "DANGER"
                logger.debug(
                    f"Urgency escalated to DANGER: {det['label']} is in collision corridor"
                )

            filtered.append(det)
        # Log how many detections survived filtering.
        logger.debug(
            f"Filtering: {len(detections)} raw → {len(filtered)} kept"
        )

        return filtered


    # =========================================================================
    # VLM SCENE UNDERSTANDING
    # =========================================================================

    def _start_vlm_thread(self, rgb_frame: np.ndarray):
        """
        Starts a background thread to run the VLM on the current frame.

        We pass a COPY of the frame (made in update()) so the thread
        works on a stable snapshot while the main loop keeps running.

        Arguments:
            rgb_frame: a copy of the current RGB frame (numpy array)
        """

        # Mark VLM as running so we don't start a second one.
        self._vlm_running = True

        # Create a new thread targeting our _vlm_worker function.
        # args=(rgb_frame,) passes the frame as an argument to the worker.
        # daemon=True means the thread auto-stops when main program exits.
        vlm_thread = threading.Thread(
            target=self._vlm_worker,
            args=(rgb_frame,),
            daemon=True
        )

        # .start() launches the thread — it runs in the background immediately.
        vlm_thread.start()


    def _vlm_worker(self, rgb_frame: np.ndarray):
        """
        Background thread function — runs the VLM on one frame.

        This runs in a separate thread so it never blocks the main loop.
        When finished, it updates self.latest_vlm_description.

        Arguments:
            rgb_frame: numpy array (H, W, 3) uint8
        """

        try:
            # Only proceed if ollama is available.
            if not self._vlm_available:
                return

            import ollama

            # ── Resize the frame before sending to VLM ─────────────────────────
            # VLMs work fine with smaller images and smaller = faster.
            # 512×320 is enough detail for scene understanding.
            # cv2.resize(frame, (width, height)) — note: width before height.
            small_frame = cv2.resize(rgb_frame, (512, 320))

            # ── Encode the frame as JPEG bytes ─────────────────────────────────
            # ollama expects image data as bytes, not a numpy array.
            # cv2.imencode encodes the array as JPEG format.
            # [1] gets the encoded byte array (index 0 is a success flag).
            # .tobytes() converts numpy byte array to Python bytes object.
            success, encoded = cv2.imencode(".jpg", small_frame)

            if not success:
                logger.warning("VLM: Failed to encode frame as JPEG.")
                return

            image_bytes = encoded.tobytes()

            # ── Send to VLM via ollama ──────────────────────────────────────────
            # ollama.chat() sends a message with an attached image.
            # model=    which local model to use (configured at top of file)
            # messages= list of message dicts, like a chat conversation
            # The "images" key attaches the frame to the message.
            logger.debug("VLM: Sending frame to ollama...")
            start_time = time.time()

            response = ollama.chat(
                model=VLM_MODEL_NAME,
                messages=[
                    {
                        "role":    "user",
                        "content": VLM_PROMPT,
                        "images":  [image_bytes],
                    }
                ]
            )

            elapsed = round((time.time() - start_time) * 1000)

            # ── Extract the text response ───────────────────────────────────────
            # response is a dict. The model's reply is in:
            # response["message"]["content"]
            description = response["message"]["content"].strip()

            logger.debug(f"VLM response ({elapsed}ms): {description}")

            # ── Store the result safely ─────────────────────────────────────────
            # Acquire the lock before writing to the shared variable.
            # This prevents the main thread from reading a half-written string.
            with self._vlm_lock:
                self.latest_vlm_description = description

        except Exception as e:
            # Any error (model not running, timeout, etc.) is logged but
            # does NOT crash the system. VLM is always optional.
            logger.warning(f"VLM error (non-fatal): {e}")

        finally:
            # Always mark VLM as finished — even if it crashed.
            # 'finally' runs whether the try block succeeded or failed.
            # Without this, _vlm_running would stay True forever after a crash,
            # and no future VLM analysis would ever run.
            self._vlm_running = False


    # =========================================================================
    # CONVENIENCE ACCESSORS
    # =========================================================================

    def get_scene_description(self) -> str:
        """
        Returns the latest VLM scene description.

        Thread-safe — handles the lock internally.
        Returns empty string if no description is available yet.
        """
        with self._vlm_lock:
            return self.latest_vlm_description


    def get_danger_zone_objects(self) -> List[Dict]:
        """
        Returns only the confirmed tracks currently in the DANGER zone.

        These are objects closer than DANGER_DIST_MM (800mm by default).
        Used by control_unit.py to trigger immediate haptic + audio alerts.
        """

        if self._last_result is None:
            return []

        # Return the "danger" key from the last result package.
        return self._last_result.get("danger", [])


    def get_warning_zone_objects(self) -> List[Dict]:
        """
        Returns only the confirmed tracks currently in the WARNING zone.

        These are objects between DANGER_DIST_MM and WARNING_DIST_MM.
        Used by control_unit.py for gentler audio warnings.
        """

        if self._last_result is None:
            return []

        return self._last_result.get("warning", [])


    def get_all_tracks(self) -> List[Dict]:
        """
        Returns all confirmed tracks regardless of urgency level.

        Used by control_unit.py when it needs the full picture,
        for example to decide which mode to switch to.
        """

        return self._tracker.get_confirmed_tracks()


    def get_most_urgent_obstacle(self) -> Optional[Dict]:
        """
        Returns the single most dangerous confirmed track.

        Delegates to KalmanTracker.get_most_urgent() which sorts by
        urgency level first, then by distance.

        Returns None if there are no confirmed tracks.
        """

        return self._tracker.get_most_urgent()


    def reset_tracker(self):
        """
        Clears all tracked objects.

        Call this when the system switches modes — for example when
        switching from NAVIGATION to OCR mode, obstacle tracks become
        irrelevant.
        """

        self._tracker.reset()
        logger.info("Obstacle tracker reset.")


    def get_stats(self) -> Dict:
        """
        Returns diagnostic statistics.

        Useful for performance monitoring and debugging.
        """

        tracker_stats = self._tracker.get_stats()

        return {
            "frame_count":    self._frame_count,
            "vlm_available":  self._vlm_available,
            "vlm_running":    self._vlm_running,
            "tracker":        tracker_stats,
            "last_scene_desc": self.latest_vlm_description[:80] + "..."
                               if len(self.latest_vlm_description) > 80
                               else self.latest_vlm_description,
        }


# =============================================================================
# SELF-TEST
# =============================================================================
# Run directly to test the detector without needing the full ECHORA pipeline.
#
# This test uses a REAL camera frame from camera.py but simulates
# the main loop manually.
#
# Requirements:
#   - OAK-D camera connected
#   - YOLO model at the path specified in config.py
#
# Run with:
#   python obstacle_detection.py

if __name__ == "__main__":

    print("=== ECHORA obstacle_detection.py self-test ===\n")

    # Import camera here (only needed for the self-test).
    from camera import EchoraCamera

    # ── Initialise camera ────────────────────────────────────────────────────
    cam = EchoraCamera()
    cam.init_pipeline()

    # ── Initialise detector ───────────────────────────────────────────────────
    detector = ObstacleDetector()

    # load_model() will raise FileNotFoundError if model doesn't exist.
    # The message will tell you exactly where to put the model file.
    detector.load_model()

    print("Camera and detector ready. Running for 100 frames...\n")
    print(f"{'Frame':>6}  {'Tracks':>6}  {'Danger':>6}  {'Warning':>7}  "
          f"{'Most urgent':<40}")
    print("-" * 80)

    frame_count = 0

    try:
        while cam.pipeline.isRunning() and frame_count < 100:

            # Get the synchronised camera bundle.
            bundle = cam.get_synced_bundle()
            if bundle is None:
                continue

            frame_count += 1

            # Run the full detection pipeline.
            result = detector.update(bundle)

            # Get the most urgent obstacle for display.
            most_urgent = detector.get_most_urgent_obstacle()
            urgent_str  = (
                f"{most_urgent['label']} "
                f"{most_urgent['distance_mm']:.0f}mm "
                f"[{most_urgent['urgency']}]"
                if most_urgent else "none"
            )

            # Print a summary line every frame.
            print(
                f"{result['frame_count']:>6}  "
                f"{len(result['tracks']):>6}  "
                f"{len(result['danger']):>6}  "
                f"{len(result['warning']):>7}  "
                f"{urgent_str:<40}"
            )

            # Show the VLM description when it updates.
            if result["scene_desc"]:
                print(f"  VLM: {result['scene_desc']}")

            # Draw detections on the frame and show it.
            frame = bundle["rgb"].copy()
            for track in result["tracks"]:
                x1, y1, x2, y2 = track["bbox"]
                colour = {
                    "DANGER":  (0, 0, 220),
                    "WARNING": (0, 165, 255),
                    "SAFE":    (0, 200, 80),
                }.get(track["urgency"], (120, 120, 120))

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    frame,
                    f"{track['label']} {track['distance_mm']:.0f}mm",
                    (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, colour, 1
                )

            cv2.imshow("ECHORA obstacle detection", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        cv2.destroyAllWindows()
        cam.release()

        # Print final stats.
        print("\n--- Final stats ---")
        stats = detector.get_stats()
        for key, val in stats.items():
            print(f"  {key}: {val}")

        print("\n=== Self-test complete ===")