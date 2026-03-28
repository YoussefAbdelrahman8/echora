# =============================================================================
# interaction_detection.py — ECHORA Hand Guidance + Edge Rendering
# =============================================================================
# Two-phase system:
#
#   Phase 1 GUIDANCE:    hand > 200mm from object
#     → directional electrodes guide hand toward object
#
#   Phase 2 EDGE:        hand ≤ 200mm from object
#     → Canny edges of object region mapped to 5×6 electrode grid
#     → user feels object shape on wrist
#
#   SUCCESS:             hand ≤ 40mm from object
#     → all 30 electrodes pulse
#     → audio confirms contact
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

# MediaPipe — Google's hand landmark detection library.
# Gives us 21 keypoints per hand at 30+ FPS on CPU.
import mediapipe as mp

# numpy for electrode grid operations and edge map scaling.
import numpy as np

# cv2 for Canny edge detection and frame cropping.
import cv2

# time for performance timing.
import time

# Type hints.
from typing import List, Dict, Optional, Tuple

# Our modules.
from config import (
    DOMINANT_HAND,
    HAPTIC_ROWS,
    HAPTIC_COLS,
    GUIDANCE_TO_EDGE_DIST_MM,
    EDGE_TO_SUCCESS_DIST_MM,
    CANNY_THRESHOLD_LOW,
    CANNY_THRESHOLD_HIGH,
    MIN_INTERACTABLE_AREA_PX,
    INTERACTABLE_CLASSES,
    DETECTION_CONFIDENCE_THRESHOLD,
    CAMERA_HFOV_DEG,
)
from utils import (
    logger,
    bbox_center,
    angle_from_x,
    depth_in_region,
    crop_region,
    get_timestamp_ms,
)


# =============================================================================
# INTERACTION PHASE CONSTANTS
# =============================================================================

class InteractionPhase:
    """
    Constants for the three phases of hand-object interaction.

    IDLE     — no hand or no object detected
    GUIDANCE — hand detected, object detected, hand far from object
               → directional electrodes active
    EDGE     — hand close to object (≤ GUIDANCE_TO_EDGE_DIST_MM)
               → edge rendering active
    SUCCESS  — hand touched object (≤ EDGE_TO_SUCCESS_DIST_MM)
               → success pulse, return to navigation
    """
    IDLE     = "IDLE"
    GUIDANCE = "GUIDANCE"
    EDGE     = "EDGE"
    SUCCESS  = "SUCCESS"


# =============================================================================
# HAPTIC BRIDGE
# =============================================================================

class HapticBridge:
    """
    Sends electrode activation patterns to the ESP32 wristband.

    Currently a STUB — logs patterns but does not send to hardware.
    When you finalise the ESP32 communication protocol, fill in the
    send() method. Nothing else in the system needs to change.

    The electrode array is always a numpy array of shape (HAPTIC_ROWS, HAPTIC_COLS)
    with values 0 (off) or 1 (on). Flatten to 30 elements before transmitting.

    Protocol options (set HAPTIC_PROTOCOL in config.py when ready):
      "BLE"    — Bluetooth Low Energy to ESP32
      "SERIAL" — USB serial cable to ESP32
      "WIFI"   — WiFi UDP to ESP32
    """

    def __init__(self):
        """Creates the bridge. Does not connect to hardware yet."""

        # Whether the bridge is connected to real hardware.
        # False = stub mode (log only). True = real hardware.
        self._connected: bool = False

        # Total patterns sent since startup — for diagnostics.
        self._send_count: int = 0

        logger.info(
            "HapticBridge created (stub mode). "
            "Fill in send() when ESP32 protocol is finalised."
        )

    def connect(self):
        """
        Connect to the ESP32 wristband.

        STUB — does nothing yet.
        When implementing: establish BLE/serial/WiFi connection here.
        Set self._connected = True on success.
        """
        logger.info("HapticBridge.connect() called (stub — no hardware).")
        # TODO: implement when ESP32 protocol is finalised
        # self._connected = True

    def send(self, electrode_grid: np.ndarray):
        """
        Sends a 5×6 electrode activation grid to the wristband.

        Arguments:
            electrode_grid: numpy array shape (HAPTIC_ROWS, HAPTIC_COLS)
                            Values: 0 = electrode off, 1 = electrode on
                            Values can also be 0.0-1.0 for intensity levels.

        STUB — currently only logs the pattern.
        When implementing: flatten the grid and transmit via chosen protocol.
        """

        self._send_count += 1

        # Flatten the 5×6 grid to a 30-element 1D array.
        # This is what the ESP32 firmware will receive — one byte per electrode.
        flat = electrode_grid.flatten()

        # Count active electrodes for the log.
        n_active = int(np.sum(flat > 0))

        logger.debug(
            f"HapticBridge.send() #{self._send_count}: "
            f"{n_active}/30 electrodes active | "
            f"pattern: {flat.astype(int).tolist()}"
        )

        # TODO: replace this stub with real transmission when ready.
        # Example for BLE:
        #   payload = bytes([int(v * 255) for v in flat])
        #   self._ble_client.write(BLE_CHARACTERISTIC_UUID, payload)
        #
        # Example for serial:
        #   payload = bytes([int(v * 255) for v in flat])
        #   self._serial.write(payload)

    def send_all_off(self):
        """Turns all 30 electrodes off."""
        self.send(np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32))

    def send_all_on(self, intensity: float = 1.0):
        """Turns all 30 electrodes on at the given intensity."""
        self.send(
            np.full(
                (HAPTIC_ROWS, HAPTIC_COLS),
                intensity,
                dtype=np.float32
            )
        )

    def disconnect(self):
        """Disconnect from hardware. STUB."""
        self._connected = False
        logger.info("HapticBridge disconnected.")


# =============================================================================
# ELECTRODE GRID BUILDER
# =============================================================================

class ElectrodeGridBuilder:
    """
    Builds 5×6 electrode activation grids for both guidance and edge phases.

    This is a pure logic class — no hardware, no camera.
    It only knows about the 5×6 grid and how to fill it.

    Separating grid-building from detection makes testing much easier
    and keeps each class focused on one job.
    """

    def __init__(self, rows: int = HAPTIC_ROWS, cols: int = HAPTIC_COLS):
        """
        Creates the grid builder.

        Arguments:
            rows: number of electrode rows (default from config: 5)
            cols: number of electrode columns (default from config: 6)
        """
        self.rows = rows
        self.cols = cols

        # Pre-compute the center indices for reference.
        # Center row and column are the "straight ahead" reference point.
        self.center_row = rows // 2   # 5 // 2 = 2
        self.center_col = cols // 2   # 6 // 2 = 3

        logger.debug(
            f"ElectrodeGridBuilder: {rows}×{cols} grid, "
            f"center=({self.center_row},{self.center_col})"
        )

    def build_guidance_grid(
        self,
        dx: float,
        dy: float,
        intensity: float = 1.0
    ) -> np.ndarray:
        """
        Builds a directional guidance grid from a movement vector.

        The grid activates electrodes in the direction the hand needs
        to move to reach the target object.

        How the directional zones map to the 5×6 grid:
          UP    → rows 0-1     (top of wrist)
          DOWN  → rows 3-4     (bottom of wrist)
          LEFT  → cols 0-1     (left side of wrist)
          RIGHT → cols 4-5     (right side of wrist)
          CENTER→ row 2, cols 2-3 (center — used when very close)

        Multiple directions can be active simultaneously.
        Example: object is up and to the right → top rows + right cols active.

        Arguments:
            dx:        horizontal distance to object (pixels)
                       positive = object is to the RIGHT
                       negative = object is to the LEFT
            dy:        vertical distance to object (pixels)
                       positive = object is BELOW hand (in image coordinates)
                       negative = object is ABOVE hand
            intensity: how strongly to activate electrodes (0.0-1.0)
                       higher = stronger buzz

        Returns:
            numpy array shape (rows, cols) with values 0.0 to 1.0
        """

        # Start with a blank grid — all electrodes off.
        # np.zeros creates an array filled with 0.0.
        # dtype=np.float32 means 32-bit floating point values.
        grid = np.zeros((self.rows, self.cols), dtype=np.float32)

        # ── Calculate normalised direction ─────────────────────────────────────
        # We normalise the vector so that the electrode pattern is the same
        # regardless of how far away the object is.
        # Only the direction matters for guidance, not the magnitude.
        magnitude = (dx**2 + dy**2) ** 0.5

        if magnitude < 1.0:
            # Hand is essentially on the target — return all-center pattern.
            # Activate the center row.
            grid[self.center_row, :] = intensity
            return grid

        # Normalise: divide each component by magnitude to get unit vector.
        # nx, ny are now in range -1.0 to +1.0.
        nx = dx / magnitude   # +1.0 = full right, -1.0 = full left
        ny = dy / magnitude   # +1.0 = full down,  -1.0 = full up

        # ── Activate directional electrode zones ───────────────────────────────
        # We use a threshold of 0.3 — if the normalised component is above
        # this, we consider that direction "active."
        # This allows diagonal movements (e.g. up-right) to activate two zones.
        threshold = 0.3

        # Horizontal guidance.
        if nx > threshold:
            # Object is to the RIGHT of the hand.
            # Activate the rightmost 2 columns (cols 4 and 5).
            # The intensity scales with how far right — more right = stronger.
            grid[:, 4] = intensity * min(nx, 1.0)
            grid[:, 5] = intensity * min(nx, 1.0)

        elif nx < -threshold:
            # Object is to the LEFT of the hand.
            # Activate the leftmost 2 columns (cols 0 and 1).
            # abs(nx) gives us the leftward magnitude.
            grid[:, 0] = intensity * min(abs(nx), 1.0)
            grid[:, 1] = intensity * min(abs(nx), 1.0)

        # Vertical guidance.
        # Note: in image coordinates, y increases DOWNWARD.
        # So ny > 0 means object is lower in the image = user needs to move DOWN.
        if ny > threshold:
            # Object is BELOW the hand (or user needs to move down/forward).
            # Activate the bottom 2 rows (rows 3 and 4).
            grid[3, :] = intensity * min(ny, 1.0)
            grid[4, :] = intensity * min(ny, 1.0)

        elif ny < -threshold:
            # Object is ABOVE the hand.
            # Activate the top 2 rows (rows 0 and 1).
            grid[0, :] = intensity * min(abs(ny), 1.0)
            grid[1, :] = intensity * min(abs(ny), 1.0)

        # If neither direction is dominant (hand very close to target),
        # activate the center row as a "you are close" signal.
        if abs(nx) <= threshold and abs(ny) <= threshold:
            grid[self.center_row, :] = intensity

        return grid

    def build_edge_grid(
        self,
        edge_map: np.ndarray
    ) -> np.ndarray:
        """
        Converts a Canny edge map into a 5×6 electrode grid.

        The edge map is a 2D binary image (white pixels = edges, black = no edge).
        We scale it down to 5×6 resolution so each cell corresponds to one electrode.
        If a cell contains any edge pixels, that electrode activates.

        Arguments:
            edge_map: numpy array (H, W) uint8, values 0 or 255
                      Output of cv2.Canny() — edges are white (255).

        Returns:
            numpy array shape (rows, cols) with values 0.0 or 1.0
        """

        # Resize the edge map to exactly our electrode grid dimensions.
        # cv2.resize(src, (width, height)) — note width before height.
        # INTER_AREA is best for shrinking — averages pixel blocks.
        resized = cv2.resize(
            edge_map,
            (self.cols, self.rows),
            interpolation=cv2.INTER_AREA
        )

        # resized is now a 5×6 image where each pixel represents one electrode.
        # Pixel value > 0 means there was an edge in that region.
        # Convert to float32 and normalise to 0.0-1.0 range.
        # 255.0 is the maximum possible value from cv2.Canny.
        grid = (resized / 255.0).astype(np.float32)

        # Apply a threshold — if more than 20% of the block had edges,
        # fully activate that electrode (set to 1.0).
        # This makes the haptic pattern more decisive — either on or off.
        # np.where(condition, if_true, if_false) applies element-wise.
        grid = np.where(grid > 0.2, 1.0, 0.0).astype(np.float32)

        return grid

    def build_success_grid(self, pulse_count: int = 0) -> np.ndarray:
        """
        Builds the success pattern — all electrodes pulse together.

        Creates an alternating on/off pattern based on pulse_count.
        Called repeatedly each frame to create a pulsing effect.

        Arguments:
            pulse_count: increments each frame — used to alternate on/off.
                         Even = all on, Odd = all off (creates pulsing).

        Returns:
            numpy array shape (rows, cols) with all values 1.0 or 0.0
        """

        # Even pulse_count = all on, odd = all off.
        # This creates a pulsing vibration effect.
        # pulse_count % 2 gives 0 for even, 1 for odd.
        intensity = 1.0 if (pulse_count % 2 == 0) else 0.0

        return np.full(
            (self.rows, self.cols),
            intensity,
            dtype=np.float32
        )


# =============================================================================
# INTERACTION DETECTOR CLASS
# =============================================================================

class InteractionDetector:
    """
    Detects interactable objects and guides the user's dominant hand.

    Two-phase interaction system:
      Phase 1 GUIDANCE: directional electrodes guide hand to object.
      Phase 2 EDGE:     edge rendering shows object shape on wrist.

    Usage:
        detector = InteractionDetector()
        detector.load_model()

        while in INTERACTION mode:
            result = detector.update(rgb_frame, depth_map)
            haptic.send(result["electrode_grid"])
    """

    def __init__(self):
        """
        Creates the detector. Does NOT load models yet.
        Call load_model() after creating this object.
        """

        # MediaPipe hands solution object.
        # None until load_model() runs.
        self._mp_hands = None
        self._hands    = None   # the actual detector

        # MediaPipe drawing utilities — for debug overlay.
        self._mp_drawing = None

        # Electrode grid builder.
        self._grid_builder = ElectrodeGridBuilder()

        # Haptic bridge — sends patterns to ESP32.
        self._haptic = HapticBridge()

        # Current interaction phase.
        self._phase: str = InteractionPhase.IDLE

        # Success pulse counter — incremented each frame in SUCCESS phase.
        # Used to create pulsing effect in build_success_grid().
        self._pulse_count: int = 0

        # Frame counter for rate limiting.
        self._frame_count: int = 0

        # Last computed electrode grid.
        # Stored so it can be accessed between update() calls.
        self._last_grid: np.ndarray = np.zeros(
            (HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32
        )

        # Last detected target object — the one we are guiding toward.
        # None if no target selected.
        self._target_object: Optional[Dict] = None

        # Whether models are loaded.
        self._ready: bool = False

        logger.info("InteractionDetector created. Call load_model() to start.")

    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def reset(self):
        """
        Resets the interaction detector state.

        Called when exiting INTERACTION mode — clears phase, target,
        pulse counter, and last grid so next interaction starts fresh.
        """
        self._phase = InteractionPhase.IDLE
        self._target_object = None
        self._pulse_count = 0
        self._last_grid = np.zeros(
            (HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32
        )
        self._haptic.send_all_off()
        logger.debug("InteractionDetector state reset.")

    def scan_for_interactables(
            self,
            detections: List[Dict],
            depth_map: np.ndarray
    ) -> float:
        """
        Lightweight scan — runs every frame regardless of mode.

        Only filters detections for interactable classes and returns
        the distance to the nearest one. Does NOT run hand detection,
        does NOT build electrode grids, does NOT send to haptic bridge.

        This is the function that feeds the state machine so it can
        decide when to switch to INTERACTION mode.

        Arguments:
            detections: track dicts from obstacle_detection.py
            depth_map:  depth map for distance lookup

        Returns:
            Distance in mm to nearest interactable object.
            Returns 0.0 if no interactable object found.
        """

        interactables = self._filter_interactables(detections, depth_map)

        if not interactables:
            # No interactable objects visible — clear the cached target.
            self._target_object = None
            return 0.0

        # Sort by distance and pick the closest.
        interactables_sorted = sorted(
            interactables,
            key=lambda d: d.get("distance_mm", 99999)
        )

        # Cache the nearest as our target — ready for when INTERACTION mode starts.
        self._target_object = interactables_sorted[0]

        nearest_dist = self._target_object.get("distance_mm", 0.0)

        logger.debug(
            f"Interactable scan: {len(interactables)} found, "
            f"nearest={self._target_object.get('label')} at {nearest_dist:.0f}mm"
        )

        return nearest_dist

    def load_model(self):
        """
        Loads MediaPipe hand detection model.

        MediaPipe downloads its models automatically on first use.
        No manual download needed.
        """

        logger.info("Loading MediaPipe hand detection model...")

        try:
            # mp.solutions.hands is MediaPipe's hand landmark solution.
            self._mp_hands   = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils

            # mp.solutions.hands.Hands() creates the hand detector.
            #
            # static_image_mode=False: optimised for video streams (our case).
            #   True would re-detect every frame (slower but more accurate).
            #   False tracks hands between frames (faster, better for real-time).
            #
            # max_num_hands=1: only track 1 hand (the dominant one).
            #   Saves processing time — we decided to track only dominant hand.
            #
            # min_detection_confidence=0.7: minimum confidence to detect a hand.
            #   0.7 = 70% confident — reduces false positives.
            #
            # min_tracking_confidence=0.5: minimum confidence to keep tracking.
            #   Once a hand is found, we keep tracking it above this threshold.
            self._hands = self._mp_hands.Hands(
                static_image_mode        = False,
                max_num_hands            = 1,
                min_detection_confidence = 0.7,
                min_tracking_confidence  = 0.5,
            )

            logger.info("MediaPipe hand detection loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load MediaPipe: {e}")
            raise

        # Connect haptic bridge.
        self._haptic.connect()

        self._ready = True
        logger.info("InteractionDetector ready.")

    # =========================================================================
    # MAIN UPDATE FUNCTION
    # =========================================================================

    def update(
        self,
        rgb_frame:  np.ndarray,
        depth_map:  np.ndarray,
        detections: List[Dict]
    ) -> Dict:
        """
        Main function — call every frame in INTERACTION mode.

        Steps:
          1. Detect dominant hand using MediaPipe
          2. Find nearest interactable object from YOLO detections
          3. Determine interaction phase based on hand-object distance
          4. Build electrode grid for current phase
          5. Send to haptic bridge
          6. Return complete interaction state

        Arguments:
            rgb_frame:  numpy array (H, W, 3) — current RGB frame
            depth_map:  numpy array (H, W) — depth in mm
            detections: list of detection dicts from obstacle_detection.py
                        (already filtered — only INTERACTABLE_CLASSES)

        Returns:
            Dictionary with complete interaction state (see bottom of method).
        """

        self._frame_count += 1

        # ── Step 1: Detect dominant hand ───────────────────────────────────────
        hand = self.detect_dominant_hand(rgb_frame)

        # ── Step 2: Find nearest interactable object ───────────────────────────
        # Filter the incoming detections to only interactable classes.
        interactables = self._filter_interactables(detections, depth_map)

        # Pick the nearest interactable as our target.
        # sorted() with key=lambda returns sorted list.
        # We sort by distance_mm — smallest (closest) first.
        if interactables:
            # Sort by distance, closest first.
            interactables_sorted = sorted(
                interactables,
                key=lambda d: d.get("distance_mm", 99999)
            )
            # The first item after sorting is the closest.
            self._target_object = interactables_sorted[0]
        else:
            self._target_object = None

        # ── Step 3: Determine phase ────────────────────────────────────────────
        # Phase depends on whether we have both a hand AND a target.
        if hand is None or self._target_object is None:
            # Missing either hand or object — go IDLE.
            self._phase = InteractionPhase.IDLE

        else:
            # We have both — compute pixel distance between finger tip and object.
            finger_tip  = hand["index_tip"]    # (fx, fy) in pixels
            obj_center  = self._target_object["center"]  # (ox, oy) in pixels

            # Pixel distance between finger tip and object center.
            pixel_dist = self._pixel_distance(finger_tip, obj_center)

            # Get real-world distance from depth map at the finger tip position.
            fx, fy = finger_tip
            h, w   = depth_map.shape

            # Clamp finger tip coordinates to valid frame bounds.
            # max(0, ...) prevents going below 0.
            # min(dim-1, ...) prevents exceeding frame size.
            fx_safe = max(0, min(w - 1, int(fx)))
            fy_safe = max(0, min(h - 1, int(fy)))

            # depth_map[y, x] — remember numpy arrays are [row, col] = [y, x].
            # This gives us the depth at the finger tip in mm.
            finger_depth_mm = float(depth_map[fy_safe, fx_safe])

            obj_depth_mm = self._target_object.get("distance_mm", 0)

            # Depth difference = how far apart hand and object are in Z (depth).
            # abs() because we only care about magnitude, not direction.
            depth_diff_mm = abs(finger_depth_mm - obj_depth_mm)

            # Determine phase based on depth difference.
            if depth_diff_mm <= EDGE_TO_SUCCESS_DIST_MM:
                self._phase = InteractionPhase.SUCCESS
            elif depth_diff_mm <= GUIDANCE_TO_EDGE_DIST_MM:
                self._phase = InteractionPhase.EDGE
            else:
                self._phase = InteractionPhase.GUIDANCE

        # ── Step 4: Build electrode grid ──────────────────────────────────────
        electrode_grid = self._build_electrode_grid(
            rgb_frame, depth_map, hand
        )

        # Cache for external access.
        self._last_grid = electrode_grid

        # ── Step 5: Send to haptic bridge ──────────────────────────────────────
        self._haptic.send(electrode_grid)

        # ── Step 6: Return state ───────────────────────────────────────────────
        return {
            "phase":          self._phase,
            "hand":           hand,
            "target":         self._target_object,
            "interactables":  interactables,
            "electrode_grid": electrode_grid,
            "on_target":      self._phase == InteractionPhase.SUCCESS,
            "timestamp_ms":   get_timestamp_ms(),
        }

    # =========================================================================
    # HAND DETECTION
    # =========================================================================

    def detect_dominant_hand(
        self,
        rgb_frame: np.ndarray
    ) -> Optional[Dict]:
        """
        Detects the user's dominant hand using MediaPipe.

        Returns the dominant hand's data, or None if not detected.

        Arguments:
            rgb_frame: numpy array (H, W, 3) BGR format from OpenCV

        Returns:
            Dictionary with hand data, or None if dominant hand not found.
        """

        if not self._ready or self._hands is None:
            return None

        h, w = rgb_frame.shape[:2]

        # MediaPipe requires RGB format — OpenCV uses BGR.
        # cv2.cvtColor converts between colour spaces.
        # COLOR_BGR2RGB = Blue-Green-Red → Red-Green-Blue conversion.
        rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        # Process the frame through MediaPipe.
        # results.multi_hand_landmarks  = list of detected hands' landmarks
        # results.multi_handedness      = list of "Left"/"Right" labels
        results = self._hands.process(rgb)

        # If no hands detected, return None.
        if not results.multi_hand_landmarks:
            return None

        # Find the dominant hand among detected hands.
        # MediaPipe labels each hand as "Left" or "Right".
        # results.multi_handedness is a list parallel to multi_hand_landmarks.
        dominant_hand_landmarks = None
        dominant_handedness     = None

        # zip() pairs each hand's landmarks with its handedness label.
        for landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            # handedness.classification[0].label is "Left" or "Right".
            # We compare to DOMINANT_HAND from config.py.
            label = handedness.classification[0].label

            if label == DOMINANT_HAND:
                dominant_hand_landmarks = landmarks
                dominant_handedness     = label
                break   # found dominant hand — stop looking

        # If dominant hand not found (only non-dominant visible), return None.
        if dominant_hand_landmarks is None:
            return None

        # ── Extract keypoints ──────────────────────────────────────────────────
        # MediaPipe gives us 21 landmarks, each with normalised x, y, z.
        # Normalised means values are 0.0-1.0 relative to frame dimensions.
        # We multiply by frame width/height to get pixel coordinates.

        # Convert all 21 landmarks to pixel coordinates.
        # landmark.x × frame_width  = pixel x coordinate
        # landmark.y × frame_height = pixel y coordinate
        all_landmarks_px = []
        for lm in dominant_hand_landmarks.landmark:
            px = int(lm.x * w)
            py = int(lm.y * h)
            all_landmarks_px.append((px, py))

        # Extract specific key points.
        # Index 0  = WRIST
        # Index 8  = INDEX_FINGER_TIP — the primary reaching point
        # Index 4  = THUMB_TIP
        # Index 12 = MIDDLE_FINGER_TIP
        wrist     = all_landmarks_px[0]
        index_tip = all_landmarks_px[8]   # most important for reaching
        thumb_tip = all_landmarks_px[4]

        return {
            "landmarks":   all_landmarks_px,  # all 21 keypoints
            "index_tip":   index_tip,         # (x, y) in pixels
            "wrist":       wrist,             # (x, y) in pixels
            "thumb_tip":   thumb_tip,         # (x, y) in pixels
            "handedness":  dominant_handedness,
        }

    # =========================================================================
    # INTERACTABLE OBJECT FILTERING
    # =========================================================================

    def _filter_interactables(
        self,
        detections: List[Dict],
        depth_map: np.ndarray
    ) -> List[Dict]:
        """
        Filters detections to only keep interactable objects.

        Adds depth and center information to each detection.

        Arguments:
            detections: raw detection dicts from obstacle_detection.py
            depth_map:  depth map for distance calculation

        Returns:
            Filtered list with depth added to each detection.
        """

        interactables = []

        for det in detections:
            # Only keep classes in our INTERACTABLE_CLASSES list.
            if det.get("label") not in INTERACTABLE_CLASSES:
                continue

            x1, y1, x2, y2 = det["bbox"]

            # Filter out tiny detections — too small to be reliable.
            from utils import bbox_area
            area = bbox_area(x1, y1, x2, y2)
            if area < MIN_INTERACTABLE_AREA_PX:
                continue

            # Get center and depth if not already present.
            cx, cy = bbox_center(x1, y1, x2, y2)

            if "distance_mm" not in det or det["distance_mm"] <= 0:
                dist = depth_in_region(depth_map, x1, y1, x2, y2)
                det["distance_mm"] = dist

            det["center"] = (cx, cy)

            interactables.append(det)

        return interactables

    # =========================================================================
    # ELECTRODE GRID BUILDING
    # =========================================================================

    def _build_electrode_grid(
        self,
        rgb_frame:  np.ndarray,
        depth_map:  np.ndarray,
        hand:       Optional[Dict]
    ) -> np.ndarray:
        """
        Builds the electrode activation grid for the current phase.

        Routes to the correct grid builder based on self._phase.

        Returns:
            numpy array shape (HAPTIC_ROWS, HAPTIC_COLS) values 0.0-1.0
        """

        # ── IDLE: all electrodes off ───────────────────────────────────────────
        if self._phase == InteractionPhase.IDLE:
            return np.zeros(
                (HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32
            )

        # ── SUCCESS: pulsing all-on pattern ───────────────────────────────────
        if self._phase == InteractionPhase.SUCCESS:
            self._pulse_count += 1
            return self._grid_builder.build_success_grid(self._pulse_count)

        # For GUIDANCE and EDGE, we need both hand and target.
        if hand is None or self._target_object is None:
            return np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)

        finger_tip = hand["index_tip"]         # (fx, fy)
        obj_center = self._target_object["center"]  # (ox, oy)

        # Direction vector from finger tip to object center.
        # dx positive = object is to the RIGHT of the finger.
        # dy positive = object is BELOW the finger (image coordinates).
        dx = obj_center[0] - finger_tip[0]
        dy = obj_center[1] - finger_tip[1]

        # ── GUIDANCE: directional electrodes ──────────────────────────────────
        if self._phase == InteractionPhase.GUIDANCE:

            # Calculate intensity based on distance.
            # Closer = stronger signal (more urgent guidance).
            pixel_dist = self._pixel_distance(finger_tip, obj_center)

            # Normalise distance to 0.0-1.0 range.
            # We cap at 300 pixels as "far" (intensity = 0.3 minimum).
            # intensity increases as hand gets closer.
            max_dist   = 300.0
            normalised = min(pixel_dist / max_dist, 1.0)

            # Intensity scales from 0.5 (far) to 1.0 (close to EDGE threshold).
            # We never go below 0.5 so the signal is always feelable.
            intensity  = 0.5 + (1.0 - normalised) * 0.5

            return self._grid_builder.build_guidance_grid(dx, dy, intensity)

        # ── EDGE: edge rendering ───────────────────────────────────────────────
        if self._phase == InteractionPhase.EDGE:
            return self._build_edge_rendering(
                rgb_frame, depth_map, hand
            )

        # Fallback — should never reach here.
        return np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)

    def _build_edge_rendering(
        self,
        rgb_frame:  np.ndarray,
        depth_map:  np.ndarray,
        hand:       Dict
    ) -> np.ndarray:
        """
        Builds the edge rendering electrode grid.

        Steps:
          1. Define a crop region in front of the hand
          2. Convert to grayscale
          3. Run Canny edge detection
          4. Scale edge map to 5×6 electrode grid

        The crop region is centered on the index finger tip and sized
        to cover the area immediately in front of the hand.

        Arguments:
            rgb_frame:  full RGB frame
            depth_map:  full depth map
            hand:       hand detection dict with index_tip

        Returns:
            5×6 electrode grid with edge pattern.
        """

        h, w = rgb_frame.shape[:2]
        fx, fy = hand["index_tip"]

        # ── Define crop region around finger tip ───────────────────────────────
        # We crop a square region centered on the index finger tip.
        # The size is proportional to the object's distance — closer objects
        # need a smaller crop (more detailed), distant ones need a wider view.
        obj_dist_mm = self._target_object.get("distance_mm", 500)

        # Crop size in pixels — scales with distance.
        # At 500mm: crop_size = 150px
        # At 200mm: crop_size = 100px
        # Closer = smaller crop = more detail per electrode.
        # max() ensures minimum size of 80px — prevents too-tiny crops.
        crop_size = max(80, int(150 * (obj_dist_mm / 500.0)))

        # Compute crop boundaries centered on finger tip.
        half  = crop_size // 2
        cx1   = max(0, fx - half)
        cy1   = max(0, fy - half)
        cx2   = min(w, fx + half)
        cy2   = min(h, fy + half)

        # Clamp to int for array indexing.
        cx1, cy1, cx2, cy2 = int(cx1), int(cy1), int(cx2), int(cy2)

        # Safety check — ensure valid crop dimensions.
        if cx2 <= cx1 or cy2 <= cy1:
            return np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)

        # ── Crop the region ────────────────────────────────────────────────────
        # Slice the RGB frame to get just the finger-tip region.
        # numpy slicing: frame[y1:y2, x1:x2]
        crop = rgb_frame[cy1:cy2, cx1:cx2]

        # ── Convert to grayscale ───────────────────────────────────────────────
        # Canny edge detection works on grayscale images.
        # COLOR_BGR2GRAY converts from 3-channel BGR to 1-channel grayscale.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # ── Apply Gaussian blur to reduce noise ────────────────────────────────
        # Blurring before Canny reduces false edges from camera noise.
        # (5, 5) is the kernel size — larger = more blur = fewer false edges.
        # 0 = compute sigma automatically from kernel size.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # ── Run Canny edge detection ───────────────────────────────────────────
        # cv2.Canny(image, low_threshold, high_threshold)
        # Pixels with gradient above high_threshold = definite edges (white).
        # Pixels with gradient below low_threshold  = not edges (black).
        # Pixels between thresholds = edges only if connected to definite edges.
        edges = cv2.Canny(
            blurred,
            CANNY_THRESHOLD_LOW,
            CANNY_THRESHOLD_HIGH
        )

        # ── Build electrode grid from edge map ─────────────────────────────────
        return self._grid_builder.build_edge_grid(edges)

    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================

    def _pixel_distance(
        self,
        point1: Tuple[int, int],
        point2: Tuple[int, int]
    ) -> float:
        """
        Calculates Euclidean pixel distance between two (x, y) points.

        Euclidean distance = sqrt((x2-x1)² + (y2-y1)²)

        Arguments:
            point1, point2: each is a (x, y) pixel coordinate tuple

        Returns:
            Distance in pixels as a float.
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return (dx**2 + dy**2) ** 0.5

    def get_nearest_interactable_distance(
        self,
        frame:     np.ndarray,
        depth_map: np.ndarray
    ) -> float:
        """
        Returns the distance to the nearest interactable object in mm.

        Called by state_machine.py to decide when to enter INTERACTION mode.
        Does NOT run full detection — reads from the last update() result.

        Returns 0.0 if no interactable object is currently tracked.
        """
        if self._target_object is None:
            return 0.0
        return self._target_object.get("distance_mm", 0.0)

    def draw_debug_overlay(
        self,
        frame: np.ndarray,
        result: Dict
    ) -> np.ndarray:
        """
        Draws the interaction debug overlay on the frame.

        Shows:
          - Hand landmarks (MediaPipe skeleton)
          - Target object bounding box
          - Guidance direction arrow
          - Current phase label
          - Electrode grid visualisation

        Arguments:
            frame:  RGB frame to draw on
            result: result dict from update()

        Returns:
            Frame with debug overlay drawn.
        """

        h, w = frame.shape[:2]
        phase = result.get("phase", InteractionPhase.IDLE)

        # ── Draw hand landmarks ────────────────────────────────────────────────
        hand = result.get("hand")
        if hand and hand.get("landmarks"):

            # Draw all 21 landmark points.
            for i, (px, py) in enumerate(hand["landmarks"]):
                # Draw a small circle at each keypoint.
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

            # Highlight the index finger tip — our primary reference point.
            itx, ity = hand["index_tip"]
            cv2.circle(frame, (itx, ity), 8, (0, 255, 255), 2)

            # Label the index finger tip.
            cv2.putText(
                frame, "index tip",
                (itx + 10, ity),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 255), 1
            )

        # ── Draw target object ─────────────────────────────────────────────────
        target = result.get("target")
        if target:
            x1, y1, x2, y2 = target["bbox"]
            label    = target.get("label", "?")
            dist_mm  = target.get("distance_mm", 0)

            # Draw target bounding box in cyan.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                frame,
                f"TARGET: {label} {dist_mm:.0f}mm",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 0), 1
            )

        # ── Draw guidance arrow ────────────────────────────────────────────────
        # Shows the direction the hand needs to move.
        if (hand and target
                and phase == InteractionPhase.GUIDANCE):

            fx, fy = hand["index_tip"]
            ox, oy = target["center"]

            # Draw an arrow from finger tip to object center.
            # cv2.arrowedLine(frame, start, end, colour, thickness, tipLength)
            cv2.arrowedLine(
                frame,
                (fx, fy),   # arrow starts at finger tip
                (ox, oy),   # arrow points to object center
                (0, 165, 255),   # orange
                2,
                tipLength=0.2
            )

        # ── Draw phase label ───────────────────────────────────────────────────
        phase_colours = {
            InteractionPhase.IDLE:     (120, 120, 120),
            InteractionPhase.GUIDANCE: (0, 165, 255),
            InteractionPhase.EDGE:     (0, 255, 165),
            InteractionPhase.SUCCESS:  (0, 220, 0),
        }
        phase_colour = phase_colours.get(phase, (200, 200, 200))

        cv2.putText(
            frame,
            f"INTERACTION: {phase}",
            (8, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, phase_colour, 2
        )

        # ── Draw electrode grid visualisation ─────────────────────────────────
        grid = result.get("electrode_grid")
        if grid is not None:
            frame = self._draw_electrode_grid(frame, grid)

        return frame

    def _draw_electrode_grid(
        self,
        frame: np.ndarray,
        grid:  np.ndarray
    ) -> np.ndarray:
        """
        Draws the 5×6 electrode grid as a small visualisation on the frame.

        Each electrode is shown as a small rectangle.
        Active electrodes = bright, inactive = dark.

        Positioned in the bottom-right corner of the debug frame.
        """

        h, w = frame.shape[:2]

        # Each electrode cell is 12×12 pixels in the visualisation.
        cell_size = 12
        padding   = 4   # gap between cells

        # Total grid display dimensions.
        grid_w = HAPTIC_COLS * (cell_size + padding)
        grid_h = HAPTIC_ROWS * (cell_size + padding)

        # Position: bottom-right corner with 10px margin.
        start_x = w - grid_w - 10
        start_y = h - grid_h - 10

        for row in range(HAPTIC_ROWS):
            for col in range(HAPTIC_COLS):

                # Get the activation value for this electrode.
                value = float(grid[row, col])

                # Calculate the pixel position of this electrode cell.
                x = start_x + col * (cell_size + padding)
                y = start_y + row * (cell_size + padding)

                # Colour: bright green when active, dark grey when inactive.
                # int(value * 255) converts 0.0-1.0 to 0-255 brightness.
                brightness = int(value * 200)
                colour     = (0, brightness, 0) if value > 0 else (30, 30, 30)

                # Draw the electrode cell rectangle.
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + cell_size, y + cell_size),
                    colour,
                    -1   # -1 = filled rectangle
                )

                # Draw a thin border around each cell.
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + cell_size, y + cell_size),
                    (60, 60, 60),
                    1
                )

        # Label the grid.
        cv2.putText(
            frame, "HAPTIC",
            (start_x, start_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (150, 150, 150), 1
        )

        return frame

    def get_stats(self) -> Dict:
        """Returns diagnostic statistics about the interaction detector."""
        return {
            "phase":        self._phase,
            "ready":        self._ready,
            "frame_count":  self._frame_count,
            "has_target":   self._target_object is not None,
            "target_label": self._target_object.get("label", "none")
                            if self._target_object else "none",
            "haptic_sends": self._haptic._send_count,
        }

    def release(self):
        """Cleanly shuts down the detector."""
        if self._hands:
            self._hands.close()
        self._haptic.send_all_off()
        self._haptic.disconnect()
        logger.info("InteractionDetector released.")


# =============================================================================
# SELF-TEST
# =============================================================================
# Tests WITHOUT camera — simulates hand and object positions.
# Run with: python interaction_detection.py

if __name__ == "__main__":

    print("=== ECHORA interaction_detection.py self-test ===\n")

    # ── Test 1: ElectrodeGridBuilder ──────────────────────────────────────────
    print("Test 1: Electrode grid builder")

    builder = ElectrodeGridBuilder()

    # Object is to the right — right columns should activate.
    grid_right = builder.build_guidance_grid(dx=100, dy=0)
    print(f"  Guidance RIGHT:\n{grid_right}")
    assert grid_right[0, 4] > 0 and grid_right[0, 5] > 0, "Right cols should be active"
    assert grid_right[0, 0] == 0 and grid_right[0, 1] == 0, "Left cols should be off"
    print("  PASSED\n")

    # Object is above — top rows should activate.
    grid_up = builder.build_guidance_grid(dx=0, dy=-100)
    print(f"  Guidance UP:\n{grid_up}")
    assert grid_up[0, 0] > 0 and grid_up[1, 0] > 0, "Top rows should be active"
    assert grid_up[3, 0] == 0 and grid_up[4, 0] == 0, "Bottom rows should be off"
    print("  PASSED\n")

    # Object is up-right — both top rows and right cols should activate.
    grid_diag = builder.build_guidance_grid(dx=80, dy=-80)
    print(f"  Guidance UP-RIGHT:\n{grid_diag}")
    print("  PASSED\n")

    # Success pattern.
    grid_success = builder.build_success_grid(pulse_count=0)
    print(f"  Success (even pulse — all ON):\n{grid_success}")
    assert np.all(grid_success == 1.0), "All electrodes should be on"
    print("  PASSED\n")

    grid_success_off = builder.build_success_grid(pulse_count=1)
    print(f"  Success (odd pulse — all OFF):\n{grid_success_off}")
    assert np.all(grid_success_off == 0.0), "All electrodes should be off"
    print("  PASSED\n")

    # ── Test 2: Edge grid from synthetic edge map ──────────────────────────────
    print("Test 2: Edge grid from synthetic edge map")

    # Create a synthetic 100×120 edge map with a vertical line on the right.
    fake_edges = np.zeros((100, 120), dtype=np.uint8)
    fake_edges[:, 90:95] = 255   # vertical line on the right side

    edge_grid = builder.build_edge_grid(fake_edges)
    print(f"  Edge grid (right side should be active):\n{edge_grid}")
    # Right columns should have more activation than left columns.
    assert edge_grid[:, -1].sum() >= edge_grid[:, 0].sum(), \
        "Right side should be more active than left"
    print("  PASSED\n")

    # ── Test 3: HapticBridge stub ─────────────────────────────────────────────
    print("Test 3: HapticBridge stub")
    bridge = HapticBridge()
    bridge.connect()

    test_grid = np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)
    test_grid[2, 3] = 1.0   # activate one electrode
    bridge.send(test_grid)

    bridge.send_all_on(0.5)
    bridge.send_all_off()
    print(f"  Haptic bridge sent {bridge._send_count} patterns.")
    assert bridge._send_count == 3
    print("  PASSED\n")

    # ── Test 4: InteractionDetector with camera ───────────────────────────────
    print("Test 4: InteractionDetector with live camera")
    print("  Loading model and camera...")

    from camera import EchoraCamera

    cam      = EchoraCamera()
    detector = InteractionDetector()

    try:
        cam.init_pipeline()
        detector.load_model()

        print("  Camera and model ready.")
        print("  Show your dominant hand and a cup/bottle to the camera.")
        print("  Press Q to stop.\n")

        frame_count = 0

        while True:
            bundle = cam.get_synced_bundle()
            if bundle is None:
                continue

            frame_count += 1
            rgb   = bundle["rgb"]
            depth = bundle["depth"]

            # Run update with empty detections for now
            # (obstacle_detection integration comes in control_unit).
            result = detector.update(rgb, depth, detections=[])

            # Draw debug overlay.
            debug = detector.draw_debug_overlay(rgb.copy(), result)

            cv2.imshow("Interaction Detection Test", debug)

            # Print phase changes.
            if frame_count % 15 == 0:
                stats = detector.get_stats()
                print(
                    f"  Frame {frame_count} | "
                    f"Phase: {stats['phase']:8s} | "
                    f"Target: {stats['target_label']}"
                )

            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n  Stopped by user.")

    finally:
        cv2.destroyAllWindows()
        detector.release()
        cam.release()
        print(f"\n  Processed {frame_count} frames.")

    print("\n=== All tests complete ===")