"""
interaction_detection.py
========================
Detects the user's right hand (MediaPipe Hands) and the nearest
interactable object (dedicated YOLOv8 instance), then computes the
full 3-D vector between the index fingertip and the target object
and converts it to directional haptic feedback on the 4×5 ESP32
wristband.

Pipeline per frame (INTERACTION mode only):
  1. MediaPipe Hands  → right-hand landmarks + index-fingertip pixel
  2. Depth map lookup → finger depth in mm
  3. YOLOv8 (COCO)   → nearest interactable object bbox + depth
  4. 3-D geometry     → dx_px, dy_px (image plane) | dz_mm (depth axis)
  5. Phase logic      → IDLE / GUIDANCE / EDGE / SUCCESS
  6. ElectrodeGridBuilder → 4×5 numpy grid
  7. HapticBridge     → 20-bit string → HTTP GET → ESP32 wristband

scan_for_interactables() is kept as a lightweight probe called by the
ControlUnit/StateMachine while in NAVIGATION mode to decide whether to
enter INTERACTION mode.  It reuses detections already produced by the
navigation YOLO so no extra inference is needed.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import requests
import torch
from ultralytics import YOLO

from src.core.config import settings
from src.core.utils import (
    bbox_area,
    bbox_center,
    depth_in_region,
    get_timestamp_ms,
    logger,
)

# ─────────────────────────────────────────────────────────────────────
# Hardware constants – match your ESP32 wristband
# ─────────────────────────────────────────────────────────────────────
GUI_IP        : str   = "127.0.0.1"
GUI_PORT      : int   = 8000
GUI_TIMEOUT   : float = 0.5      # seconds – low so it never blocks the loop
GRID_ROWS     : int   = 4        # physical rows on the wristband
GRID_COLS     : int   = 5        # physical cols on the wristband  (4×5 = 20 electrodes)

# ─────────────────────────────────────────────────────────────────────
# Interaction thresholds
# ─────────────────────────────────────────────────────────────────────
SUCCESS_XY_PX   : int   = 60     # finger tip must be within this radius (px) of obj centre
SUCCESS_Z_MM    : float = 50.0   # and within this depth delta (mm)
EDGE_XY_PX      : int   = 110    # XY-aligned threshold → enter EDGE phase
EDGE_Z_MM       : float = 250.0  # depth threshold → switch GUIDANCE→EDGE
GUIDANCE_MAX_PX : float = 450.0  # normalisation denominator for dx/dy


# ══════════════════════════════════════════════════════════════════════
# InteractionPhase
# ══════════════════════════════════════════════════════════════════════
class InteractionPhase:
    IDLE     = "IDLE"      # no hand or no target visible
    GUIDANCE = "GUIDANCE"  # hand far from target – send full 3-D direction cue
    EDGE     = "EDGE"      # XY aligned, hand needs to push forward in Z
    SUCCESS  = "SUCCESS"   # hand is at the object – all-pulse celebration


# ══════════════════════════════════════════════════════════════════════
# HapticBridge  –  20-electrode ESP32 HTTP driver
# ══════════════════════════════════════════════════════════════════════
class HapticBridge:
    """
    Converts a 4×5 numpy float grid into a 20-character bit string and
    sends it to the Flask Web GUI via:

        POST http://127.0.0.1:8000/api/pattern  {"bits": "..."}

    Degrades gracefully to STUB mode (log-only) when the GUI is
    unreachable so the rest of the pipeline can still be tested on a PC.
    """

    def __init__(self, gui_ip: str = GUI_IP, gui_port: int = GUI_PORT) -> None:
        self._url          = f"http://{gui_ip}:{gui_port}/api/pattern"
        self._ping_url     = f"http://{gui_ip}:{gui_port}/api/state"
        self._session      = requests.Session()
        self._stub         = True           # safe default until connect() succeeds
        self._connected    = False
        self._send_count   = 0
        self._error_count  = 0
        self._last_bits    = "0" * 20

    # ── Lifecycle ─────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Ping the ESP32.  Returns True on success, falls back to STUB on
        failure (no exception raised – haptic errors must never crash the
        navigation loop).
        """
        try:
            self._session.get(self._ping_url, timeout=2.0)
            self._connected = True
            self._stub      = False
            logger.info(f"HapticBridge: connected to local GUI at {GUI_IP}:{GUI_PORT}")
            return True
        except Exception as exc:
            self._connected = False
            self._stub      = True
            logger.warning(
                f"HapticBridge: local GUI unreachable ({exc}). Running in STUB (log-only) mode."
            )
            return False

    def disconnect(self) -> None:
        self.all_off()
        self._connected = False
        logger.info(
            f"HapticBridge disconnected. sends={self._send_count} errors={self._error_count}"
        )

    # ── Core send ─────────────────────────────────────────────────────

    def send_bits(self, bits: str) -> bool:
        """
        Send a 20-character binary string ('0'/'1') to the ESP32/GUI.
        This is fired in a background thread so network latency or
        timeouts NEVER block the main camera loop.
        """
        if len(bits) != 20:
            logger.error(f"HapticBridge.send_bits: expected 20 bits, got {len(bits)}")
            return False

        self._send_count += 1
        
        # Don't DDOS the local Flask server with identical states at 24 FPS
        if bits == self._last_bits:
            return True
            
        self._last_bits   = bits

        if self._stub:
            if self._send_count % 30 == 0:
                active = bits.count("1")
                logger.debug(f"[HAPTIC STUB #{self._send_count}] {active}/20 active  {bits}")
            return True

        def _worker():
            try:
                self._session.post(self._url, json={"bits": bits}, timeout=GUI_TIMEOUT)
            except Exception as exc:
                self._error_count += 1
                if self._error_count % 10 == 1:   # avoid log spam
                    logger.warning(f"HapticBridge: send to GUI failed: {exc}")

        threading.Thread(target=_worker, daemon=True).start()
        return True

    def send_grid(self, grid: np.ndarray) -> bool:
        """
        Convert a (GRID_ROWS × GRID_COLS) float32 grid to a 20-bit string
        and transmit.  Values > 0.5 → '1', otherwise → '0'.
        """
        flat = grid.flatten()[:20]
        bits = "".join("1" if v > 0.5 else "0" for v in flat)
        return self.send_bits(bits)

    # ── Convenience helpers ───────────────────────────────────────────

    def all_off(self) -> bool:
        return self.send_bits("0" * 20)

    def all_on(self) -> bool:
        return self.send_bits("1" * 20)

    def pulse_success(self, pulses: int = 3, interval: float = 0.12) -> None:
        """Runs in a background daemon thread – never blocks the main loop."""
        def _worker() -> None:
            for _ in range(pulses):
                self.all_on()
                time.sleep(interval)
                self.all_off()
                time.sleep(interval)
        threading.Thread(target=_worker, daemon=True).start()

    # ── Diagnostics ───────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            "stub":      self._stub,
            "connected": self._connected,
            "sends":     self._send_count,
            "errors":    self._error_count,
            "last_bits": self._last_bits,
        }


# ══════════════════════════════════════════════════════════════════════
# ElectrodeGridBuilder  –  maps a 3-D guidance vector to a 4×5 pattern
# ══════════════════════════════════════════════════════════════════════
class ElectrodeGridBuilder:
    """
    Physical wristband layout (worn on right wrist, palm facing down):

        col →    0      1      2      3      4
        row 0  [ UP ] [ UP ] [ FWD] [ UP ] [ UP ]   ← top of wrist
        row 1  [LEFT] [LEFT] [CTR ] [RGT ] [RGT ]
        row 2  [LEFT] [LEFT] [CTR ] [RGT ] [RGT ]
        row 3  [ DN ] [ DN ] [ FWD] [ DN ] [ DN ]   ← bottom of wrist

    Encoding rules
    ──────────────
    • LEFT  (dx_norm < –0.25) → cols 0, 1 active
    • RIGHT (dx_norm > +0.25) → cols 3, 4 active
    • UP    (dy_norm < –0.25) → row 0 active
    • DOWN  (dy_norm > +0.25) → row 3 active
    • FWD   (dz_norm > +0.10) → col 2 (center column) active
      ↳ This gives the user a distinct "push forward" cue
    • XY aligned + no FWD    → center col dim pulse (depth already ok)

    Directions can combine freely (e.g. UP + RIGHT + FWD all at once).
    Intensity scales with proximity so the wristband feels more urgent
    the closer the hand gets.
    """

    ROWS = GRID_ROWS   # 4
    COLS = GRID_COLS   # 5

    # ── Phase builders ────────────────────────────────────────────────

    def build_idle(self) -> np.ndarray:
        return np.zeros((self.ROWS, self.COLS), dtype=np.float32)

    def build_guidance(
        self,
        dx_norm: float,
        dy_norm: float,
        dz_norm: float,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """
        Progressive directional pattern.
        
        dx_norm : –1 (left) … +1 (right)
        dz_norm : –1 (backward/pull) … +1 (forward/push)
        """
        grid = np.zeros((self.ROWS, self.COLS), dtype=np.float32)
        
        # ── Horizontal axis (Left / Right) ────────────────────────────
        if dx_norm > 0.6:                      # Far Right -> 2 rightmost columns
            grid[:, 3] = intensity
            grid[:, 4] = intensity
        elif dx_norm > 0.2:                    # Slight Right -> 1 rightmost column
            grid[:, 4] = intensity
            
        elif dx_norm < -0.6:                   # Far Left -> 2 leftmost columns
            grid[:, 0] = intensity
            grid[:, 1] = intensity
        elif dx_norm < -0.2:                   # Slight Left -> 1 leftmost column
            grid[:, 0] = intensity

        # ── Depth axis (Forward / Backward mapped to Top / Bottom) ────
        if dz_norm > 0.6:                      # Far Forward -> top 2 rows
            grid[0, :] = intensity
            grid[1, :] = intensity
        elif dz_norm > 0.2:                    # Slight Forward -> top row
            grid[0, :] = intensity
            
        elif dz_norm < -0.6:                   # Far Backward -> bottom 2 rows
            grid[2, :] = intensity
            grid[3, :] = intensity
        elif dz_norm < -0.2:                   # Slight Backward -> bottom row
            grid[3, :] = intensity

        # ── Up / Down (Y-axis) ────────────────────────────────────────
        # (Optional) Since top/bottom rows are now used for forward/backward,
        # we can pulse the center column if we want up/down, or we can just 
        # omit Y-axis guidance if you only want to guide them on a 2D plane (like a table).
        # We'll leave Y out for now to keep the forward/backward mapping clean.

        # If perfectly aligned in both X and Z
        if abs(dx_norm) <= 0.2 and abs(dz_norm) <= 0.2:
            grid[1, 2] = intensity * 0.5
            grid[2, 2] = intensity * 0.5

        return grid

    def build_forward_push(self, intensity: float = 1.0) -> np.ndarray:
        """
        EDGE phase: finger tip is XY-aligned with the object but still
        needs to push forward in depth.  Full center column lights up.
        """
        grid = np.zeros((self.ROWS, self.COLS), dtype=np.float32)
        grid[:, 2] = intensity          # entire center column
        return grid

    def build_success(self, pulse_count: int) -> np.ndarray:
        """All electrodes alternating on/off – tactile 'you made it!' signal."""
        intensity = 1.0 if (pulse_count % 2 == 0) else 0.0
        return np.full((self.ROWS, self.COLS), intensity, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
# InteractionDetector  –  the public interface
# ══════════════════════════════════════════════════════════════════════
class InteractionDetector:
    """
    Owns the full INTERACTION-mode pipeline:

      • MediaPipe Hands  – right-hand detection & 21-landmark extraction
      • YOLOv8 (COCO)   – dedicated lightweight instance for interactable
                          object detection (runs only in INTERACTION mode)
      • 3-D geometry     – computes dx_px, dy_px, dz_mm
      • Phase FSM        – IDLE → GUIDANCE → EDGE → SUCCESS
      • HapticBridge     – drives the 4×5 ESP32 wristband

    scan_for_interactables() is a lightweight probe (no separate YOLO
    inference) used by ControlUnit while still in NAVIGATION mode to
    decide whether to switch into INTERACTION mode.
    """

    def __init__(self) -> None:
        # MediaPipe
        self._mp_hands   : Optional[object] = None
        self._hands      : Optional[object] = None
        self._mp_drawing : Optional[object] = None

        # Haptic
        self._haptic       = HapticBridge()
        self._grid_builder = ElectrodeGridBuilder()

        # State
        self._phase       : str            = InteractionPhase.IDLE
        self._target_obj  : Optional[Dict] = None
        self._pulse_count : int            = 0
        self._frame_count : int            = 0
        self._last_grid   : np.ndarray     = np.zeros(
            (GRID_ROWS, GRID_COLS), dtype=np.float32
        )
        self._ready       : bool           = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load MediaPipe Hands. Call once at startup."""
        logger.info("InteractionDetector: loading MediaPipe Hands...")
        try:
            self._mp_hands   = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            self._hands      = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.50,
                min_tracking_confidence=0.50,
                model_complexity=1,
            )
        except Exception as exc:
            logger.error(f"InteractionDetector: MediaPipe load failed: {exc}")
            raise

        self._haptic.connect()
        self._ready = True
        logger.info("InteractionDetector: ready.")

    def release(self) -> None:
        if self._hands:
            self._hands.close()
        self._haptic.all_off()
        self._haptic.disconnect()
        self._ready = False

    def reset(self) -> None:
        """Called by StateMachine on mode exit – clears all transient state."""
        self._phase       = InteractionPhase.IDLE
        self._target_obj  = None
        self._pulse_count = 0
        self._last_grid   = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        self._haptic.all_off()

    # ── Navigation haptic alerts (shared bridge) ──────────────────────

    def pulse_danger(self, pulses: int = 3, interval: float = 0.1) -> None:
        """
        Rapid checkerboard burst — DANGER alert for navigation mode.
        Uses the same HapticBridge as interaction mode so the signal
        actually reaches the ESP32 via the Flask GUI.
        Runs in a background thread so it never blocks the camera loop.
        """
        _DANGER_BITS = "10101010101010101010"  # checkerboard on 4×5 grid
        def _worker() -> None:
            for _ in range(pulses):
                self._haptic.send_bits(_DANGER_BITS)
                time.sleep(interval)
                self._haptic.all_off()
                time.sleep(interval)
        threading.Thread(target=_worker, daemon=True).start()

    def pulse_warning(self, pulses: int = 2, interval: float = 0.25) -> None:
        """
        Slow double pulse — WARNING alert for navigation mode.
        Softer than danger: all-on at lower cadence.
        """
        def _worker() -> None:
            for _ in range(pulses):
                self._haptic.all_on()
                time.sleep(interval)
                self._haptic.all_off()
                time.sleep(interval)
        threading.Thread(target=_worker, daemon=True).start()

    # ── Lightweight probe (called while in NAVIGATION mode) ───────────

    def scan_for_interactables(
        self, detections: List[Dict], depth_map: np.ndarray
    ) -> float:
        """
        Filters the navigation-YOLO detections for interactable objects and
        returns the distance to the nearest one in mm (0.0 if none found).

        This does NOT run a separate YOLO inference – it reuses detections
        already computed by ObstacleDetector, keeping overhead minimal.
        """
        candidates = self._filter_interactables(detections, depth_map)
        if not candidates:
            self._target_obj = None
            return 0.0

        nearest = min(candidates, key=lambda d: d.get("distance_mm", 99_999))
        self._target_obj = nearest
        return nearest.get("distance_mm", 0.0)

    # ── Main per-frame update (called while in INTERACTION mode) ───────

    def update(self, rgb_frame: np.ndarray, depth_map: np.ndarray, detections: List[Dict]) -> Dict:
        """
        Full INTERACTION-mode pipeline.  Call every frame.
        """
        if not self._ready:
            return self._empty_result()

        self._frame_count += 1

        # 1 ── Detect right hand
        hand = self._detect_right_hand(rgb_frame)

        # 2 ── Find nearest interactable object (from passed-in YOLO detections)
        candidates = self._filter_interactables(detections, depth_map)
        if candidates:
            self._target_obj = min(candidates, key=lambda d: d.get("distance_mm", 99_999))
        else:
            self._target_obj = None

        # 3 ── Compute 3-D vector: finger tip → object centre
        vector = self._compute_3d_vector(hand, self._target_obj, depth_map, rgb_frame)

        # 4 ── Determine phase
        prev_phase   = self._phase
        self._phase  = self._compute_phase(hand, self._target_obj, vector)

        # 5 ── Build electrode grid
        grid           = self._build_grid(vector)
        self._last_grid = grid

        # 6 ── Send to wristband
        self._haptic.send_grid(grid)

        # 7 ── Trigger one-shot success pulse on first SUCCESS frame
        if self._phase == InteractionPhase.SUCCESS:
            if prev_phase != InteractionPhase.SUCCESS:
                self._haptic.pulse_success()
            self._pulse_count += 1
        else:
            self._pulse_count = 0

        return {
            "phase":          self._phase,
            "hand":           hand,
            "target":         self._target_obj,
            "vector":         vector,
            "electrode_grid": grid,
            "on_target":      self._phase == InteractionPhase.SUCCESS,
            "timestamp_ms":   get_timestamp_ms(),
        }

    # ── Hand detection ────────────────────────────────────────────────

    def _detect_right_hand(self, rgb_frame: np.ndarray) -> Optional[Dict]:
        """
        Run MediaPipe Hands on the frame and return structured data for
        the user's dominant (right) hand only.

        Returns None if no right hand is visible.
        """
        if self._hands is None:
            return None

        h, w = rgb_frame.shape[:2]
        # MediaPipe requires RGB; OAK-D delivers BGR
        rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        for landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness.classification[0].label   # "Left" or "Right"
            # Accept any hand in the frame since egocentric left/right
            # classification in MediaPipe is often inverted.
            all_px = [
                (int(lm.x * w), int(lm.y * h))
                for lm in landmarks.landmark
            ]
            return {
                "landmarks":  all_px,
                "index_tip":  all_px[8],   # MediaPipe INDEX_FINGER_TIP
                "middle_tip": all_px[12],  # MIDDLE_FINGER_TIP
                "thumb_tip":  all_px[4],   # THUMB_TIP
                "wrist":      all_px[0],   # WRIST
                "handedness": label,
            }

        return None   # dominant hand not in frame

    # ── Object detection ──────────────────────────────────────────────

    def _filter_interactables(
        self, detections: List[Dict], depth_map: np.ndarray
    ) -> List[Dict]:
        """
        Filters the navigation-YOLO detections for interactable objects.
        """
        out: List[Dict] = []
        for det in detections:
            if det.get("label") not in settings.INTERACTABLE_CLASSES:
                continue

            x1, y1, x2, y2 = det["bbox"]
            if bbox_area(x1, y1, x2, y2) < settings.MIN_INTERACTABLE_AREA_PX:
                continue

            if "distance_mm" not in det or det["distance_mm"] <= 0:
                det["distance_mm"] = depth_in_region(depth_map, x1, y1, x2, y2)

            det["center"] = bbox_center(x1, y1, x2, y2)
            out.append(det)

        return out

    # ── 3-D vector ────────────────────────────────────────────────────

    def _compute_3d_vector(
        self,
        hand    : Optional[Dict],
        target  : Optional[Dict],
        depth_map: np.ndarray,
        rgb_frame: np.ndarray,
    ) -> Optional[Dict]:
        """
        Compute the 3-D offset from the index finger tip to the object centre.

        Returns a dict:
            dx_px          : pixel delta  (positive → object is to the RIGHT)
            dy_px          : pixel delta  (positive → object is BELOW)
            dz_mm          : depth delta  (positive → object is FURTHER AWAY, push forward)
            dx_norm        : dx_px  normalised to –1..+1
            dy_norm        : dy_px  normalised to –1..+1
            dz_norm        : dz_mm  normalised to –1..+1  (over ±1000 mm range)
            xy_dist_px     : 2-D Euclidean distance in pixels
            finger_depth_mm: depth of index finger tip in mm
            obj_depth_mm   : median depth of object bbox in mm
        """
        if hand is None or target is None:
            return None

        h_px, w_px = rgb_frame.shape[:2]

        fx, fy = hand["index_tip"]
        fx_s   = max(0, min(w_px - 1, fx))
        fy_s   = max(0, min(h_px - 1, fy))
        finger_depth_mm = float(depth_map[fy_s, fx_s])

        if finger_depth_mm <= 0:
            # Depth not available for this pixel – fall back to raw delta only
            finger_depth_mm = target["distance_mm"]   # assume same plane

        obj_cx, obj_cy = target["center"]
        obj_depth_mm   = target["distance_mm"]

        dx_px = obj_cx - fx
        dy_px = obj_cy - fy
        dz_mm = obj_depth_mm - finger_depth_mm    # > 0 → push hand forward

        xy_dist_px = float((dx_px ** 2 + dy_px ** 2) ** 0.5)

        dx_norm = float(np.clip(dx_px / GUIDANCE_MAX_PX, -1.0, 1.0))
        dy_norm = float(np.clip(dy_px / GUIDANCE_MAX_PX, -1.0, 1.0))
        dz_norm = float(np.clip(dz_mm / 1000.0,         -1.0, 1.0))

        return {
            "dx_px":           dx_px,
            "dy_px":           dy_px,
            "dz_mm":           dz_mm,
            "dx_norm":         dx_norm,
            "dy_norm":         dy_norm,
            "dz_norm":         dz_norm,
            "xy_dist_px":      xy_dist_px,
            "finger_depth_mm": finger_depth_mm,
            "obj_depth_mm":    obj_depth_mm,
        }

    # ── Phase FSM ─────────────────────────────────────────────────────

    def _compute_phase(
        self,
        hand  : Optional[Dict],
        target: Optional[Dict],
        vector: Optional[Dict],
    ) -> str:
        if hand is None or target is None or vector is None:
            return InteractionPhase.IDLE

        xy_dist = vector["xy_dist_px"]
        z_dist  = abs(vector["dz_mm"])

        # SUCCESS: within reach in all three axes
        if xy_dist <= SUCCESS_XY_PX and z_dist <= SUCCESS_Z_MM:
            return InteractionPhase.SUCCESS

        # EDGE: XY aligned but still needs to push forward/back
        if xy_dist <= EDGE_XY_PX:
            return InteractionPhase.EDGE

        # GUIDANCE: general direction feedback
        return InteractionPhase.GUIDANCE

    # ── Electrode grid construction ───────────────────────────────────

    def _build_grid(self, vector: Optional[Dict]) -> np.ndarray:
        if self._phase == InteractionPhase.IDLE or vector is None:
            return self._grid_builder.build_idle()

        if self._phase == InteractionPhase.SUCCESS:
            return self._grid_builder.build_success(self._pulse_count)

        # Intensity scales with proximity (closer → stronger vibration)
        xy      = vector["xy_dist_px"]
        intensity = 0.50 + 0.50 * (1.0 - min(xy / GUIDANCE_MAX_PX, 1.0))

        if self._phase == InteractionPhase.EDGE:
            # Depth-only feedback: tell user to push forward
            depth_intensity = min(abs(vector["dz_mm"]) / 500.0, 1.0)
            return self._grid_builder.build_forward_push(
                max(intensity, depth_intensity)
            )

        # GUIDANCE: full 3-D directional pattern
        return self._grid_builder.build_guidance(
            dx_norm=vector["dx_norm"],
            dy_norm=vector["dy_norm"],
            dz_norm=vector["dz_norm"],
            intensity=intensity,
        )

    # ── Private helpers ───────────────────────────────────────────────



    def _empty_result(self) -> Dict:
        return {
            "phase":          InteractionPhase.IDLE,
            "hand":           None,
            "target":         None,
            "vector":         None,
            "electrode_grid": self._last_grid,
            "on_target":      False,
            "timestamp_ms":   get_timestamp_ms(),
        }

    # ── Debug overlay ─────────────────────────────────────────────────

    def draw_debug_overlay(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draws hand landmarks, target box, guidance arrow, phase label,
        3-D vector readout and electrode grid visualisation onto *frame*.
        Returns the annotated frame.
        """
        h, w = frame.shape[:2]
        phase  = result.get("phase",  InteractionPhase.IDLE)
        hand   = result.get("hand")
        target = result.get("target")
        vector = result.get("vector")

        # ── Hand landmarks ────────────────────────────────────────────
        if hand:
            for px, py in hand["landmarks"]:
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
            itx, ity = hand["index_tip"]
            cv2.circle(frame, (itx, ity), 10, (0, 255, 255), 2)
            cv2.putText(
                frame, "R",
                (itx + 13, ity),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
            )

        # ── Target bounding box ───────────────────────────────────────
        if target:
            x1, y1, x2, y2 = target["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            label_txt = f"{target['label']}  {target['distance_mm']:.0f} mm"
            cv2.putText(
                frame, label_txt, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA,
            )

        # ── Guidance arrow ────────────────────────────────────────────
        if hand and target and phase in (InteractionPhase.GUIDANCE, InteractionPhase.EDGE):
            fx, fy = hand["index_tip"]
            ox, oy = target["center"]
            cv2.arrowedLine(frame, (fx, fy), (ox, oy), (0, 165, 255), 2, tipLength=0.2)

        # ── 3-D vector readout ────────────────────────────────────────
        if vector:
            vinfo = (
                f"dx={vector['dx_px']:+.0f}px  "
                f"dy={vector['dy_px']:+.0f}px  "
                f"dz={vector['dz_mm']:+.0f}mm  "
                f"xy={vector['xy_dist_px']:.0f}px  "
                f"fngr={vector['finger_depth_mm']:.0f}mm  "
                f"obj={vector['obj_depth_mm']:.0f}mm"
            )
            cv2.putText(
                frame, vinfo, (8, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA,
            )

        # ── Phase label ───────────────────────────────────────────────
        phase_colors = {
            InteractionPhase.IDLE:     (100, 100, 100),
            InteractionPhase.GUIDANCE: (0, 165, 255),
            InteractionPhase.EDGE:     (0, 220, 165),
            InteractionPhase.SUCCESS:  (0, 220, 0),
        }
        cv2.putText(
            frame, f"INTERACT: {phase}", (8, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            phase_colors.get(phase, (200, 200, 200)), 2, cv2.LINE_AA,
        )

        # ── Electrode grid visualisation ──────────────────────────────
        grid = result.get("electrode_grid")
        if grid is not None:
            cell_sz = 14
            padding = 3
            sx = w - (GRID_COLS * (cell_sz + padding)) - 10
            sy = h - (GRID_ROWS * (cell_sz + padding)) - 10
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    val = float(grid[r, c])
                    x = sx + c * (cell_sz + padding)
                    y = sy + r * (cell_sz + padding)
                    color = (0, int(val * 220), int(val * 80)) if val > 0 else (25, 25, 25)
                    cv2.rectangle(frame, (x, y), (x + cell_sz, y + cell_sz), color, -1)
                    cv2.rectangle(frame, (x, y), (x + cell_sz, y + cell_sz), (55, 55, 55), 1)

        return frame

    # ── Stats ─────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            "phase":        self._phase,
            "ready":        self._ready,
            "frame_count":  self._frame_count,
            "has_target":   self._target_obj is not None,
            "target_label": (
                self._target_obj.get("label", "none")
                if self._target_obj
                else "none"
            ),
            "haptic":       self._haptic.get_stats(),
        }