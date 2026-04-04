import mediapipe as mp
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple

from src.core.config import (
    DOMINANT_HAND,
    HAPTIC_ROWS,
    HAPTIC_COLS,
    GUIDANCE_TO_EDGE_DIST_MM,
    EDGE_TO_SUCCESS_DIST_MM,
    CANNY_THRESHOLD_LOW,
    CANNY_THRESHOLD_HIGH,
    MIN_INTERACTABLE_AREA_PX,
    INTERACTABLE_CLASSES,
)
from src.core.utils import logger, bbox_center, depth_in_region, bbox_area, get_timestamp_ms

class InteractionPhase:
    IDLE     = "IDLE"
    GUIDANCE = "GUIDANCE"
    EDGE     = "EDGE"
    SUCCESS  = "SUCCESS"

class HapticBridge:
    """Sends electrode activation patterns to the ESP32 wristband (STUB)."""

    def __init__(self):
        self._connected: bool = False
        self._send_count: int = 0

    def connect(self):
        pass

    def send(self, electrode_grid: np.ndarray):
        self._send_count += 1
        flat = electrode_grid.flatten()
        n_active = int(np.sum(flat > 0))
        logger.debug(f"HapticBridge.send(): {n_active}/30 active electrodes")

    def send_all_off(self):
        self.send(np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32))

    def send_all_on(self, intensity: float = 1.0):
        self.send(np.full((HAPTIC_ROWS, HAPTIC_COLS), intensity, dtype=np.float32))

    def disconnect(self):
        self._connected = False

class ElectrodeGridBuilder:
    """Builds 5x6 electrode activation grids."""

    def __init__(self, rows: int = HAPTIC_ROWS, cols: int = HAPTIC_COLS):
        self.rows = rows
        self.cols = cols
        self.center_row = rows // 2
        self.center_col = cols // 2

    def build_guidance_grid(self, dx: float, dy: float, intensity: float = 1.0) -> np.ndarray:
        grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        magnitude = (dx**2 + dy**2) ** 0.5

        if magnitude < 1.0:
            grid[self.center_row, :] = intensity
            return grid

        nx = dx / magnitude
        ny = dy / magnitude
        threshold = 0.3

        if nx > threshold:
            grid[:, 4] = intensity * min(nx, 1.0)
            grid[:, 5] = intensity * min(nx, 1.0)
        elif nx < -threshold:
            grid[:, 0] = intensity * min(abs(nx), 1.0)
            grid[:, 1] = intensity * min(abs(nx), 1.0)

        if ny > threshold:
            grid[3, :] = intensity * min(ny, 1.0)
            grid[4, :] = intensity * min(ny, 1.0)
        elif ny < -threshold:
            grid[0, :] = intensity * min(abs(ny), 1.0)
            grid[1, :] = intensity * min(abs(ny), 1.0)

        if abs(nx) <= threshold and abs(ny) <= threshold:
            grid[self.center_row, :] = intensity

        return grid

    def build_edge_grid(self, edge_map: np.ndarray) -> np.ndarray:
        resized = cv2.resize(edge_map, (self.cols, self.rows), interpolation=cv2.INTER_AREA)
        grid = (resized / 255.0).astype(np.float32)
        return np.where(grid > 0.2, 1.0, 0.0).astype(np.float32)

    def build_success_grid(self, pulse_count: int = 0) -> np.ndarray:
        intensity = 1.0 if (pulse_count % 2 == 0) else 0.0
        return np.full((self.rows, self.cols), intensity, dtype=np.float32)

class InteractionDetector:
    """Detects interactable objects and guides user hand."""

    def __init__(self):
        self._mp_hands = None
        self._hands = None
        self._mp_drawing = None
        self._grid_builder = ElectrodeGridBuilder()
        self._haptic = HapticBridge()
        
        self._phase = InteractionPhase.IDLE
        self._pulse_count = 0
        self._frame_count = 0
        self._last_grid = np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)
        self._target_object = None
        self._ready = False

    def reset(self):
        self._phase = InteractionPhase.IDLE
        self._target_object = None
        self._pulse_count = 0
        self._last_grid = np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)
        self._haptic.send_all_off()

    def scan_for_interactables(self, detections: List[Dict], depth_map: np.ndarray) -> float:
        interactables = self._filter_interactables(detections, depth_map)
        if not interactables:
            self._target_object = None
            return 0.0

        interactables_sorted = sorted(interactables, key=lambda d: d.get("distance_mm", 99999))
        self._target_object = interactables_sorted[0]
        return self._target_object.get("distance_mm", 0.0)

    def load_model(self):
        logger.info("Loading MediaPipe hand detection model...")
        try:
            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            logger.error(f"Failed to load MediaPipe: {e}")
            raise

        self._haptic.connect()
        self._ready = True

    def update(self, rgb_frame: np.ndarray, depth_map: np.ndarray, detections: List[Dict]) -> Dict:
        self._frame_count += 1
        hand = self.detect_dominant_hand(rgb_frame)
        
        interactables = self._filter_interactables(detections, depth_map)
        if interactables:
            interactables_sorted = sorted(interactables, key=lambda d: d.get("distance_mm", 99999))
            self._target_object = interactables_sorted[0]
        else:
            self._target_object = None

        if hand is None or self._target_object is None:
            self._phase = InteractionPhase.IDLE
        else:
            fx, fy = hand["index_tip"]
            h, w = depth_map.shape
            fx_safe = max(0, min(w - 1, int(fx)))
            fy_safe = max(0, min(h - 1, int(fy)))
            
            finger_depth_mm = float(depth_map[fy_safe, fx_safe])
            obj_depth_mm = self._target_object.get("distance_mm", 0)
            depth_diff_mm = abs(finger_depth_mm - obj_depth_mm)

            if depth_diff_mm <= EDGE_TO_SUCCESS_DIST_MM:
                self._phase = InteractionPhase.SUCCESS
            elif depth_diff_mm <= GUIDANCE_TO_EDGE_DIST_MM:
                self._phase = InteractionPhase.EDGE
            else:
                self._phase = InteractionPhase.GUIDANCE

        electrode_grid = self._build_electrode_grid(rgb_frame, depth_map, hand)
        self._last_grid = electrode_grid
        self._haptic.send(electrode_grid)

        return {
            "phase":          self._phase,
            "hand":           hand,
            "target":         self._target_object,
            "interactables":  interactables,
            "electrode_grid": electrode_grid,
            "on_target":      self._phase == InteractionPhase.SUCCESS,
            "timestamp_ms":   get_timestamp_ms(),
        }

    def detect_dominant_hand(self, rgb_frame: np.ndarray) -> Optional[Dict]:
        if not self._ready or self._hands is None:
            return None

        h, w = rgb_frame.shape[:2]
        rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        dominant_hand_landmarks = None
        dominant_handedness = None

        for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            if label == DOMINANT_HAND:
                dominant_hand_landmarks = landmarks
                dominant_handedness = label
                break

        if dominant_hand_landmarks is None:
            return None

        all_landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in dominant_hand_landmarks.landmark]
        
        return {
            "landmarks":  all_landmarks_px,
            "index_tip":  all_landmarks_px[8],
            "wrist":      all_landmarks_px[0],
            "thumb_tip":  all_landmarks_px[4],
            "handedness": dominant_handedness,
        }

    def _filter_interactables(self, detections: List[Dict], depth_map: np.ndarray) -> List[Dict]:
        interactables = []
        for det in detections:
            if det.get("label") not in INTERACTABLE_CLASSES:
                continue

            x1, y1, x2, y2 = det["bbox"]
            if bbox_area(x1, y1, x2, y2) < MIN_INTERACTABLE_AREA_PX:
                continue

            if "distance_mm" not in det or det["distance_mm"] <= 0:
                det["distance_mm"] = depth_in_region(depth_map, x1, y1, x2, y2)

            det["center"] = bbox_center(x1, y1, x2, y2)
            interactables.append(det)

        return interactables

    def _build_electrode_grid(self, rgb_frame: np.ndarray, depth_map: np.ndarray, hand: Optional[Dict]) -> np.ndarray:
        if self._phase == InteractionPhase.IDLE:
            return np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)

        if self._phase == InteractionPhase.SUCCESS:
            self._pulse_count += 1
            return self._grid_builder.build_success_grid(self._pulse_count)

        if hand is None or self._target_object is None:
            return np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)

        finger_tip = hand["index_tip"]
        obj_center = self._target_object["center"]

        if self._phase == InteractionPhase.GUIDANCE:
            dx, dy = obj_center[0] - finger_tip[0], obj_center[1] - finger_tip[1]
            pixel_dist = ((dx)**2 + (dy)**2) ** 0.5
            normalised = min(pixel_dist / 300.0, 1.0)
            intensity = 0.5 + (1.0 - normalised) * 0.5
            return self._grid_builder.build_guidance_grid(dx, dy, intensity)

        if self._phase == InteractionPhase.EDGE:
            return self._build_edge_rendering(rgb_frame, depth_map, hand)

        return np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)

    def _build_edge_rendering(self, rgb_frame: np.ndarray, depth_map: np.ndarray, hand: Dict) -> np.ndarray:
        h, w = rgb_frame.shape[:2]
        fx, fy = hand["index_tip"]
        
        obj_dist_mm = self._target_object.get("distance_mm", 500)
        crop_size = max(80, int(150 * (obj_dist_mm / 500.0)))
        half = crop_size // 2
        
        cx1, cy1 = max(0, fx - half), max(0, fy - half)
        cx2, cy2 = min(w, fx + half), min(h, fy + half)
        
        if cx2 <= cx1 or cy2 <= cy1:
            return np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)

        crop = rgb_frame[int(cy1):int(cy2), int(cx1):int(cx2)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
        return self._grid_builder.build_edge_grid(edges)

    def draw_debug_overlay(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        h, w = frame.shape[:2]
        phase = result.get("phase", InteractionPhase.IDLE)
        hand = result.get("hand")
        target = result.get("target")

        if hand and hand.get("landmarks"):
            for px, py in hand["landmarks"]:
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)
            itx, ity = hand["index_tip"]
            cv2.circle(frame, (itx, ity), 8, (0, 255, 255), 2)

        if target:
            x1, y1, x2, y2 = target["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"TARGET {target.get('distance_mm', 0):.0f}mm", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if hand and target and phase == InteractionPhase.GUIDANCE:
            fx, fy = hand["index_tip"]
            ox, oy = target["center"]
            cv2.arrowedLine(frame, (fx, fy), (ox, oy), (0, 165, 255), 2, tipLength=0.2)

        phase_colours = {
            InteractionPhase.IDLE: (120, 120, 120), InteractionPhase.GUIDANCE: (0, 165, 255),
            InteractionPhase.EDGE: (0, 255, 165), InteractionPhase.SUCCESS: (0, 220, 0)
        }
        cv2.putText(frame, f"INTERACTION: {phase}", (8, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_colours.get(phase, (200, 200, 200)), 2)

        grid = result.get("electrode_grid")
        if grid is not None:
            cell_sz, padding = 12, 4
            start_x = w - (HAPTIC_COLS * (cell_sz + padding)) - 10
            start_y = h - (HAPTIC_ROWS * (cell_sz + padding)) - 10

            for row in range(HAPTIC_ROWS):
                for col in range(HAPTIC_COLS):
                    val = float(grid[row, col])
                    x, y = start_x + col * (cell_sz + padding), start_y + row * (cell_sz + padding)
                    colour = (0, int(val * 200), 0) if val > 0 else (30, 30, 30)
                    cv2.rectangle(frame, (x, y), (x + cell_sz, y + cell_sz), colour, -1)
                    cv2.rectangle(frame, (x, y), (x + cell_sz, y + cell_sz), (60, 60, 60), 1)

        return frame

    def get_stats(self) -> Dict:
        return {
            "phase":        self._phase,
            "ready":        self._ready,
            "frame_count":  self._frame_count,
            "has_target":   self._target_object is not None,
            "target_label": self._target_object.get("label", "none") if self._target_object else "none",
            "haptic_sends": self._haptic._send_count,
        }

    def release(self):
        if self._hands:
            self._hands.close()
        self._haptic.send_all_off()
        self._haptic.disconnect()