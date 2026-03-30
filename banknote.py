# =============================================================================
# banknote.py — ECHORA Egyptian Banknote Recognition
# =============================================================================
# Detects and classifies Egyptian Pound banknotes using a YOLOv8 model
# trained on the Banha University Egyptian Currency dataset.
#
# Six denominations: 5, 10, 20, 50, 100, 200 EGP
#
# Two-phase design:
#   Phase 1: detect_banknote() — fast check, runs every frame
#             tells state machine whether a note is visible
#   Phase 2: classify_denomination() — full classification
#             runs only when in BANKNOTE mode
#
# Calibrated for glasses-mounted camera:
#   User holds note at 20-50cm from face
#   BANKNOTE_MAX_DIST_MM = 500mm
#   Requires BANKNOTE_STABILITY_FRAMES consecutive detections
# =============================================================================
# =============================================================================
# IMPORTS
# =============================================================================

# YOLO from ultralytics — same library we use for obstacle detection.
from ultralytics import YOLO

# numpy for frame operations.
import numpy as np

# cv2 for frame preprocessing.
import cv2

# time for performance timing.
import time

# collections.deque for denomination stability tracking.
# Same approach as OCR — confirm across N frames before speaking.
from collections import deque

# Type hints.
from typing import Optional, Dict, List

# Our config and utils.
from config import (
    BANKNOTE_MODEL_PATH,
    BANKNOTE_CONFIDENCE_THRESHOLD,
    BANKNOTE_STABILITY_FRAMES,
    BANKNOTE_MAX_DIST_MM,
    DEPTH_MIN_MM,
)
from utils import logger, get_timestamp_ms, depth_in_region, bbox_center


# =============================================================================
# DENOMINATION MAPPING
# =============================================================================
# Maps raw YOLO class names to clean spoken announcements.
#
# The exact class names depend on what the Roboflow dataset used.
# Common formats found in Egyptian currency datasets:
#   "5", "10", "20", "50", "100", "200"
#   "5_EGP", "10_EGP", "20_EGP", etc.
#   "five", "ten", "twenty", "fifty", "hundred", "two_hundred"
#
# We map ALL possible formats to the same clean spoken string.
# This makes the code robust regardless of which dataset naming
# convention the trained model used.
#
# After you train and check model.names, if any class name is missing
# here, just add it to the dictionary — no other code needs to change.

DENOMINATION_MAP = {
    # Numeric string format
    "5":           "5 pounds",
    "10":          "10 pounds",
    "20":          "20 pounds",
    "50":          "50 pounds",
    "100":         "100 pounds",
    "200":         "200 pounds",

    # EGP suffix format
    "5_EGP":       "5 pounds",
    "10_EGP":      "10 pounds",
    "20_EGP":      "20 pounds",
    "50_EGP":      "50 pounds",
    "100_EGP":     "100 pounds",
    "200_EGP":     "200 pounds",

    # Underscore format
    "5_pounds":    "5 pounds",
    "10_pounds":   "10 pounds",
    "20_pounds":   "20 pounds",
    "50_pounds":   "50 pounds",
    "100_pounds":  "100 pounds",
    "200_pounds":  "200 pounds",

    # Word format
    "five":          "5 pounds",
    "ten":           "10 pounds",
    "twenty":        "20 pounds",
    "fifty":         "50 pounds",
    "hundred":       "100 pounds",
    "two_hundred":   "200 pounds",

    # Hyphen format
    "5-pounds":    "5 pounds",
    "10-pounds":   "10 pounds",
    "20-pounds":   "20 pounds",
    "50-pounds":   "50 pounds",
    "100-pounds":  "100 pounds",
    "200-pounds":  "200 pounds",

    # Roboflow sometimes adds spaces
    "5 pounds":    "5 pounds",
    "10 pounds":   "10 pounds",
    "20 pounds":   "20 pounds",
    "50 pounds":   "50 pounds",
    "100 pounds":  "100 pounds",
    "200 pounds":  "200 pounds",
}


# =============================================================================
# BANKNOTE DETECTOR CLASS
# =============================================================================

class BanknoteDetector:
    """
    Detects and classifies Egyptian Pound banknotes.

    Uses a YOLOv8 model trained on Egyptian currency images.
    Designed for glasses-mounted camera — user holds note at 20-50cm.

    Usage:
        detector = BanknoteDetector()
        detector.load_model()

        # Every frame — feeds state machine:
        is_note = detector.detect_banknote(rgb_frame)

        # In BANKNOTE mode only — full classification:
        denomination = detector.classify_denomination(rgb_frame)
        # Returns e.g. "50 pounds" or "" if not confident
    """

    def __init__(self):
        """
        Creates the detector. Does NOT load the model yet.
        Call load_model() after creating this object.
        """

        # The YOLO model — None until load_model() runs.
        self._model: Optional[YOLO] = None

        # Whether the model loaded successfully.
        self._ready: bool = False

        # Whether we are running in stub mode (no model file found).
        # Stub mode = always returns False/empty — system keeps running.
        self._stub_mode: bool = False

        # The device to run inference on (cpu/mps/cuda).
        # Auto-detected in load_model().
        self._device: str = "cpu"

        # ── Stability tracking ─────────────────────────────────────────────────
        # deque stores the last BANKNOTE_STABILITY_FRAMES denomination results.
        # When all N frames agree on the same denomination → speak it.
        # Same pattern as OCR stability — prevents announcing from one bad frame.
        self._denomination_history: deque = deque(
            maxlen=BANKNOTE_STABILITY_FRAMES
        )

        # The last denomination we spoke — prevents repeating.
        self._last_spoken: str = ""

        # ── Performance tracking ───────────────────────────────────────────────
        self._detect_count:  int   = 0
        self._success_count: int   = 0
        self._avg_infer_ms:  float = 0.0

        logger.info("BanknoteDetector created. Call load_model() to start.")


    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def load_model(self):
        """
        Loads the Egyptian banknote YOLOv8 model from disk.

        If the model file doesn't exist yet (still training on Colab),
        the detector enters stub mode — it logs a warning and keeps
        running without crashing. The moment banknote_egp.pt appears in the
        models folder and the system restarts, real detection activates.

        Also auto-detects the best available device (CPU/MPS/CUDA).
        """

        # ── Check if model file exists ─────────────────────────────────────────
        if not BANKNOTE_MODEL_PATH.exists():
            logger.warning(
                f"Banknote model not found at: {BANKNOTE_MODEL_PATH}\n"
                f"Running in STUB MODE — no banknote detection.\n"
                f"Drop banknote_egp.pt into models/ folder and restart to activate."
            )
            self._stub_mode = True
            self._ready     = True   # ready in stub mode — won't crash
            return

        # ── Detect best available compute device ───────────────────────────────
        import torch

        if torch.backends.mps.is_available():
            self._device = "mps"
            logger.info("Banknote detector will use Apple MPS GPU.")
        elif torch.cuda.is_available():
            self._device = "cuda"
            logger.info("Banknote detector will use CUDA GPU.")
        else:
            self._device = "cpu"
            logger.info("Banknote detector will use CPU.")

        # ── Load the YOLO model ────────────────────────────────────────────────
        logger.info(f"Loading banknote model from: {BANKNOTE_MODEL_PATH}")

        try:
            # YOLO() loads the model weights from the .pt file.
            self._model = YOLO(str(BANKNOTE_MODEL_PATH))

            # Log the class names so we can verify denomination mapping.
            logger.info(
                f"Banknote model loaded. "
                f"Classes: {self._model.names}"
            )

            # Warn about any unmapped class names.
            # If a class name is not in DENOMINATION_MAP, we can't announce it.
            for class_name in self._model.names.values():
                if class_name not in DENOMINATION_MAP:
                    logger.warning(
                        f"Unmapped banknote class: '{class_name}'. "
                        f"Add it to DENOMINATION_MAP in banknote.py."
                    )

        except Exception as e:
            logger.error(f"Failed to load banknote model: {e}")
            logger.warning("Falling back to stub mode.")
            self._stub_mode = True

        self._ready = True
        logger.info(
            "BanknoteDetector ready "
            f"({'stub mode' if self._stub_mode else 'real detection'})."
        )


    # =========================================================================
    # MAIN PUBLIC FUNCTIONS
    # =========================================================================

    def detect_banknote(self, rgb_frame: np.ndarray) -> bool:
        """
        Fast check — is a banknote visible in this frame?

        Called every frame by control_unit.py to feed the state machine.
        Returns True/False only — does NOT classify the denomination.
        This is intentionally lightweight so it runs every frame.

        The state machine uses this to decide when to switch to
        BANKNOTE mode. Once in BANKNOTE mode, classify_denomination()
        runs for the full classification.

        Arguments:
            rgb_frame: numpy array (H, W, 3) — current RGB frame

        Returns:
            True if a banknote is detected with sufficient confidence.
            False if no banknote found or in stub mode.
        """

        # Stub mode — always return False.
        if self._stub_mode or self._model is None:
            return False

        try:
            # Run YOLO inference.
            # We use a lower resolution for this fast check.
            # imgsz=320 is half of 640 — 4x fewer pixels = much faster.
            results = self._model(
                rgb_frame,
                verbose = False,
                conf    = BANKNOTE_CONFIDENCE_THRESHOLD,
                imgsz   = 320,
                device  = self._device,
                half = True
            )

            result = results[0]

            # If any detection exists → banknote is visible.
            # We don't care which denomination yet — just presence.
            if result.boxes is not None and len(result.boxes) > 0:
                return True

            return False

        except Exception as e:
            logger.error(f"Banknote detection error: {e}")
            return False


    def classify_denomination(self, rgb_frame: np.ndarray) -> str:
        """
        Full classification — what denomination is this banknote?

        Called only when in BANKNOTE mode by control_unit.py.
        Runs at full resolution for maximum accuracy.
        Uses stability checking — same denomination must appear in
        BANKNOTE_STABILITY_FRAMES consecutive frames before speaking.

        Arguments:
            rgb_frame: numpy array (H, W, 3) — current RGB frame

        Returns:
            Clean denomination string ready to speak, e.g. "50 pounds"
            Returns "" if:
              - No banknote detected
              - Not stable yet (denomination changing between frames)
              - Same as last announced denomination
              - In stub mode
        """

        # Stub mode — always return empty string.
        if self._stub_mode or self._model is None:
            return ""

        self._detect_count += 1
        start_ms = get_timestamp_ms()

        try:
            # ── Run full YOLO inference at full resolution ──────────────────────
            # imgsz=640 gives maximum accuracy for classification.
            results = self._model(
                rgb_frame,
                verbose = False,
                conf    = BANKNOTE_CONFIDENCE_THRESHOLD,
                imgsz   = 640,
                device  = self._device,
                half=True
            )

            result = results[0]

            # ── Extract the best detection ─────────────────────────────────────
            # We expect at most one banknote in frame at a time.
            # If multiple detected, take the highest-confidence one.
            if result.boxes is None or len(result.boxes) == 0:
                # No banknote detected this frame.
                self._denomination_history.append("")
                return ""

            # Find the detection with the highest confidence score.
            # result.boxes.conf is a tensor of confidence scores.
            # .argmax() returns the index of the highest value.
            best_idx   = int(result.boxes.conf.argmax())
            best_box   = result.boxes[best_idx]

            # Extract class index and confidence.
            class_idx  = int(best_box.cls.item())
            confidence = float(best_box.conf.item())

            # Get the raw class name from the model.
            # self._model.names is a dict: {0: "50", 1: "100", ...}
            raw_class = self._model.names[class_idx]

            # Map raw class name to clean spoken denomination.
            # .get() returns None if the key is not in the dictionary.
            denomination = DENOMINATION_MAP.get(raw_class)

            if denomination is None:
                # Unknown class name — log it and treat as no detection.
                logger.warning(
                    f"Unknown banknote class: '{raw_class}'. "
                    f"Add it to DENOMINATION_MAP."
                )
                self._denomination_history.append("")
                return ""

            logger.debug(
                f"Banknote detected: {raw_class} → '{denomination}' "
                f"(conf={confidence:.2f})"
            )

            # ── Add to stability history ────────────────────────────────────────
            # We require the SAME denomination in all recent frames.
            self._denomination_history.append(denomination)

            # ── Check stability ────────────────────────────────────────────────
            if not self._is_stable():
                # Denomination is still changing — don't speak yet.
                logger.debug(
                    f"Denomination not stable yet: {denomination}"
                )
                return ""

            # ── Check for new denomination ─────────────────────────────────────
            # Don't repeat the same denomination we already announced.
            if denomination == self._last_spoken:
                return ""

            # ── Stable new denomination — return it ────────────────────────────
            self._last_spoken = denomination
            self._success_count += 1

            # Update rolling average inference time.
            elapsed = get_timestamp_ms() - start_ms
            self._avg_infer_ms = (self._avg_infer_ms * 0.8) + (elapsed * 0.2)

            logger.info(f"Banknote classified: {denomination}")
            return denomination

        except Exception as e:
            logger.error(f"Banknote classification error: {e}")
            return ""


    # =========================================================================
    # DEPTH-AWARE DETECTION
    # =========================================================================

    def is_note_in_range(
        self,
        rgb_frame: np.ndarray,
        depth_map: np.ndarray
    ) -> bool:
        """
        Checks if a detected banknote is within the expected holding distance.

        When worn on glasses, the user holds the note at 20-50cm from face.
        Notes farther than BANKNOTE_MAX_DIST_MM (500mm) are probably on a
        table or counter — not being held for scanning. We ignore those.

        This prevents accidentally entering BANKNOTE mode when the camera
        sees a note on a table while the user is walking past.

        Arguments:
            rgb_frame: current RGB frame
            depth_map: depth map from camera

        Returns:
            True if a banknote is detected AND within holding distance.
        """

        if self._stub_mode or self._model is None:
            return False

        try:
            # Run fast detection first.
            results = self._model(
                rgb_frame,
                verbose = False,
                conf    = BANKNOTE_CONFIDENCE_THRESHOLD,
                imgsz   = 320,
                device  = self._device,
            )

            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                return False

            # Check depth for the best detection.
            best_idx = int(result.boxes.conf.argmax())
            best_box = result.boxes[best_idx]

            # Get bounding box coordinates.
            xyxy = best_box.xyxy.cpu().numpy()[0]
            x1   = int(xyxy[0])
            y1   = int(xyxy[1])
            x2   = int(xyxy[2])
            y2   = int(xyxy[3])

            # Sample depth at the center of the banknote.
            dist_mm = depth_in_region(depth_map, x1, y1, x2, y2)

            if dist_mm <= 0:
                # No valid depth — assume it might be in range.
                # Better to trigger and check than to miss a real note.
                return True

            # Check if within holding distance.
            in_range = dist_mm <= BANKNOTE_MAX_DIST_MM

            if not in_range:
                logger.debug(
                    f"Banknote detected but too far: {dist_mm:.0f}mm "
                    f"(max={BANKNOTE_MAX_DIST_MM}mm)"
                )

            return in_range

        except Exception as e:
            logger.error(f"Banknote range check error: {e}")
            return False


    # =========================================================================
    # STABILITY CHECKING
    # =========================================================================

    def _is_stable(self) -> bool:
        """
        Returns True if the same denomination appeared in all recent frames.

        Requires BANKNOTE_STABILITY_FRAMES consecutive frames with the
        same non-empty denomination before we consider it confirmed.

        Same logic as OCR stability — prevents announcing from one bad frame.
        """

        # Not enough history yet.
        if len(self._denomination_history) < BANKNOTE_STABILITY_FRAMES:
            return False

        history_list = list(self._denomination_history)

        # All entries must be the same non-empty string.
        # set() removes duplicates — one element means all values identical.
        unique = set(history_list)

        return (
            len(unique) == 1
            and list(unique)[0] != ""
        )


    # =========================================================================
    # DEBUG OVERLAY
    # =========================================================================

    def draw_debug_overlay(
        self,
        frame: np.ndarray
    ) -> np.ndarray:
        """
        Draws banknote detection results on the debug frame.

        Shows:
          - Bounding box around detected banknote
          - Denomination label and confidence
          - Stability counter

        Arguments:
            frame: RGB frame to draw on

        Returns:
            Frame with debug overlay.
        """

        if self._stub_mode or self._model is None:
            cv2.putText(
                frame, "BANKNOTE: stub mode",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (120, 120, 120), 1
            )
            return frame

        try:
            results = self._model(
                frame,
                verbose = False,
                conf    = BANKNOTE_CONFIDENCE_THRESHOLD,
                imgsz   = 640,
                device  = self._device,
            )

            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                cv2.putText(
                    frame, "BANKNOTE: no note detected",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (120, 120, 120), 1
                )
                return frame

            for box in result.boxes:
                xyxy       = box.xyxy.cpu().numpy()[0]
                x1, y1     = int(xyxy[0]), int(xyxy[1])
                x2, y2     = int(xyxy[2]), int(xyxy[3])
                confidence = float(box.conf.item())
                class_idx  = int(box.cls.item())
                raw_class  = self._model.names[class_idx]
                denomination = DENOMINATION_MAP.get(raw_class, raw_class)

                # Draw bounding box in gold colour.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 2)

                # Draw denomination label above the box.
                label = f"{denomination} ({confidence:.0%})"
                cv2.putText(
                    frame, label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 215, 255), 2
                )

            # Draw stability counter in bottom left.
            stability_str = (
                f"Stable: {len(self._denomination_history)}/"
                f"{BANKNOTE_STABILITY_FRAMES}"
            )
            cv2.putText(
                frame, stability_str,
                (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 215, 255), 1
            )

        except Exception as e:
            logger.error(f"Banknote overlay error: {e}")

        return frame


    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def reset(self):
        """
        Resets the detector state.

        Called when exiting BANKNOTE mode — clears stability history
        so the same denomination can be announced again next time.
        """

        self._denomination_history.clear()
        self._last_spoken = ""
        logger.debug("BanknoteDetector state reset.")


    def get_stats(self) -> Dict:
        """Returns diagnostic statistics."""
        return {
            "ready":        self._ready,
            "stub_mode":    self._stub_mode,
            "detect_count": self._detect_count,
            "success_count": self._success_count,
            "avg_infer_ms": round(self._avg_infer_ms, 1),
            "last_spoken":  self._last_spoken,
            "stability":    len(self._denomination_history),
        }


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================
# Same pattern as ocr.py — one shared instance, model loaded once.

_detector: Optional[BanknoteDetector] = None


def init_banknote():
    """
    Initialises the module-level banknote detector.
    Call once at startup from control_unit.py.
    """

    global _detector

    if _detector is not None:
        logger.debug("Banknote detector already initialised.")
        return

    _detector = BanknoteDetector()
    _detector.load_model()
    logger.info("Module-level banknote detector ready.")


def detect_banknote(rgb_frame: np.ndarray) -> bool:
    """
    Module-level function — is a banknote visible?
    Called every frame by control_unit.py to feed the state machine.
    """

    if _detector is None:
        return False

    return _detector.detect_banknote(rgb_frame)


def classify_denomination(rgb_frame: np.ndarray) -> str:
    """
    Module-level function — what denomination is this note?
    Called only in BANKNOTE mode by control_unit.py.
    """

    if _detector is None:
        return ""

    return _detector.classify_denomination(rgb_frame)


def reset_banknote():
    """Module-level reset — called when exiting BANKNOTE mode."""

    if _detector is not None:
        _detector.reset()


# =============================================================================
# SELF-TEST
# =============================================================================
# Run with: python banknote.py
# Hold an Egyptian banknote in front of the camera.

if __name__ == "__main__":

    print("=== ECHORA banknote.py self-test ===\n")

    from camera import EchoraCamera

    cam      = EchoraCamera()
    detector = BanknoteDetector()

    try:
        print("Starting camera and loading model...")
        cam.init_pipeline()
        detector.load_model()

        if detector._stub_mode:
            print(
                "Running in STUB MODE — model file not found.\n"
                f"Expected: {BANKNOTE_MODEL_PATH}\n"
                "Drop banknote_egp.pt into models/ folder and restart."
            )
        else:
            print("Model loaded. Class names from model:")
            for idx, name in detector._model.names.items():
                spoken = DENOMINATION_MAP.get(name, f"UNMAPPED: {name}")
                print(f"  Class {idx}: '{name}' → '{spoken}'")

        print("\nHold an Egyptian banknote in front of the camera.")
        print("Press Q to quit.\n")

        frame_count = 0

        while True:
            bundle = cam.get_synced_bundle()
            if bundle is None:
                continue

            frame_count += 1
            rgb   = bundle["rgb"]
            depth = bundle["depth"]

            # Check if note is visible and in range.
            note_visible = detector.is_note_in_range(rgb, depth)

            # Try to classify denomination.
            denomination = ""
            if note_visible:
                denomination = detector.classify_denomination(rgb)

            # Print status every 10 frames.
            if frame_count % 10 == 0:
                print(
                    f"Frame {frame_count:4d} | "
                    f"Note visible: {'YES' if note_visible else 'no ':3s} | "
                    f"Denomination: '{denomination}'"
                )

            # Draw debug overlay.
            debug = detector.draw_debug_overlay(rgb.copy())

            # Show depth at frame center for reference.
            h, w = depth.shape
            center_depth = depth[h//2, w//2]
            cv2.putText(
                debug,
                f"Center depth: {center_depth}mm",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1
            )

            cv2.imshow("ECHORA Banknote Test", debug)
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        cv2.destroyAllWindows()
        cam.release()
        stats = detector.get_stats()
        print(f"\nFinal stats: {stats}")
        print("\n=== Self-test complete ===")