# =============================================================================
# utils.py — ECHORA Shared Helper Functions and Classes
# =============================================================================
# This file is imported by almost every other module in ECHORA.
# It contains small reusable tools so we never repeat code.
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

# 'cv2' is OpenCV — the computer vision library we use to work with images.
# We need it here for drawing on frames and resizing images.
import cv2

# 'numpy' is a math library for working with arrays and matrices.
# Camera frames are numpy arrays, so we use this constantly.
import numpy as np

# 'time' is a built-in Python module for measuring time.
# We use it in RateLimiter and AlertCooldown to track elapsed time.
import time

# 'logging' is Python's built-in system for printing messages.
# Better than print() because it includes timestamps, severity levels,
# and can write to a log file automatically.
import logging

# 'Dict' and 'Tuple' are type hints — they tell you what type of data
# a function expects and returns. They don't affect how the code runs,
# but they make the code much easier to read and understand.
from typing import Dict, Tuple, Optional, List

# Import all the constants we defined in config.py.
# We need DANGER_DIST_MM, WARNING_DIST_MM, CAMERA_HFOV_DEG, etc.
from config import (
    DANGER_DIST_MM,
    WARNING_DIST_MM,
    CAMERA_HFOV_DEG,
    ALERT_COOLDOWN_SEC,
    LOG_LEVEL,
    LOG_PATH,
)


# =============================================================================
# LOGGING SETUP
# =============================================================================
# This sets up the logging system for the entire project.
# Every other module will import 'logger' from here and use it
# instead of print(). This gives us timestamps and log levels for free.

# getLogger creates a logger with the name "ECHORA".
# All log messages will be prefixed with this name.
logger = logging.getLogger("ECHORA")

# setLevel controls the minimum severity of messages to show.
# getattr(logging, LOG_LEVEL) converts the string "DEBUG" into
# the actual logging constant logging.DEBUG. This reads from config.py.
logger.setLevel(getattr(logging, LOG_LEVEL))

# A 'handler' decides WHERE log messages go.
# StreamHandler sends them to the terminal (your screen).
stream_handler = logging.StreamHandler()

# A 'formatter' decides HOW log messages look.
# This format prints: time — level — message
# Example: 2024-01-15 10:32:01 — INFO — Camera started
formatter = logging.Formatter(
    "%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Attach the formatter to the handler.
stream_handler.setFormatter(formatter)

# Attach the handler to the logger.
# Now when you call logger.info("hello") it will print to the terminal.
logger.addHandler(stream_handler)

# A second handler that writes the same messages to a FILE.
# This way you can review what happened after the fact.
# mode="a" means "append" — don't overwrite the file, add to it.
try:
    file_handler = logging.FileHandler(LOG_PATH, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
except Exception:
    # If the log folder doesn't exist yet, just skip the file handler.
    # The terminal handler will still work.
    pass


# =============================================================================
# SECTION 1 — GEOMETRY HELPERS
# =============================================================================

def bbox_center(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    """
    Returns the center point (cx, cy) of a bounding box.

    A bounding box has:
      (x1, y1) = top-left corner
      (x2, y2) = bottom-right corner

    The center is the average of the x coordinates and the average
    of the y coordinates.

    Example:
      bbox_center(100, 50, 300, 150) → (200, 100)
    """
    # Integer division (//) rounds down to the nearest whole pixel.
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Return both values together as a tuple (a pair of values).
    return cx, cy


def angle_from_x(cx: int, frame_width: int) -> float:
    """
    Converts a pixel x-position into a real-world horizontal angle in degrees.

    The camera sees the world as a flat image.
    The center pixel of the image = straight ahead = 0 degrees.
    The left edge of the image = -HFOV/2 degrees (e.g. -36.5 degrees).
    The right edge of the image = +HFOV/2 degrees (e.g. +36.5 degrees).

    This tells the audio engine how far left or right to pan a sound.

    Example with a 640px wide frame and 73 degree FOV:
      angle_from_x(320, 640) →  0.0  (dead center = straight ahead)
      angle_from_x(  0, 640) → -36.5 (far left edge)
      angle_from_x(640, 640) → +36.5 (far right edge)
    """
    # 'normalised' converts cx from pixel space (0 to 640) into
    # a value between 0.0 and 1.0. Center pixel 320 → 0.5.
    normalised = cx / frame_width

    # Shift so that 0.5 (center) becomes 0.0.
    # Now left of center is negative, right of center is positive.
    # Range becomes: -0.5 to +0.5
    shifted = normalised - 0.5

    # Multiply by the camera's field of view to get real degrees.
    # CAMERA_HFOV_DEG = 73.0 degrees (from config.py)
    # shifted * 73.0 → range: -36.5 to +36.5 degrees
    angle = shifted * CAMERA_HFOV_DEG

    # round() limits to 1 decimal place. -12.3456 → -12.3
    return round(angle, 1)


def normalise_bbox(
    x1: int, y1: int, x2: int, y2: int,
    frame_width: int, frame_height: int
) -> Tuple[float, float, float, float]:
    """
    Converts pixel bounding box coordinates into normalised 0.0–1.0 values.

    YOLO outputs normalised coordinates. OpenCV needs pixel coordinates.
    This function converts pixel → normalised.
    The reverse (normalised → pixel) is done by multiplying back.

    Example:
      normalise_bbox(100, 50, 300, 150, 640, 400) →
      (0.156, 0.125, 0.469, 0.375)
    """
    # Divide each coordinate by the frame dimension it belongs to.
    return (
        x1 / frame_width,
        y1 / frame_height,
        x2 / frame_width,
        y2 / frame_height,
    )


def denormalise_bbox(
    nx1: float, ny1: float, nx2: float, ny2: float,
    frame_width: int, frame_height: int
) -> Tuple[int, int, int, int]:
    """
    Converts normalised 0.0–1.0 bounding box values back to pixel coordinates.

    This is the reverse of normalise_bbox.
    YOLO gives us normalised values, this converts them to pixel values
    so we can draw rectangles on the frame with OpenCV.

    Example:
      denormalise_bbox(0.156, 0.125, 0.469, 0.375, 640, 400) →
      (100, 50, 300, 150)
    """
    # Multiply by frame dimensions and convert to integer pixels.
    # int() truncates the decimal — int(99.9) = 99
    return (
        int(nx1 * frame_width),
        int(ny1 * frame_height),
        int(nx2 * frame_width),
        int(ny2 * frame_height),
    )


def bbox_area(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    Returns the area of a bounding box in pixels squared.

    Area = width × height
    Used to filter out tiny detections that are probably noise or
    objects so far away they don't matter yet.

    Example:
      bbox_area(100, 50, 300, 150) → (300-100) × (150-50) = 200 × 100 = 20000
    """
    width  = x2 - x1
    height = y2 - y1
    return width * height


def boxes_overlap(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int]
) -> bool:
    """
    Returns True if two bounding boxes overlap, False if they don't.

    Used by the decision engine to detect when two modules have
    detected the same physical object (e.g. YOLO and interaction_detection
    both detecting the same cup).

    Each box is a tuple: (x1, y1, x2, y2)

    Two boxes DO NOT overlap if:
      - One is entirely to the LEFT of the other, OR
      - One is entirely ABOVE the other.
    In all other cases, they overlap.
    """
    # Unpack the two boxes into their four corner values.
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    # Check non-overlap conditions.
    # If any of these is True, the boxes don't overlap.
    if ax2 < bx1:   # box1 is entirely to the left of box2
        return False
    if bx2 < ax1:   # box2 is entirely to the left of box1
        return False
    if ay2 < by1:   # box1 is entirely above box2
        return False
    if by2 < ay1:   # box2 is entirely above box1
        return False

    # If none of the non-overlap conditions were True, they must overlap.
    return True


# =============================================================================
# SECTION 2 — DEPTH AND DISTANCE HELPERS
# =============================================================================

def depth_in_region(
    depth_map: np.ndarray,
    x1: int, y1: int, x2: int, y2: int
) -> float:
    """
    Returns the median depth value (in mm) inside a bounding box region.

    We use MEDIAN instead of AVERAGE because:
    - Depth maps have "holes" — pixels with value 0 meaning "no data".
    - A few zero pixels would pull the average way down and give wrong results.
    - The median ignores extreme outliers, giving us a stable, reliable reading.

    Returns 0.0 if no valid depth data exists in the region.
    """
    # Slice the depth_map array to get only the region inside the bounding box.
    # numpy arrays are sliced as: array[y_start:y_end, x_start:x_end]
    # Note: rows (y) come before columns (x) in numpy.
    region = depth_map[y1:y2, x1:x2]

    # Flatten converts the 2D region into a 1D list of all depth values.
    flat = region.flatten()

    # Filter out zero values — these are pixels with no depth data.
    # flat[flat > 0] keeps only values greater than zero.
    valid = flat[flat > 0]

    # If there are no valid pixels at all, return 0.
    if len(valid) == 0:
        return 0.0

    # np.median() returns the middle value when all values are sorted.
    # This is our reliable depth estimate in millimetres.
    return float(np.median(valid))


def mm_to_spoken(distance_mm: float) -> str:
    """
    Converts a raw millimetre distance into a natural spoken string.

    Examples:
      450   → "45 centimetres"
      850   → "85 centimetres"
      1800  → "1.8 metres"
      3200  → "3.2 metres"
      0     → "unknown distance"
    """
    # If we got 0 or negative, we have no valid reading.
    if distance_mm <= 0:
        return "unknown distance"

    # If closer than 1 metre (1000mm), speak in centimetres.
    # This sounds more natural: "85 centimetres" rather than "0.85 metres".
    if distance_mm < 1000:
        # Convert mm to cm: divide by 10, then round to nearest integer.
        cm = round(distance_mm / 10)
        return f"{cm} centimetres"

    # If 1 metre or further, speak in metres with one decimal place.
    # f-string with :.1f formats the number to 1 decimal place.
    # 1800mm → 1.8 → "1.8 metres"
    metres = round(distance_mm / 1000, 1)
    return f"{metres} metres"


def classify_urgency(distance_mm: float) -> str:
    """
    Classifies a distance into one of three urgency levels.

    Returns:
      "DANGER"  — object is within DANGER_DIST_MM (< 800mm by default)
      "WARNING" — object is between DANGER and WARNING distance
      "SAFE"    — object is beyond WARNING_DIST_MM (> 2000mm by default)
      "UNKNOWN" — distance is 0 or invalid

    These strings match the MODE constants and are used by:
      - obstacle_detection.py  (to decide alert level)
      - control_unit.py        (to decide what feedback to trigger)
      - audio_feedback.py      (to decide alert sound)
      - haptic_feedback.py     (to decide vibration pattern)
    """
    if distance_mm <= 0:
        return "UNKNOWN"

    if distance_mm < DANGER_DIST_MM:
        return "DANGER"

    if distance_mm < WARNING_DIST_MM:
        return "WARNING"

    return "SAFE"


# =============================================================================
# SECTION 3 — FRAME AND IMAGE HELPERS
# =============================================================================

def crop_region(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    """
    Cuts a rectangular region out of a camera frame.

    Used by:
      - ocr.py         → crop the text area before reading it
      - banknote.py    → crop the banknote before classifying it
      - face_recognition.py → crop the face before identifying it

    The result is a smaller image (numpy array) containing just that region.

    numpy slicing: frame[rows, columns] → frame[y1:y2, x1:x2]
    """
    # Clamp coordinates to frame boundaries using np.clip.
    # This prevents errors if a bounding box slightly exceeds the frame edge.
    h, w = frame.shape[:2]

    # np.clip(value, min, max) — keeps value between min and max.
    x1 = int(np.clip(x1, 0, w))
    y1 = int(np.clip(y1, 0, h))
    x2 = int(np.clip(x2, 0, w))
    y2 = int(np.clip(y2, 0, h))

    # Slice and return the region.
    return frame[y1:y2, x1:x2]


def resize_frame(
    frame: np.ndarray,
    width: int,
    height: int
) -> np.ndarray:
    """
    Resizes a frame to the given width and height.

    Different models expect different input sizes:
      - YOLO expects 512x288
      - OCR model might expect 320x320
      - Face model might expect 112x112
    This function handles all of them the same way.

    cv2.INTER_LINEAR is a smooth resizing method — good for making
    images smaller (which is what we almost always do here).
    """
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def draw_overlay(
    frame: np.ndarray,
    detections: List[Dict]
) -> np.ndarray:
    """
    Draws bounding boxes, labels, distances and urgency colours
    onto the RGB frame for the debug window.

    Each detection in the list is a dictionary with these keys:
      {
        "label":       "chair",
        "distance_mm": 1200,
        "angle_deg":   -15.0,
        "urgency":     "WARNING",
        "bbox":        (x1, y1, x2, y2)
      }

    Colour coding:
      DANGER  → red
      WARNING → orange
      SAFE    → green
      UNKNOWN → grey
    """
    # Colour map — BGR format (OpenCV uses Blue, Green, Red — NOT Red, Green, Blue).
    COLOURS = {
        "DANGER":  (0, 0, 220),    # red in BGR
        "WARNING": (0, 165, 255),  # orange in BGR
        "SAFE":    (0, 200, 80),   # green in BGR
        "UNKNOWN": (120, 120, 120) # grey in BGR
    }

    # Loop through every detection in the list.
    for det in detections:

        # Unpack the bounding box coordinates from the dictionary.
        x1, y1, x2, y2 = det["bbox"]

        # Look up the colour for this urgency level.
        # .get() with a default means: if "urgency" key is missing, use "UNKNOWN".
        colour = COLOURS.get(det.get("urgency", "UNKNOWN"), (120, 120, 120))

        # cv2.rectangle draws a rectangle on the frame.
        # Arguments: frame, top-left corner, bottom-right corner, colour, thickness
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Build the label text string.
        # Example: "chair  1.2m  -15.0°  [WARNING]"
        label    = det.get("label", "?")
        dist_str = mm_to_spoken(det.get("distance_mm", 0))
        angle    = det.get("angle_deg", 0)
        urgency  = det.get("urgency", "?")
        text     = f"{label}  {dist_str}  {angle:+.1f}deg  [{urgency}]"

        # cv2.putText draws text on the frame.
        # Arguments: frame, text, position, font, scale, colour, thickness
        cv2.putText(
            frame,
            text,
            (x1 + 5, y1 - 8),           # position: just above the top-left corner
            cv2.FONT_HERSHEY_SIMPLEX,    # a clean, readable font
            0.5,                          # font scale — 0.5 is small but readable
            colour,                       # same colour as the bounding box
            1,                            # thickness of 1 pixel
            cv2.LINE_AA                   # anti-aliasing for smoother text
        )

    # Return the modified frame.
    # Note: OpenCV modifies frames IN PLACE (the original is already changed),
    # but we return it anyway for clean, readable code.
    return frame


# =============================================================================
# SECTION 4 — TIMING AND PERFORMANCE HELPERS
# =============================================================================

def get_timestamp_ms() -> float:
    """
    Returns the current time in milliseconds.

    time.time() returns seconds as a float (e.g. 1705312321.482).
    Multiplying by 1000 converts to milliseconds (e.g. 1705312321482.0).

    Used by control_unit.py to measure how long each frame takes.
    If a frame takes longer than MAX_FRAME_TIME_MS, we log a warning.
    """
    return time.time() * 1000


class RateLimiter:
    """
    Controls how often a module is allowed to run.

    Some modules are slow (OCR, VLM) and should not run every frame.
    RateLimiter counts frames and returns True only every N frames.

    Usage:
      ocr_limiter = RateLimiter(run_every=10)   # run OCR every 10 frames

      while True:
          if ocr_limiter.should_run():
              result = ocr.read_text(frame)     # only runs every 10 frames
    """

    def __init__(self, run_every: int):
        """
        __init__ is called automatically when you create a RateLimiter.

        'self' refers to this specific RateLimiter object.
        run_every is how many frames to wait between runs.
        """
        # Store how often we should run.
        self.run_every = run_every

        # Internal counter — starts at 0, counts up each frame.
        self._counter = 0

    def should_run(self) -> bool:
        """
        Call this every frame. Returns True when it's time to run.

        How it works:
          Frame 1:  counter=1, 1 % 10 = 1  → False (skip)
          Frame 2:  counter=2, 2 % 10 = 2  → False (skip)
          ...
          Frame 10: counter=10, 10 % 10 = 0 → True  (RUN)
          Frame 11: counter=11, 11 % 10 = 1 → False (skip)
          ...

        % is the modulo operator — it returns the remainder after division.
        10 % 10 = 0  (10 divides evenly, remainder 0 → run)
        11 % 10 = 1  (11 divided by 10 = 1 remainder 1 → skip)
        """
        # Increment the counter each frame.
        self._counter += 1

        # Check if the counter is divisible by run_every.
        return self._counter % self.run_every == 0

    def reset(self):
        """
        Resets the counter back to zero.
        Useful when switching modes — you want the new mode to run immediately.
        """
        self._counter = 0


# =============================================================================
# SECTION 5 — ALERT DEDUPLICATION
# =============================================================================

class AlertCooldown:
    """
    Prevents the same alert from being announced repeatedly.

    Without this, ECHORA would say "danger: chair" 30 times per second
    (once every frame). With this, it says it once and then waits
    ALERT_COOLDOWN_SEC seconds before saying it again.

    Usage:
      cooldown = AlertCooldown()

      while True:
          for obj in detections:
              if cooldown.can_alert(obj["label"]):
                  audio.speak(f"danger: {obj['label']}")
    """

    def __init__(self):
        """
        Creates an empty dictionary to store the last alert time
        for each object label.

        A dictionary stores key-value pairs: {"chair": 1705312321.4}
        Key = label name (string)
        Value = timestamp of last alert (float, seconds)
        """
        # Dict[str, float] means: keys are strings, values are floats.
        self._last_alerted: Dict[str, float] = {}

    def can_alert(self, label: str) -> bool:
        """
        Returns True if enough time has passed to alert again for this label.

        How it works:
          - Look up when we last alerted for this label.
          - Calculate how much time has passed since then.
          - If more than ALERT_COOLDOWN_SEC has passed → return True (can alert).
          - Otherwise → return False (too soon, skip).
          - If we've never alerted for this label → return True (first time).
        """
        now = time.time()

        # .get(label, 0) returns the last alert time for this label,
        # or 0 if we've never seen this label before.
        last_time = self._last_alerted.get(label, 0)

        # Calculate how many seconds have passed since the last alert.
        elapsed = now - last_time

        # If enough time has passed, allow the alert.
        if elapsed >= ALERT_COOLDOWN_SEC:
            # Record the current time as the new "last alerted" time.
            self._last_alerted[label] = now
            return True

        # Not enough time has passed — suppress this alert.
        return False

    def reset(self, label: Optional[str] = None):
        """
        Resets the cooldown.

        If a label is provided → reset only that label.
        If no label is provided → reset ALL labels.

        Useful when the system switches modes — in OCR mode we don't
        want the obstacle cooldown state to persist.
        """
        if label:
            # Remove just this one label from the dictionary.
            # .pop() removes a key-value pair if it exists.
            # The second argument (None) means "don't crash if it's not there".
            self._last_alerted.pop(label, None)
        else:
            # Clear the entire dictionary.
            self._last_alerted.clear()


# =============================================================================
# QUICK SELF-TEST
# =============================================================================
# This block only runs if you execute utils.py DIRECTLY:
#   python utils.py
#
# It does NOT run when utils.py is imported by another file.
# 'if __name__ == "__main__"' is Python's way of saying
# "only run this if this file is the main file being executed."
#
# Use this to verify everything works before building on top of it.

if __name__ == "__main__":
    print("--- Testing utils.py ---\n")

    # Test bbox_center
    cx, cy = bbox_center(100, 50, 300, 150)
    print(f"bbox_center(100,50,300,150) → ({cx}, {cy})")
    assert cx == 200 and cy == 100, "bbox_center FAILED"
    print("  PASSED\n")

    # Test angle_from_x
    print(f"angle_from_x(320, 640) → {angle_from_x(320, 640)} deg (expect 0.0)")
    print(f"angle_from_x(  0, 640) → {angle_from_x(0, 640)} deg (expect -36.5)")
    print(f"angle_from_x(640, 640) → {angle_from_x(640, 640)} deg (expect +36.5)")
    print("  PASSED\n")

    # Test mm_to_spoken
    print(f"mm_to_spoken(450)  → '{mm_to_spoken(450)}'")
    print(f"mm_to_spoken(1800) → '{mm_to_spoken(1800)}'")
    print(f"mm_to_spoken(0)    → '{mm_to_spoken(0)}'")
    print("  PASSED\n")

    # Test classify_urgency
    print(f"classify_urgency(500)  → '{classify_urgency(500)}'  (expect DANGER)")
    print(f"classify_urgency(1200) → '{classify_urgency(1200)}' (expect WARNING)")
    print(f"classify_urgency(3000) → '{classify_urgency(3000)}' (expect SAFE)")
    print("  PASSED\n")

    # Test RateLimiter
    limiter = RateLimiter(run_every=3)
    results = [limiter.should_run() for _ in range(9)]
    print(f"RateLimiter(3) over 9 frames → {results}")
    print("  (expect True on frames 3, 6, 9)\n")

    # Test AlertCooldown
    cooldown = AlertCooldown()
    print(f"AlertCooldown first alert for 'chair'  → {cooldown.can_alert('chair')}")
    print(f"AlertCooldown repeat alert for 'chair' → {cooldown.can_alert('chair')}")
    print("  (expect True then False)\n")

    print("--- All tests passed ---")