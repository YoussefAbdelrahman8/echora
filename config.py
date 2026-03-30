# =============================================================================
# config.py — ECHORA Central Configuration
# =============================================================================
# This is the ONLY file where you set constants and settings.
# Every other file imports from here. Never hardcode numbers elsewhere.
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================
# 'os' is a built-in Python module that lets us work with file paths.
# We use it to build paths that work on any operating system (Mac, Windows, Linux).
import os

# 'Path' is a cleaner way to handle file and folder paths.
# Instead of writing "/users/youssef/echora/models" manually,
# we use Path to build the path automatically relative to this file.
from pathlib import Path


# =============================================================================
# 2. PROJECT ROOT PATH
# =============================================================================
# __file__ means "the path to THIS file (config.py)".
# .parent means "the folder that contains this file".
# So BASE_DIR will be something like: /users/youssef/echora/
# Every other path in the project is built on top of this.
BASE_DIR = Path(__file__).parent

# =============================================================================
# FACE RECOGNITION SETTINGS
# =============================================================================

# Path to the face embeddings database folder.
FACE_DB_PATH = BASE_DIR / "database" / "faces"

# Minimum face detection confidence to trigger FACE_ID mode.
# 0.3 = a reasonably visible face.
FACE_CONFIDENCE_THRESHOLD = 0.15

# How many consecutive frames the same name must appear before announcing.
FACE_STABILITY_FRAMES = 3

# Maximum distance between embeddings to count as a match.
# 0.5 is the recommended default from the face_recognition library.
# Lower = stricter (fewer false positives).
# Higher = looser (better for glasses, hats, different lighting).
FACE_RECOGNITION_TOLERANCE = 0.6
# =============================================================================
# 3. CAMERA SETTINGS
# =============================================================================
# These values are passed to camera.py when it starts the depthai pipeline.

# The width and height of the RGB frame in pixels.
# 640x400 is a good balance between detail and speed on the OAK-D Lite.
CAMERA_RGB_WIDTH  = 1280
CAMERA_RGB_HEIGHT = 800

# The width and height used for the stereo (depth) cameras.
# Must match the RGB dimensions for proper depth-to-RGB alignment.
CAMERA_DEPTH_WIDTH  = 1280
CAMERA_DEPTH_HEIGHT = 800

# How many frames per second the camera captures.
# 30 FPS means the whole pipeline must complete one cycle in under 33ms.
CAMERA_FPS = 30

# The maximum depth range the stereo camera will measure, in millimetres.
# Anything beyond 8 metres is too far to be relevant for navigation.
DEPTH_MAX_MM = 8000

# The minimum depth range — ignore anything closer than 10cm.
# Objects this close are usually the user's own body or the glasses frame.
DEPTH_MIN_MM = 100


# =============================================================================
# 4. DISTANCE THRESHOLDS
# =============================================================================
# These define the three urgency zones ECHORA uses.
# All values are in MILLIMETRES (1000mm = 1 metre).

# DANGER zone: object is less than 80cm away.
# Trigger: urgent haptic pulse + loud audio alert immediately.
DANGER_DIST_MM = 1500

# WARNING zone: object is between 80cm and 200cm away.
# Trigger: gentle audio warning, mild haptic buzz.
WARNING_DIST_MM = 3000

# SAFE zone: object is beyond 200cm.
# Trigger: no alert. Object is noted but not announced.
# (Anything beyond WARNING_DIST_MM is automatically considered safe.)


# =============================================================================
# 5. KALMAN FILTER PARAMETERS
# =============================================================================
# The Kalman filter smooths the noisy position data from the camera
# and predicts where a moving object will be in the NEXT frame.
#
# Think of it like this:
#   - The camera gives you a new position every frame, but it has noise/jitter.
#   - The Kalman filter combines the camera reading with its own prediction
#     to produce a smoother, more accurate estimate.
#
# PROCESS_NOISE: How much we trust our own movement prediction model.
#   - Higher value = we trust the prediction less, react faster to sensor changes.
#   - Lower value  = we trust the prediction more, smoother but slower to react.
KALMAN_PROCESS_NOISE = 0.03

# MEASUREMENT_NOISE: How much we trust the raw camera sensor readings.
#   - Higher value = we trust the camera less (more smoothing).
#   - Lower value  = we trust the camera more (less smoothing, more jitter).
KALMAN_MEASUREMENT_NOISE = 0.1

# How many frames of history the tracker keeps per object.
# If an object disappears for more than this many frames, we stop tracking it.
KALMAN_MAX_MISSED_FRAMES = 10


# =============================================================================
# 6. SYSTEM MODES
# =============================================================================
# These are the names of all the states the state_machine.py can be in.
# We store them as a class so you can write MODE.NAVIGATION instead of
# typing the string "NAVIGATION" every time (avoids typos).

class MODE:
    # Default mode. YOLO obstacle detection is active.
    # The system is helping the user walk safely.
    NAVIGATION = "NAVIGATION"

    # Activated when text is detected close enough to read.
    # OCR module takes priority. Obstacle detection pauses temporarily.
    OCR = "OCR"

    # Activated when a door handle, cup, elevator button, etc. is detected.
    # Hand guidance via haptic feedback is active.
    # Hand detection is also active.
    INTERACTION = "INTERACTION"

    # Activated when a face is detected in frame.
    # Face recognition module runs, looks up the person in the database.
    FACE_ID = "FACE_ID"

    # Activated when a banknote is detected in frame.
    # Currency classification module runs.
    BANKNOTE = "BANKNOTE"


# =============================================================================
# 7. DETECTION SETTINGS
# =============================================================================

# Confidence threshold for YOLO detections (0.0 to 1.0).
# A detection is only accepted if YOLO is at least this confident.
# 0.5 means 50% confident. Too low = lots of false positives.
# Too high = misses real objects. 0.5 is a safe starting point.
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# The YOLO model file. This is built relative to BASE_DIR.
# So the full path will be: /your/project/folder/models/yolov8n.tar.xz
YOLO_MODEL_PATH = BASE_DIR / "models" / "yolov8s.pt"
YOLO_INPUT_WIDTH  =416
YOLO_INPUT_HEIGHT = 256

# The COCO classes that actually matter for a blind user navigating the world.
# YOLO can detect 80 classes, but most are irrelevant (Frisbee, snowboard, etc.)
# We only alert on these. Everything else is silently ignored.
RELEVANT_CLASSES = [
    "person",           # most important — other people
    "bicycle",          # moving hazard
    "car",              # moving hazard
    "motorcycle",       # moving hazard
    "bus",              # moving hazard
    "truck",            # moving hazard
    "chair",            # stationary obstacle
    "couch",            # stationary obstacle
    "dining table",     # stationary obstacle
    "bed",              # stationary obstacle
    "toilet",           # useful landmark
    "sink",             # useful landmark
    "door",             # navigation target
    "stairs",           # high priority hazard
    "potted plant",     # common obstacle
    "dog",              # moving hazard
    "cat",              # moving hazard
    "bottle",           # interactable
    "cup",              # interactable
    "bowl",             # interactable
    "laptop",           # useful landmark
    "tv",               # useful landmark
    "book",             # potential OCR target
    "clock",            # useful info
    "cell phone",       # interactable
    "backpack",         # obstacle / belonging
    "umbrella",         # obstacle
    "bench",            # landmark / seating
    "fire hydrant",     # outdoor hazard
    "stop sign",        # outdoor landmark
    "traffic light",    # outdoor landmark
]

# Scale factor for the bounding box region used to calculate depth.
# 0.5 means we use the centre 50% of the bounding box.
# This avoids depth errors at the edges of detected objects.
DEPTH_BBOX_SCALE = 0.5


# =============================================================================
# 8. INTERACTABLE OBJECTS
# =============================================================================
# These are objects the user might want to physically reach and touch.
# When detected, the system switches to INTERACTION mode and guides
# the user's hand toward the object via haptic feedback.

INTERACTABLE_CLASSES = [
    "door handle",      # grab to open door
    "cup",              # pick up to drink
    "bottle",           # pick up to drink
    "bowl",             # pick up
    "cell phone",       # pick up
    "remote",           # pick up
    "keyboard",         # type
    "elevator button",  # press
    "light switch",     # press
    "door",             # push/pull
]

# When detected object is this close or closer, trigger hand guidance.
# 1500mm = 1.5 metres — close enough that reaching makes sense.
INTERACTION_TRIGGER_DIST_MM = 800


# =============================================================================
# 9. AUDIO SETTINGS
# =============================================================================

# Text-to-speech speaking rate. 150 words per minute is natural pace.
# Lower = slower and clearer. Higher = faster but harder to follow.
TTS_RATE = 150

# TTS volume (0.0 to 1.0).
TTS_VOLUME = 0.9

# Volume for spatial warning sounds (separate from speech).
ALERT_VOLUME = 0.8

# The horizontal field of view of the OAK-D Lite camera in degrees.
# Used to convert a pixel x-position into a real-world angle.
# This tells the audio engine how far left or right to pan the sound.
CAMERA_HFOV_DEG = 73.0

# Paths to alert sound files.
SOUND_DANGER_PATH  = BASE_DIR / "sounds" / "danger.wav"
SOUND_WARNING_PATH = BASE_DIR / "sounds" / "warning.wav"
SOUND_CHIME_PATH   = BASE_DIR / "sounds" / "chime.wav"


# The central "collision corridor" — objects within this angle AND within
# DANGER_DIST_MM are the highest priority alerts.
# Objects outside this angle are still warned about but at lower urgency.
# At face level, the user's body occupies about ±20 degrees of their path.
COLLISION_CORRIDOR_DEG = 20.0

# =============================================================================
# BANKNOTE DETECTION SETTINGS
# =============================================================================

# Path to the Egyptian banknote YOLOv8 model weights.
BANKNOTE_MODEL_PATH = BASE_DIR / "models" / "banknote_egp.pt"

# Minimum confidence to accept a banknote detection.
BANKNOTE_CONFIDENCE_THRESHOLD = 0.6

# How many consecutive frames the same denomination must appear before announcing.
# Prevents announcing from a single blurry frame.
BANKNOTE_STABILITY_FRAMES = 3

# Maximum distance in mm to attempt banknote recognition.
# User holds the note close to the glasses camera — within 50cm.
BANKNOTE_MAX_DIST_MM = 500
# =============================================================================
# 10. HAPTIC / BLE SETTINGS
# =============================================================================
# These are used by haptic_feedback.py to connect to the ESP32 wristband.

# The BLE service UUID of the ESP32 wristband.
# This is like the wristband's "address" that BLE uses to find it.
# You will replace this with the actual UUID from your ESP32 firmware.
BLE_SERVICE_UUID      = "0000ffe0-0000-1000-8000-00805f9b34fb"

# The BLE characteristic UUID — the specific "channel" we write haptic
# commands to on the wristband.
BLE_CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

# How long (in milliseconds) each vibration pulse lasts for each urgency level.
HAPTIC_DANGER_PATTERN  = [200, 100, 200, 100, 200]  # 3 short sharp bursts
HAPTIC_WARNING_PATTERN = [400, 200, 400]             # 2 medium pulses
HAPTIC_GUIDE_PATTERN   = [100, 50]                   # rapid tick for guidance

# Number of electrodes on the wristband array.
# The ESP32 controls these individually to encode direction.
HAPTIC_ELECTRODE_COUNT = 8


# =============================================================================
# 11. OCR SETTINGS
# =============================================================================

# Minimum height of a text region in pixels before we bother reading it.
# Tiny text is usually too blurry to read accurately anyway.
OCR_MIN_TEXT_HEIGHT_PX = 20

# OCR confidence threshold (0.0 to 1.0).
# Only speak text that OCR is at least this confident about.
OCR_CONFIDENCE_THRESHOLD = 0.7

# The language code for OCR. "en" = English.
# Can be extended to support Arabic, French, etc. in the future.
OCR_LANGUAGE = ["en", "ar"]

# Maximum number of characters to read aloud at once.
# Avoids reading an entire page of text in one go.
OCR_MAX_CHARS = 200

# Distance threshold — only trigger OCR if text is closer than this (mm).
# Text far away is too blurry to read accurately.
OCR_TRIGGER_DIST_MM = 2000


# =============================================================================
# 12. FACE RECOGNITION SETTINGS
# =============================================================================


# =============================================================================
# 13. BANKNOTE SETTINGS
# =============================================================================



# Currency locale — used to format the spoken output ("50 pounds" vs "50 dollars").
BANKNOTE_CURRENCY = "EGP"   # Egyptian Pound — change to your local currency


# =============================================================================
# 14. PERFORMANCE / TIMING SETTINGS
# =============================================================================
# Not every module needs to run on every single frame.
# Running everything every frame would cause lag.
# These values mean "run this module once every N frames".
# Example: OCR_RUN_EVERY = 10 means OCR runs on frame 1, 11, 21, 31...

# Obstacle detection is critical — run every frame.
OBSTACLE_RUN_EVERY = 1

# OCR is slow — run every 10 frames (~3 times per second at 30fps).
OCR_RUN_EVERY = 10

# Banknote recognition — run every 5 frames.
BANKNOTE_RUN_EVERY = 5

# Face recognition — run every 15 frames (once per half second).
FACE_RUN_EVERY = 15

# Interaction detection — run every 2 frames.
INTERACTION_RUN_EVERY = 2

# VLM scene understanding is the slowest — run every 30 frames (once per second).
VLM_RUN_EVERY = 30

# Minimum time (in seconds) between repeating the same alert.
# Prevents the system from saying "danger: chair" 30 times per second.
ALERT_COOLDOWN_SEC = 2.0

# Maximum time (in milliseconds) allowed for one full pipeline cycle.
# If a frame takes longer than this, we log a warning.
MAX_FRAME_TIME_MS = 50   # 50ms = 20fps minimum acceptable rate


# =============================================================================
# 15. LOGGING
# =============================================================================

# Where to save log files.
LOG_PATH = BASE_DIR / "logs" / "echora.log"

# Log level. Options: "DEBUG", "INFO", "WARNING", "ERROR".
# DEBUG shows everything (verbose). INFO shows normal operation messages.
# Use DEBUG while building, INFO when running normally.
LOG_LEVEL = "INFO"


MAX_MOTION_FOR_STILL_MODES = 0.8   # m/s² — above this = user is moving
IMU_MOTION_THRESHOLD       = 0.5   # m/s² — gentle motion detection threshold




# =============================================================================
# INTERACTION DETECTION SETTINGS
# =============================================================================

# Dominant hand to track — "Right" or "Left"
DOMINANT_HAND = "Right"

# Electrode grid dimensions
HAPTIC_ROWS = 5
HAPTIC_COLS = 6

# Phase transition thresholds (mm)
GUIDANCE_TO_EDGE_DIST_MM  = 150   # switch from guidance to edge rendering
EDGE_TO_SUCCESS_DIST_MM   = 30    # hand has touched the object

# Canny edge detection thresholds
CANNY_THRESHOLD_LOW  = 50
CANNY_THRESHOLD_HIGH = 150

# Minimum object size (pixels²) to consider as interactable
MIN_INTERACTABLE_AREA_PX = 1000
# =============================================================================
# 16. AUTO-CREATE FOLDERS
# =============================================================================
# When Python first imports this file, it automatically creates any folders
# that don't exist yet. This way you never get a "folder not found" error.

# os.makedirs creates a folder and all parent folders if they don't exist.
# exist_ok=True means: don't throw an error if the folder already exists.
os.makedirs(BASE_DIR / "models",   exist_ok=True)
os.makedirs(BASE_DIR / "sounds",   exist_ok=True)
os.makedirs(BASE_DIR / "database", exist_ok=True)
os.makedirs(BASE_DIR / "logs",     exist_ok=True)