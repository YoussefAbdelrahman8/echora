import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

FACE_DB_PATH = BASE_DIR / "database" / "faces"
FACE_CONFIDENCE_THRESHOLD = 0.15
FACE_STABILITY_FRAMES = 3
FACE_RECOGNITION_TOLERANCE = 0.6

CAMERA_RGB_WIDTH  = 1280
CAMERA_RGB_HEIGHT = 800
CAMERA_DEPTH_WIDTH  = 1280
CAMERA_DEPTH_HEIGHT = 800
CAMERA_FPS = 30
DEPTH_MAX_MM = 8000
DEPTH_MIN_MM = 100

DANGER_DIST_MM = 1500
WARNING_DIST_MM = 3000

KALMAN_PROCESS_NOISE = 0.03
KALMAN_MEASUREMENT_NOISE = 0.1
KALMAN_MAX_MISSED_FRAMES = 10

class MODE:
    NAVIGATION = "NAVIGATION"
    OCR = "OCR"
    INTERACTION = "INTERACTION"
    FACE_ID = "FACE_ID"
    BANKNOTE = "BANKNOTE"

DETECTION_CONFIDENCE_THRESHOLD = 0.5
YOLO_MODEL_PATH = BASE_DIR / "models" / "yolov8s.pt"
YOLO_INPUT_WIDTH  = 416
YOLO_INPUT_HEIGHT = 256

RELEVANT_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "chair", "couch",
    "dining table", "bed", "toilet", "sink", "door", "stairs", "potted plant",
    "dog", "cat", "bottle", "cup", "bowl", "laptop", "tv", "book", "clock",
    "cell phone", "backpack", "umbrella", "bench", "fire hydrant", "stop sign", "traffic light",
]

DEPTH_BBOX_SCALE = 0.5

INTERACTABLE_CLASSES = [
    "door handle", "cup", "bottle", "bowl", "cell phone", "remote",
    "keyboard", "elevator button", "light switch", "door",
]
INTERACTION_TRIGGER_DIST_MM = 800

TTS_RATE = 150
TTS_VOLUME = 0.9
ALERT_VOLUME = 0.8
CAMERA_HFOV_DEG = 73.0

SOUND_DANGER_PATH  = BASE_DIR / "sounds" / "danger.wav"
SOUND_WARNING_PATH = BASE_DIR / "sounds" / "warning.wav"
SOUND_CHIME_PATH   = BASE_DIR / "sounds" / "chime.wav"

COLLISION_CORRIDOR_DEG = 20.0

BANKNOTE_MODEL_PATH = BASE_DIR / "models" / "banknote_egp.pt"
BANKNOTE_CONFIDENCE_THRESHOLD = 0.6
BANKNOTE_STABILITY_FRAMES = 3
BANKNOTE_MAX_DIST_MM = 500
BANKNOTE_CURRENCY = "EGP"

BLE_SERVICE_UUID      = "0000ffe0-0000-1000-8000-00805f9b34fb"
BLE_CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

HAPTIC_DANGER_PATTERN  = [200, 100, 200, 100, 200]
HAPTIC_WARNING_PATTERN = [400, 200, 400]
HAPTIC_GUIDE_PATTERN   = [100, 50]

HAPTIC_ELECTRODE_COUNT = 8

OCR_MIN_TEXT_HEIGHT_PX = 20
OCR_CONFIDENCE_THRESHOLD = 0.7
OCR_LANGUAGE = ["en", "ar"]
OCR_MAX_CHARS = 200
OCR_TRIGGER_DIST_MM = 2000

OBSTACLE_RUN_EVERY = 1
OCR_RUN_EVERY = 10
BANKNOTE_RUN_EVERY = 5
FACE_RUN_EVERY = 15
INTERACTION_RUN_EVERY = 2
VLM_RUN_EVERY = 30

ALERT_COOLDOWN_SEC = 2.0
MAX_FRAME_TIME_MS = 50

LOG_PATH = BASE_DIR / "logs" / "echora.log"
LOG_LEVEL = "INFO"

MAX_MOTION_FOR_STILL_MODES = 0.8
IMU_MOTION_THRESHOLD       = 0.5

DOMINANT_HAND = "Right"
HAPTIC_ROWS = 5
HAPTIC_COLS = 6

GUIDANCE_TO_EDGE_DIST_MM  = 150
EDGE_TO_SUCCESS_DIST_MM   = 30

CANNY_THRESHOLD_LOW  = 50
CANNY_THRESHOLD_HIGH = 150
MIN_INTERACTABLE_AREA_PX = 1000

os.makedirs(BASE_DIR / "models",   exist_ok=True)
os.makedirs(BASE_DIR / "sounds",   exist_ok=True)
os.makedirs(BASE_DIR / "database", exist_ok=True)
os.makedirs(BASE_DIR / "logs",     exist_ok=True)