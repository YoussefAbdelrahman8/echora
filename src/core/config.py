import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List

class EchoraConfig(BaseSettings):
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    ASSETS_DIR: Path = BASE_DIR / "assets"
    
    FACE_DB_PATH: Path = ASSETS_DIR / "database" / "faces"
    FACE_CONFIDENCE_THRESHOLD: float = 0.15
    FACE_STABILITY_FRAMES: int = 3
    FACE_RECOGNITION_TOLERANCE: float = 0.6
    
    CAMERA_RGB_WIDTH: int = 1280
    CAMERA_RGB_HEIGHT: int = 800
    CAMERA_DEPTH_WIDTH: int = 1280
    CAMERA_DEPTH_HEIGHT: int = 800
    CAMERA_FPS: int = 30
    DEPTH_MAX_MM: int = 8000
    DEPTH_MIN_MM: int = 100
    
    DANGER_DIST_MM: int = 1500
    WARNING_DIST_MM: int = 3000
    
    KALMAN_PROCESS_NOISE: float = 0.03
    KALMAN_MEASUREMENT_NOISE: float = 0.1
    KALMAN_MAX_MISSED_FRAMES: int = 10
    
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.5
    YOLO_MODEL_PATH: Path = ASSETS_DIR / "models" / "yolov8s.pt"
    YOLO_INPUT_WIDTH: int = 416
    YOLO_INPUT_HEIGHT: int = 256
    
    RELEVANT_CLASSES: List[str] = [
        "person", "bicycle", "car", "motorcycle", "bus", "truck", "chair", "couch",
        "dining table", "bed", "toilet", "sink", "door", "stairs", "potted plant",
        "dog", "cat", "bottle", "cup", "bowl", "laptop", "tv", "book", "clock",
        "cell phone", "backpack", "umbrella", "bench", "fire hydrant", "stop sign", "traffic light",
    ]
    DEPTH_BBOX_SCALE: float = 0.5
    
    INTERACTABLE_CLASSES: List[str] = [
        "door handle", "cup", "bottle", "bowl", "cell phone", "remote",
        "keyboard", "elevator button", "light switch", "door",
    ]
    INTERACTION_TRIGGER_DIST_MM: int = 800
    
    TTS_RATE: int = 150
    TTS_VOLUME: float = 0.9
    ALERT_VOLUME: float = 0.8
    CAMERA_HFOV_DEG: float = 73.0
    
    SOUND_DANGER_PATH: Path  = ASSETS_DIR / "sounds" / "danger.wav"
    SOUND_WARNING_PATH: Path = ASSETS_DIR / "sounds" / "warning.wav"
    SOUND_CHIME_PATH: Path   = ASSETS_DIR / "sounds" / "chime.wav"
    
    COLLISION_CORRIDOR_DEG: float = 20.0
    
    BANKNOTE_MODEL_PATH: Path = ASSETS_DIR / "models" / "banknote_egp.pt"
    BANKNOTE_CONFIDENCE_THRESHOLD: float = 0.6
    BANKNOTE_STABILITY_FRAMES: int = 3
    BANKNOTE_MAX_DIST_MM: int = 500
    BANKNOTE_CURRENCY: str = "EGP"
    
    BLE_SERVICE_UUID: str = "0000ffe0-0000-1000-8000-00805f9b34fb"
    BLE_CHARACTERISTIC_UUID: str = "0000ffe1-0000-1000-8000-00805f9b34fb"
    
    HAPTIC_DANGER_PATTERN:  List[int] = [200, 100, 200, 100, 200]
    HAPTIC_WARNING_PATTERN: List[int] = [400, 200, 400]
    HAPTIC_GUIDE_PATTERN:   List[int] = [100, 50]
    HAPTIC_ELECTRODE_COUNT: int = 8
    
    OCR_MIN_TEXT_HEIGHT_PX: int = 20
    OCR_CONFIDENCE_THRESHOLD: float = 0.7
    OCR_LANGUAGE: List[str] = ["en", "ar"]
    OCR_MAX_CHARS: int = 200
    OCR_TRIGGER_DIST_MM: int = 2000
    
    OBSTACLE_RUN_EVERY: int = 1
    OCR_RUN_EVERY: int = 10
    BANKNOTE_RUN_EVERY: int = 5
    FACE_RUN_EVERY: int = 15
    INTERACTION_RUN_EVERY: int = 2
    VLM_RUN_EVERY: int = 30
    
    ALERT_COOLDOWN_SEC: float = 2.0
    MAX_FRAME_TIME_MS: int = 50
    
    LOG_PATH: Path = ASSETS_DIR / "logs" / "echora.log"
    LOG_LEVEL: str = "INFO"
    
    MAX_MOTION_FOR_STILL_MODES: float = 0.8
    IMU_MOTION_THRESHOLD: float = 0.5
    
    DOMINANT_HAND: str = "Right"
    HAPTIC_ROWS: int = 5
    HAPTIC_COLS: int = 6
    
    GUIDANCE_TO_EDGE_DIST_MM: int = 150
    EDGE_TO_SUCCESS_DIST_MM: int = 30
    
    CANNY_THRESHOLD_LOW: int  = 50
    CANNY_THRESHOLD_HIGH: int = 150
    MIN_INTERACTABLE_AREA_PX: int = 1000

    NAVIGATION_MIN_DWELL_SEC: float = 0.5
    OCR_MIN_DWELL_SEC: float = 2.0
    INTERACTION_MIN_DWELL_SEC: float = 3.0
    FACE_ID_MIN_DWELL_SEC: float = 1.5
    BANKNOTE_MIN_DWELL_SEC: float = 1.5

    FACE_ID_MIN_CONSECUTIVE_FRAMES: int = 3
    BANKNOTE_MIN_CONSECUTIVE_FRAMES: int = 3

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

class MODE:
    NAVIGATION = "NAVIGATION"
    OCR = "OCR"
    INTERACTION = "INTERACTION"
    FACE_ID = "FACE_ID"
    BANKNOTE = "BANKNOTE"

settings = EchoraConfig()

os.makedirs(settings.ASSETS_DIR / "models",   exist_ok=True)
os.makedirs(settings.ASSETS_DIR / "sounds",   exist_ok=True)
os.makedirs(settings.ASSETS_DIR / "database", exist_ok=True)
os.makedirs(settings.ASSETS_DIR / "logs",     exist_ok=True)