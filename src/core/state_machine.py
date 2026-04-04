import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

from src.core.config import (
    MODE,
    DANGER_DIST_MM,
    WARNING_DIST_MM,
    OCR_TRIGGER_DIST_MM,
    INTERACTION_TRIGGER_DIST_MM,
    FACE_CONFIDENCE_THRESHOLD,
    MAX_MOTION_FOR_STILL_MODES,
    IMU_MOTION_THRESHOLD,
)
from src.core.utils import logger, get_timestamp_ms

NAVIGATION_MIN_DWELL_SEC = 0.5
OCR_MIN_DWELL_SEC = 2.0
INTERACTION_MIN_DWELL_SEC = 3.0
FACE_ID_MIN_DWELL_SEC = 1.5
BANKNOTE_MIN_DWELL_SEC = 1.5

MODE_DWELL_TIMES = {
    MODE.NAVIGATION:  NAVIGATION_MIN_DWELL_SEC,
    MODE.OCR:         OCR_MIN_DWELL_SEC,
    MODE.INTERACTION: INTERACTION_MIN_DWELL_SEC,
    MODE.FACE_ID:     FACE_ID_MIN_DWELL_SEC,
    MODE.BANKNOTE:    BANKNOTE_MIN_DWELL_SEC,
}

FACE_ID_MIN_CONSECUTIVE_FRAMES = 3
BANKNOTE_MIN_CONSECUTIVE_FRAMES = 3

@dataclass
class ModeTransition:
    from_mode: str
    to_mode: str
    timestamp_ms: float
    reason: str

class StateMachine:
    def __init__(self):
        self._current_mode: str = MODE.NAVIGATION
        self._mode_entered_at: float = time.time()

        self._face_frames: int = 0
        self._banknote_frames: int = 0
        self._ocr_frames: int = 0
        self._interaction_frames: int = 0

        self._motion_level: float = 0.0
        self._ocr_trigger_distance: float = 0.0
        self._interaction_distance: float = 0.0
        self._face_confidence: float = 0.0

        self._callbacks: Dict[str, Dict[str, List[Callable]]] = {}
        for mode_name in [MODE.NAVIGATION, MODE.OCR, MODE.INTERACTION, MODE.FACE_ID, MODE.BANKNOTE]:
            self._callbacks[mode_name] = {"on_enter": [], "on_exit": []}

        self._history: List[ModeTransition] = []
        self._max_history: int = 50
        self._frame_count: int = 0

        logger.info(f"StateMachine initialised. Starting in {self._current_mode} mode.")

    def update(self, bundle: Dict, obstacle_result: Dict, ocr_text_distance: float = 0.0,
               face_confidence: float = 0.0, interactable_distance: float = 0.0,
               banknote_visible: bool = False) -> str:
        self._frame_count += 1

        self._ocr_trigger_distance = ocr_text_distance
        self._face_confidence = face_confidence
        self._interaction_distance = interactable_distance

        self._motion_level = self._get_imu_motion_level(bundle)

        if 0 < ocr_text_distance < OCR_TRIGGER_DIST_MM and self._motion_level < MAX_MOTION_FOR_STILL_MODES:
            self._ocr_frames += 1
        else:
            self._ocr_frames = 0

        if 0 < interactable_distance < INTERACTION_TRIGGER_DIST_MM:
            self._interaction_frames += 1
        else:
            self._interaction_frames = 0

        if face_confidence >= FACE_CONFIDENCE_THRESHOLD:
            self._face_frames += 1
        else:
            self._face_frames = 0

        if banknote_visible and self._motion_level < MAX_MOTION_FOR_STILL_MODES:
            self._banknote_frames += 1
        else:
            self._banknote_frames = 0

        new_mode = self._check_transitions(obstacle_result)
        if new_mode != self._current_mode:
            self.switch_to(new_mode, reason=f"transition from {self._current_mode}")

        return self._current_mode

    def _check_transitions(self, obstacle_result: Dict) -> str:
        if self._emergency_override(obstacle_result):
            if self._current_mode != MODE.NAVIGATION:
                logger.warning(f"EMERGENCY OVERRIDE: DANGER obstacle detected. Forcing NAVIGATION from {self._current_mode}.")
            return MODE.NAVIGATION

        if self.is_stable():
            if self._current_mode == MODE.OCR and self._ocr_frames == 0:
                logger.info("OCR trigger gone. Returning to NAVIGATION.")
                return MODE.NAVIGATION

            if self._current_mode == MODE.INTERACTION and self._interaction_frames == 0:
                logger.info("Interaction trigger gone. Returning to NAVIGATION.")
                return MODE.NAVIGATION

            if self._current_mode == MODE.FACE_ID and self._face_frames == 0:
                logger.info("Face lost. Returning to NAVIGATION.")
                return MODE.NAVIGATION

            if self._current_mode == MODE.BANKNOTE and self._banknote_frames == 0:
                logger.info("Banknote gone. Returning to NAVIGATION.")
                return MODE.NAVIGATION

        if self._current_mode == MODE.NAVIGATION and self.is_stable():
            if self._should_enter_banknote():
                return MODE.BANKNOTE
            if self._should_enter_face_id():
                return MODE.FACE_ID
            if self._should_enter_interaction(obstacle_result):
                return MODE.INTERACTION
            if self._should_enter_ocr():
                return MODE.OCR

        return self._current_mode

    def _should_enter_ocr(self) -> bool:
        if self._ocr_frames < 2 or self._ocr_trigger_distance <= 0 or self._motion_level > MAX_MOTION_FOR_STILL_MODES:
            return False
        logger.debug(f"OCR entry condition met: text at {self._ocr_trigger_distance:.0f}mm, motion={self._motion_level:.2f}m/s²")
        return True

    def _should_enter_interaction(self, obstacle_result: Dict) -> bool:
        if self._interaction_frames < 2 or self._interaction_distance <= 0:
            return False
        if obstacle_result.get("danger", []):
            logger.debug("Interaction suppressed: DANGER obstacle present.")
            return False
        logger.debug(f"Interaction entry condition met: object at {self._interaction_distance:.0f}mm")
        return True

    def _should_enter_face_id(self) -> bool:
        if self._face_frames < FACE_ID_MIN_CONSECUTIVE_FRAMES or self._face_confidence < FACE_CONFIDENCE_THRESHOLD:
            return False
        logger.debug(f"Face ID entry condition met: confidence={self._face_confidence:.2f}, frames={self._face_frames}")
        return True

    def _should_enter_banknote(self) -> bool:
        if self._banknote_frames < BANKNOTE_MIN_CONSECUTIVE_FRAMES or self._motion_level > MAX_MOTION_FOR_STILL_MODES:
            return False
        logger.debug(f"Banknote entry condition met: frames={self._banknote_frames}, motion={self._motion_level:.2f}m/s²")
        return True

    def _emergency_override(self, obstacle_result: Dict) -> bool:
        if self._current_mode == MODE.NAVIGATION:
            return False

        danger_tracks = obstacle_result.get("danger", [])
        if not danger_tracks:
            return False

        if self._current_mode == MODE.FACE_ID:
            non_person_dangers = [t for t in danger_tracks if t.get("label", "") != "person"]
            if not non_person_dangers:
                return False
            logger.warning(f"Emergency override in FACE_ID: {non_person_dangers[0]['label']} is DANGER.")
            return True

        logger.warning(f"Emergency override: {danger_tracks[0]['label']} is in DANGER zone.")
        return True

    def _get_imu_motion_level(self, bundle: Dict) -> float:
        imu = bundle.get("imu", {})
        accel = imu.get("accel", {"x": 0.0, "y": 0.0, "z": 9.81})
        magnitude = (float(accel.get("x", 0.0))**2 + float(accel.get("y", 0.0))**2 + float(accel.get("z", 9.81))**2) ** 0.5
        return abs(magnitude - 9.81)

    def switch_to(self, new_mode: str, reason: str = ""):
        old_mode = self._current_mode

        for callback in self._callbacks.get(old_mode, {}).get("on_exit", []):
            try:
                callback()
            except Exception as e:
                logger.error(f"on_exit callback error for {old_mode}: {e}")

        logger.info(f"Mode switch: {old_mode} -> {new_mode}" + (f" ({reason})" if reason else ""))
        
        self._current_mode = new_mode
        self._mode_entered_at = time.time()

        self._history.append(ModeTransition(
            from_mode=old_mode, to_mode=new_mode, timestamp_ms=get_timestamp_ms(), reason=reason
        ))

        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        for callback in self._callbacks.get(new_mode, {}).get("on_enter", []):
            try:
                callback()
            except Exception as e:
                logger.error(f"on_enter callback error for {new_mode}: {e}")

    def register_callback(self, mode: str, on_enter: Optional[Callable] = None, on_exit: Optional[Callable] = None):
        if mode not in self._callbacks:
            self._callbacks[mode] = {"on_enter": [], "on_exit": []}
        if on_enter:
            self._callbacks[mode]["on_enter"].append(on_enter)
            logger.debug(f"Registered on_enter callback for {mode}: {on_enter.__name__}")
        if on_exit:
            self._callbacks[mode]["on_exit"].append(on_exit)
            logger.debug(f"Registered on_exit callback for {mode}: {on_exit.__name__}")

    def get_mode(self) -> str:
        return self._current_mode

    def get_mode_duration(self) -> float:
        return time.time() - self._mode_entered_at

    def is_stable(self) -> bool:
        min_dwell = MODE_DWELL_TIMES.get(self._current_mode, 1.0)
        return self.get_mode_duration() >= min_dwell

    def is_in_mode(self, mode: str) -> bool:
        return self._current_mode == mode

    def get_history(self, last_n: int = 10) -> List[ModeTransition]:
        return self._history[-last_n:]

    def get_stats(self) -> Dict:
        return {
            "current_mode":    self._current_mode,
            "mode_duration_s": round(self.get_mode_duration(), 2),
            "is_stable":       self.is_stable(),
            "motion_level":    round(self._motion_level, 3),
            "frame_count":     self._frame_count,
            "ocr_frames":      self._ocr_frames,
            "face_frames":     self._face_frames,
            "interaction_frames": self._interaction_frames,
            "banknote_frames": self._banknote_frames,
            "total_switches":  len(self._history),
        }

    def force_mode(self, mode: str, reason: str = "forced"):
        logger.info(f"Force mode: {mode} ({reason})")
        self.switch_to(mode, reason=reason)