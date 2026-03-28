# =============================================================================
# state_machine.py — ECHORA Autonomous Mode Switching
# =============================================================================
# Always in exactly one mode at a time.
# Watches sensors and switches modes automatically based on rules.
# Only one module is fully active per mode — keeps the system fast and focused.
#
# Modes:      NAVIGATION → OCR → INTERACTION → FACE_ID → BANKNOTE
# Priority:   DANGER override always wins, regardless of current mode.
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

# time for measuring how long we've been in each mode (dwell time).
import time

# dataclass decorator — generates __init__ automatically for simple classes.
from dataclasses import dataclass, field

# Type hints for cleaner, more readable code.
from typing import Dict, List, Optional, Callable

# Our constants and helpers.
from config import (
    MODE,
    DANGER_DIST_MM,
    WARNING_DIST_MM,
    OCR_TRIGGER_DIST_MM,
    INTERACTION_TRIGGER_DIST_MM,
    FACE_CONFIDENCE_THRESHOLD,
    MAX_MOTION_FOR_STILL_MODES,   # ← add this
    IMU_MOTION_THRESHOLD,
)
from utils import logger, get_timestamp_ms


# =============================================================================
# DWELL TIME CONFIGURATION
# =============================================================================
# Minimum seconds a mode must stay active before it can switch again.
# Prevents flickering between modes when a trigger condition is unstable.

# How long to stay in NAVIGATION before any other mode can activate.
# 1.0 second prevents immediately switching when a face or text flashes briefly.
NAVIGATION_MIN_DWELL_SEC = 0.5

# How long to stay in OCR mode minimum.
# 3 seconds gives the user time to hear the text being read.
OCR_MIN_DWELL_SEC = 2.0

# How long to stay in INTERACTION mode minimum.
# 5 seconds gives the user time to reach the object.
INTERACTION_MIN_DWELL_SEC = 3.0

# How long to stay in FACE_ID mode minimum.
# 2 seconds for the recognition to complete and speak the name.
FACE_ID_MIN_DWELL_SEC = 1.5

# How long to stay in BANKNOTE mode minimum.
# 2 seconds to classify and speak the denomination.
BANKNOTE_MIN_DWELL_SEC = 1.5

# Map of mode → minimum dwell time.
# This lets us look up the dwell time for any mode in one line.
MODE_DWELL_TIMES = {
    MODE.NAVIGATION:  NAVIGATION_MIN_DWELL_SEC,
    MODE.OCR:         OCR_MIN_DWELL_SEC,
    MODE.INTERACTION: INTERACTION_MIN_DWELL_SEC,
    MODE.FACE_ID:     FACE_ID_MIN_DWELL_SEC,
    MODE.BANKNOTE:    BANKNOTE_MIN_DWELL_SEC,
}

# How many consecutive frames a face must be visible before triggering FACE_ID.
# Prevents switching to FACE_ID for a face that appears for just one frame.
FACE_ID_MIN_CONSECUTIVE_FRAMES = 3

# How many consecutive frames a banknote must be visible before triggering BANKNOTE.
BANKNOTE_MIN_CONSECUTIVE_FRAMES = 3

# IMU motion threshold — how much acceleration (m/s²) counts as "moving".
# Above this value, the user is considered to be walking/moving.
# Below it, the user is considered relatively still.
# 1.5 m/s² is a gentle threshold — normal walking creates about 2-4 m/s².


# Maximum motion level allowed for OCR and BANKNOTE modes.
# If the user is moving faster than this, we don't switch to these modes.
# A moving user can't hold the camera steady enough to read text or scan notes.



# =============================================================================
# MODE TRANSITION RECORD
# =============================================================================

@dataclass
class ModeTransition:
    """
    Records one mode switch event for the history log.

    @dataclass automatically generates:
      __init__(self, from_mode, to_mode, timestamp_ms, reason)
      __repr__  (clean string representation for printing)

    Stored in StateMachine._history so we can review what happened.
    """

    # The mode we were in before the switch.
    from_mode: str

    # The mode we switched to.
    to_mode: str

    # When the switch happened — milliseconds since epoch.
    timestamp_ms: float

    # Why the switch happened — a human-readable explanation.
    # This is extremely useful for debugging unexpected mode switches.
    reason: str


# =============================================================================
# STATE MACHINE CLASS
# =============================================================================

class StateMachine:
    """
    Controls which ECHORA module is active at any given moment.

    Always in exactly one of: NAVIGATION, OCR, INTERACTION, FACE_ID, BANKNOTE.

    Every frame, call update() with the latest sensor data.
    It evaluates switching rules and returns the current mode.

    Other modules register callbacks to be notified when modes change.

    Usage:
        sm = StateMachine()

        while True:
            bundle = cam.get_synced_bundle()
            result = detector.update(bundle)
            mode   = sm.update(bundle, result)

            if mode == MODE.NAVIGATION:
                # run navigation logic
            elif mode == MODE.OCR:
                # run OCR logic
    """

    def __init__(self):
        """
        Creates the state machine.

        Starts in NAVIGATION mode — the safe default.
        All timers and counters start at zero.
        """

        # ── Current mode ──────────────────────────────────────────────────────
        # The mode we are currently in. Always exactly one of the MODE constants.
        self._current_mode: str = MODE.NAVIGATION

        # ── Timing ────────────────────────────────────────────────────────────
        # time.time() returns the current Unix timestamp as a float (seconds).
        # We record when we entered the current mode so we can calculate
        # how long we've been in it.
        self._mode_entered_at: float = time.time()

        # ── Consecutive frame counters ─────────────────────────────────────────
        # These count how many frames in a row a trigger condition has been true.
        # Used to prevent false triggers from single-frame detections.

        # How many consecutive frames a face has been visible.
        self._face_frames: int = 0

        # How many consecutive frames a banknote has been visible.
        self._banknote_frames: int = 0

        # How many consecutive frames OCR-worthy text has been visible.
        self._ocr_frames: int = 0

        # How many consecutive frames an interactable object has been visible.
        self._interaction_frames: int = 0

        # ── Latest sensor readings ─────────────────────────────────────────────
        # Cached values from the most recent update() call.
        # Used internally by the transition checking functions.

        # The latest IMU motion level (0.0 to ~10.0 m/s²).
        self._motion_level: float = 0.0

        # The latest OCR trigger distance (mm). 0 = no text detected.
        self._ocr_trigger_distance: float = 0.0

        # The latest interactable object distance (mm). 0 = none detected.
        self._interaction_distance: float = 0.0

        # The latest face detection confidence (0.0 to 1.0). 0 = no face.
        self._face_confidence: float = 0.0

        # ── Callbacks ─────────────────────────────────────────────────────────
        # Dictionary of callbacks for each mode.
        # Structure:
        #   {
        #     "NAVIGATION": {"on_enter": [func1, func2], "on_exit": [func3]},
        #     "OCR":        {"on_enter": [func4],        "on_exit": []},
        #     ...
        #   }
        # Each mode can have multiple on_enter and on_exit callbacks.
        # They are all called when the mode is entered or exited.
        self._callbacks: Dict[str, Dict[str, List[Callable]]] = {}

        # Initialise an empty callback dict for every mode.
        # This prevents KeyError when we try to call callbacks for a mode
        # that hasn't had any callbacks registered yet.
        for mode_name in [
            MODE.NAVIGATION, MODE.OCR,
            MODE.INTERACTION, MODE.FACE_ID, MODE.BANKNOTE
        ]:
            # Each mode starts with empty lists for both event types.
            self._callbacks[mode_name] = {
                "on_enter": [],   # called when we ENTER this mode
                "on_exit":  [],   # called when we EXIT this mode
            }

        # ── History log ───────────────────────────────────────────────────────
        # A list of ModeTransition objects recording every mode switch.
        # We keep the last 50 transitions for debugging.
        # 'field(default_factory=list)' is the dataclass way of saying
        # "start with an empty list" — we just use a plain list here.
        self._history: List[ModeTransition] = []

        # Maximum number of history entries to keep.
        # 50 entries is enough to see recent patterns without using much memory.
        self._max_history: int = 50

        # ── Frame counter ──────────────────────────────────────────────────────
        # Total number of frames processed since startup.
        self._frame_count: int = 0

        logger.info(f"StateMachine initialised. Starting in {self._current_mode} mode.")


    # =========================================================================
    # MAIN UPDATE FUNCTION
    # =========================================================================

    def update(
        self,
        bundle: Dict,
        obstacle_result: Dict,
        ocr_text_distance: float   = 0.0,
        face_confidence: float     = 0.0,
        interactable_distance: float = 0.0,
        banknote_visible: bool     = False,
    ) -> str:
        """
        Main function — call every frame from control_unit.py.

        Evaluates all mode switching rules and switches if needed.
        Returns the current mode string after any switching.

        Arguments:
            bundle:                camera bundle from camera.py (for IMU)
            obstacle_result:       result dict from obstacle_detection.py
            ocr_text_distance:     distance to nearest readable text in mm
                                   (0 = no text detected)
            face_confidence:       confidence of face detection (0 = no face)
            interactable_distance: distance to nearest interactable object in mm
                                   (0 = none detected)
            banknote_visible:      True if a banknote is detected in frame

        Returns:
            Current mode string, e.g. MODE.NAVIGATION or MODE.OCR
        """

        # Increment frame counter.
        self._frame_count += 1

        # ── Update cached sensor readings ──────────────────────────────────────
        # These are used by the private _should_enter_X() functions below.
        self._ocr_trigger_distance    = ocr_text_distance
        self._face_confidence         = face_confidence
        self._interaction_distance    = interactable_distance

        # ── Calculate IMU motion level ─────────────────────────────────────────
        # This tells us if the user is walking or standing still.
        self._motion_level = self._get_imu_motion_level(bundle)

        # ── Update consecutive frame counters ──────────────────────────────────
        # These count how many frames in a row each trigger has been active.
        # If the trigger is not active this frame, reset the counter to 0.

        # OCR trigger: text must be close enough AND user must be relatively still.
        if (ocr_text_distance > 0
                and ocr_text_distance < OCR_TRIGGER_DIST_MM
                and self._motion_level < MAX_MOTION_FOR_STILL_MODES):
            self._ocr_frames += 1  # trigger active — increment counter
        else:
            self._ocr_frames = 0  # trigger not active — reset to 0

        # Interaction trigger: interactable object must be close enough.
        if (interactable_distance > 0
                and interactable_distance < INTERACTION_TRIGGER_DIST_MM):
            self._interaction_frames += 1
        else:
            self._interaction_frames = 0

        # Face ID trigger: face must have sufficient confidence.
        if face_confidence >= FACE_CONFIDENCE_THRESHOLD:
            self._face_frames += 1
        else:
            self._face_frames = 0

        # Banknote trigger: banknote must be visible AND user must be still.
        if (banknote_visible
                and self._motion_level < MAX_MOTION_FOR_STILL_MODES):
            self._banknote_frames += 1
        else:
            self._banknote_frames = 0

        # ── Check for mode transitions ─────────────────────────────────────────
        new_mode = self._check_transitions(obstacle_result)

        # ── Perform the switch if needed ───────────────────────────────────────
        # Only switch if the new mode is different from the current mode.
        if new_mode != self._current_mode:
            self.switch_to(new_mode, reason=f"transition from {self._current_mode}")

        return self._current_mode


    # =========================================================================
    # TRANSITION LOGIC
    # =========================================================================

    def _check_transitions(self, obstacle_result: Dict) -> str:
        """
        Evaluates all mode switching rules in priority order.

        Priority (highest to lowest):
          1. Emergency override — DANGER obstacle detected → NAVIGATION
          2. BANKNOTE trigger
          3. FACE_ID trigger
          4. INTERACTION trigger
          5. OCR trigger
          6. Stay in current mode (no change)

        The emergency override has highest priority because user safety
        always trumps everything else. If something is about to hit the
        user, we drop whatever we're doing and warn them immediately.

        Returns the mode we should be in (may be same as current).
        """

        # ── Priority 1: Emergency override ────────────────────────────────────
        # If a DANGER obstacle is detected, always return to NAVIGATION.
        # This overrides ALL other modes — even if we're mid-OCR or mid-face-scan.
        if self._emergency_override(obstacle_result):
            if self._current_mode != MODE.NAVIGATION:
                logger.warning(
                    f"EMERGENCY OVERRIDE: DANGER obstacle detected. "
                    f"Forcing NAVIGATION from {self._current_mode}."
                )
            return MODE.NAVIGATION

        # ── Check if current mode should be exited ─────────────────────────────
        # Before checking if we should enter a new mode, check if we should
        # EXIT the current mode (because its trigger condition is gone).

        # is_stable() returns True if the dwell time has passed.
        # We only allow mode exits after the minimum dwell time.
        if self.is_stable():

            # If we're in OCR but no text is visible anymore → back to NAV.
            if (self._current_mode == MODE.OCR
                    and self._ocr_frames == 0):
                logger.info("OCR trigger gone. Returning to NAVIGATION.")
                return MODE.NAVIGATION

            # If we're in INTERACTION but no interactable object nearby → back to NAV.
            if (self._current_mode == MODE.INTERACTION
                    and self._interaction_frames == 0):
                logger.info("Interaction trigger gone. Returning to NAVIGATION.")
                return MODE.NAVIGATION

            # If we're in FACE_ID but no face visible → back to NAV.
            if (self._current_mode == MODE.FACE_ID
                    and self._face_frames == 0):
                logger.info("Face lost. Returning to NAVIGATION.")
                return MODE.NAVIGATION

            # If we're in BANKNOTE but no banknote visible → back to NAV.
            if (self._current_mode == MODE.BANKNOTE
                    and self._banknote_frames == 0):
                logger.info("Banknote gone. Returning to NAVIGATION.")
                return MODE.NAVIGATION

        # ── Check if we should enter a new mode ────────────────────────────────
        # We only enter new modes from NAVIGATION.
        # You cannot switch directly from OCR to FACE_ID, for example.
        # You must return to NAVIGATION first, then enter the new mode.
        # This keeps mode transitions predictable and safe.

        if self._current_mode == MODE.NAVIGATION and self.is_stable():

            # Priority 2: BANKNOTE — check before face because a banknote
            # held up close is very intentional and specific.
            if self._should_enter_banknote():
                return MODE.BANKNOTE

            # Priority 3: FACE_ID
            if self._should_enter_face_id():
                return MODE.FACE_ID

            # Priority 4: INTERACTION — before OCR because touching an object
            # is more immediate than reading text.
            if self._should_enter_interaction(obstacle_result):
                return MODE.INTERACTION

            # Priority 5: OCR — lowest priority of the non-emergency modes.
            if self._should_enter_ocr():
                return MODE.OCR

        # ── No transition needed ───────────────────────────────────────────────
        # Return current mode unchanged.
        return self._current_mode


    # =========================================================================
    # MODE ENTRY CONDITIONS
    # =========================================================================

    def _should_enter_ocr(self) -> bool:
        """
        Returns True if conditions are met to enter OCR mode.

        Conditions:
          1. Text has been detected for at least 2 consecutive frames
             (prevents single-frame false triggers)
          2. Text is closer than OCR_TRIGGER_DIST_MM (from config.py)
          3. User is relatively still (motion below MAX_MOTION_FOR_STILL_MODES)
             (moving user can't hold camera steady enough for reliable OCR)
        """

        # _ocr_frames counts consecutive frames where OCR trigger was active.
        # Require at least 2 consecutive frames to confirm text is really there.
        if self._ocr_frames < 2:
            return False

        # The distance check is already encoded in _ocr_frames (we only
        # increment it when distance < OCR_TRIGGER_DIST_MM), but we check
        # again explicitly for clarity and safety.
        if self._ocr_trigger_distance <= 0:
            return False

        # Motion check — already encoded in _ocr_frames too, but explicit here.
        if self._motion_level > MAX_MOTION_FOR_STILL_MODES:
            return False

        logger.debug(
            f"OCR entry condition met: text at {self._ocr_trigger_distance:.0f}mm, "
            f"motion={self._motion_level:.2f}m/s²"
        )
        return True


    def _should_enter_interaction(self, obstacle_result: Dict) -> bool:
        """
        Returns True if conditions are met to enter INTERACTION mode.

        Conditions:
          1. An interactable object has been detected for at least 2 frames
          2. The object is closer than INTERACTION_TRIGGER_DIST_MM
          3. No DANGER obstacle is currently present
             (safety first — don't interact with objects while in danger)
        """

        if self._interaction_frames < 2:
            return False

        if self._interaction_distance <= 0:
            return False

        # Safety check — don't switch to interaction if something is dangerous.
        # obstacle_result["danger"] is the list of DANGER-level obstacles.
        # If the list is not empty, there is a danger — don't switch.
        if obstacle_result.get("danger", []):
            logger.debug(
                "Interaction suppressed: DANGER obstacle present. "
                "Safety takes priority."
            )
            return False

        logger.debug(
            f"Interaction entry condition met: object at "
            f"{self._interaction_distance:.0f}mm"
        )
        return True


    def _should_enter_face_id(self) -> bool:
        """
        Returns True if conditions are met to enter FACE_ID mode.

        Conditions:
          1. A face has been visible for at least FACE_ID_MIN_CONSECUTIVE_FRAMES
          2. Face detection confidence is above FACE_CONFIDENCE_THRESHOLD
        """

        if self._face_frames < FACE_ID_MIN_CONSECUTIVE_FRAMES:
            return False

        if self._face_confidence < FACE_CONFIDENCE_THRESHOLD:
            return False

        logger.debug(
            f"Face ID entry condition met: confidence={self._face_confidence:.2f}, "
            f"frames={self._face_frames}"
        )
        return True


    def _should_enter_banknote(self) -> bool:
        """
        Returns True if conditions are met to enter BANKNOTE mode.

        Conditions:
          1. A banknote has been visible for at least BANKNOTE_MIN_CONSECUTIVE_FRAMES
          2. User is relatively still (banknote scanning needs steady camera)
        """

        if self._banknote_frames < BANKNOTE_MIN_CONSECUTIVE_FRAMES:
            return False

        # The motion check is already encoded in _banknote_frames,
        # but we double-check here for safety.
        if self._motion_level > MAX_MOTION_FOR_STILL_MODES:
            return False

        logger.debug(
            f"Banknote entry condition met: frames={self._banknote_frames}, "
            f"motion={self._motion_level:.2f}m/s²"
        )
        return True


    def _emergency_override(self, obstacle_result: Dict) -> bool:
        """
        Returns True if a DANGER obstacle requires immediate return to NAVIGATION.

        This is the highest-priority check. It fires if ANY confirmed track
        is within DANGER_DIST_MM, regardless of what mode we are in.

        The only exception is if we are ALREADY in NAVIGATION mode — in that
        case there is nothing to override.
        """

        # If we're already in NAVIGATION, no override needed.
        if self._current_mode == MODE.NAVIGATION:
            return False

        # Check if there are any DANGER-level obstacles.
        # obstacle_result["danger"] is the list of tracks in the danger zone.
        danger_tracks = obstacle_result.get("danger", [])

        if danger_tracks:
            # Log which object triggered the override.
            # danger_tracks[0] is the most urgent (sorted by distance).
            most_urgent = danger_tracks[0]
            logger.warning(
                f"Emergency override: {most_urgent['label']} at "
                f"{most_urgent['distance_mm']:.0f}mm is in DANGER zone."
            )
            return True

        return False


    # =========================================================================
    # IMU MOTION ANALYSIS
    # =========================================================================

    def _get_imu_motion_level(self, bundle: Dict) -> float:
        """
        Calculates a single motion level value from the IMU data.

        Uses the accelerometer to measure total acceleration magnitude.
        High value = user is moving. Low value = user is still.

        The accelerometer measures total acceleration including gravity
        (which is always ~9.81 m/s² downward). Since we use ACCELEROMETER_RAW
        (not LINEAR_ACCELERATION), we need to account for gravity.

        In practice, when the user is still, the accelerometer reads ~9.81 m/s²
        (just gravity). When they walk, it reads 10-14 m/s² due to head bob.

        We subtract 9.81 to get the "excess" acceleration above gravity,
        giving us a motion level of ~0 when still and ~2-4 when walking.

        Returns:
            Motion level in m/s² (0.0 = perfectly still, higher = more movement)
        """

        # Safely extract the IMU data from the bundle.
        # .get() with a default prevents KeyError if imu key is missing.
        imu  = bundle.get("imu", {})
        accel = imu.get("accel", {"x": 0.0, "y": 0.0, "z": 9.81})

        # Extract the three acceleration components.
        ax = float(accel.get("x", 0.0))
        ay = float(accel.get("y", 0.0))
        az = float(accel.get("z", 9.81))

        # Calculate the magnitude of the acceleration vector.
        # Magnitude = sqrt(ax² + ay² + az²)
        # This is the total acceleration regardless of direction.
        # ** 2 = squared, ** 0.5 = square root
        magnitude = (ax**2 + ay**2 + az**2) ** 0.5

        # Subtract the gravity component (9.81 m/s²) to get excess motion.
        # abs() ensures we get a positive value even if magnitude < 9.81.
        # This shouldn't happen normally but protects against sensor noise.
        motion = abs(magnitude - 9.81)

        return motion


    # =========================================================================
    # MODE SWITCHING
    # =========================================================================

    def switch_to(self, new_mode: str, reason: str = ""):
        """
        Performs the actual mode switch.

        Steps:
          1. Call on_exit callbacks for the current mode
          2. Log the transition
          3. Update the current mode and timestamp
          4. Record the transition in history
          5. Call on_enter callbacks for the new mode

        Arguments:
            new_mode: the mode to switch to (use MODE.X constants)
            reason:   human-readable explanation for why we're switching
        """

        old_mode = self._current_mode

        # ── Call on_exit callbacks for the old mode ────────────────────────────
        # These are functions registered by other modules to run when
        # we leave a particular mode.
        # Example: obstacle_detector.reset_tracker() when leaving NAVIGATION.
        for callback in self._callbacks.get(old_mode, {}).get("on_exit", []):
            try:
                # Call the callback function.
                # try/except prevents a broken callback from crashing everything.
                callback()
            except Exception as e:
                logger.error(f"on_exit callback error for {old_mode}: {e}")

        # ── Log the transition ─────────────────────────────────────────────────
        logger.info(
            f"Mode switch: {old_mode} → {new_mode}"
            + (f" ({reason})" if reason else "")
        )

        # ── Update state ───────────────────────────────────────────────────────
        # Update the current mode.
        self._current_mode = new_mode

        # Record when we entered this new mode.
        # This is used by is_stable() to check dwell time.
        self._mode_entered_at = time.time()

        # ── Record in history ──────────────────────────────────────────────────
        # Create a ModeTransition record and add it to the history list.
        transition = ModeTransition(
            from_mode    = old_mode,
            to_mode      = new_mode,
            timestamp_ms = get_timestamp_ms(),
            reason       = reason
        )
        self._history.append(transition)

        # Keep only the last _max_history entries.
        # If the list is too long, remove the oldest entry from the front.
        # [-self._max_history:] is a Python slice that keeps the LAST N items.
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # ── Call on_enter callbacks for the new mode ───────────────────────────
        # These are functions registered by other modules to run when
        # we enter a particular mode.
        # Example: ocr_module.prepare() when entering OCR mode.
        for callback in self._callbacks.get(new_mode, {}).get("on_enter", []):
            try:
                callback()
            except Exception as e:
                logger.error(f"on_enter callback error for {new_mode}: {e}")


    # =========================================================================
    # CALLBACK REGISTRATION
    # =========================================================================

    def register_callback(
        self,
        mode:     str,
        on_enter: Optional[Callable] = None,
        on_exit:  Optional[Callable] = None
    ):
        """
        Registers functions to be called when a mode is entered or exited.

        Other modules call this once during setup to hook into mode changes.
        The state machine then calls these functions automatically at the
        right moment — the caller doesn't need to check the mode manually.

        Arguments:
            mode:     the mode to register for (use MODE.X constants)
            on_enter: function to call when we ENTER this mode (optional)
            on_exit:  function to call when we EXIT this mode (optional)

        Example usage in control_unit.py:
            sm.register_callback(
                mode     = MODE.NAVIGATION,
                on_exit  = obstacle_detector.reset_tracker
            )
            sm.register_callback(
                mode     = MODE.OCR,
                on_enter = audio.stop_obstacle_alerts,
                on_exit  = audio.resume_obstacle_alerts
            )
        """

        # Make sure the mode has an entry in our callbacks dictionary.
        # This handles the case where someone registers a callback for a
        # mode that wasn't in our original initialisation list.
        if mode not in self._callbacks:
            self._callbacks[mode] = {"on_enter": [], "on_exit": []}

        # Register the on_enter callback if provided.
        # 'if on_enter' checks if the argument is not None.
        if on_enter:
            self._callbacks[mode]["on_enter"].append(on_enter)
            logger.debug(f"Registered on_enter callback for {mode}: {on_enter.__name__}")

        # Register the on_exit callback if provided.
        if on_exit:
            self._callbacks[mode]["on_exit"].append(on_exit)
            logger.debug(f"Registered on_exit callback for {mode}: {on_exit.__name__}")


    # =========================================================================
    # STATE ACCESSORS
    # =========================================================================

    def get_mode(self) -> str:
        """
        Returns the current mode string.

        Use this in control_unit.py to decide which modules to run:
            mode = sm.get_mode()
            if mode == MODE.NAVIGATION:
                run_navigation_logic()
        """
        return self._current_mode


    def get_mode_duration(self) -> float:
        """
        Returns how many seconds we have been in the current mode.

        Used for dwell time checks and diagnostics.
        time.time() gives current time, _mode_entered_at gives entry time.
        Their difference is the duration.
        """
        return time.time() - self._mode_entered_at


    def is_stable(self) -> bool:
        """
        Returns True if the minimum dwell time for the current mode has passed.

        The dwell time is the minimum time a mode must stay active before
        we are allowed to switch away from it. This prevents flickering.

        MODE_DWELL_TIMES is the dictionary we defined at the top of this file.
        .get() with a default of 1.0 handles any unlisted mode safely.
        """

        # Look up the minimum dwell time for the current mode.
        min_dwell = MODE_DWELL_TIMES.get(self._current_mode, 1.0)

        # Check if we've been in this mode for at least that long.
        return self.get_mode_duration() >= min_dwell


    def is_in_mode(self, mode: str) -> bool:
        """
        Convenience function — returns True if we are in the given mode.

        Slightly more readable than: sm.get_mode() == MODE.NAVIGATION
        Usage: if sm.is_in_mode(MODE.NAVIGATION): ...
        """
        return self._current_mode == mode


    def get_history(self, last_n: int = 10) -> List[ModeTransition]:
        """
        Returns the last N mode transitions from the history log.

        Useful for debugging unexpected mode switches.

        Arguments:
            last_n: how many recent transitions to return (default 10)

        Returns:
            List of ModeTransition dataclass instances.
        """

        # [-last_n:] is a Python slice that returns the last N items.
        # If the list has fewer than last_n items, returns all of them.
        return self._history[-last_n:]


    def get_stats(self) -> Dict:
        """
        Returns a dictionary of diagnostic statistics about the state machine.

        Useful for the debug overlay and log files.
        """

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
        """
        Forcibly switches to a mode, bypassing dwell time and all conditions.

        Use this for testing or for manual user-triggered mode changes
        (e.g. a button press that forces OCR mode).

        NOT used in normal operation — normal operation uses update().
        """

        logger.info(f"Force mode: {mode} ({reason})")
        self.switch_to(mode, reason=reason)


# =============================================================================
# SELF-TEST
# =============================================================================
# Tests the state machine WITHOUT a camera — simulates sensor data manually.
# Run with: python state_machine.py

if __name__ == "__main__":

    print("=== ECHORA state_machine.py self-test ===\n")

    # Create the state machine.
    sm = StateMachine()

    # ── Register some test callbacks ──────────────────────────────────────────
    # These simple functions just print messages so we can verify
    # callbacks are being called at the right times.

    def on_enter_ocr():
        print("  [CALLBACK] Entering OCR mode — stopping obstacle audio")

    def on_exit_ocr():
        print("  [CALLBACK] Exiting OCR mode — resuming obstacle audio")

    def on_enter_face():
        print("  [CALLBACK] Entering FACE_ID mode — starting face recognition")

    # Register the callbacks.
    sm.register_callback(mode=MODE.OCR,     on_enter=on_enter_ocr, on_exit=on_exit_ocr)
    sm.register_callback(mode=MODE.FACE_ID, on_enter=on_enter_face)

    # ── Helper: build a fake camera bundle with IMU data ──────────────────────
    def make_bundle(accel_x=0.0, accel_y=0.0, accel_z=9.81):
        """
        Creates a minimal fake camera bundle with just IMU data.
        Used to simulate different motion levels.
        """
        return {
            "rgb":   None,
            "depth": None,
            "imu": {
                "accel": {"x": accel_x, "y": accel_y, "z": accel_z},
                "gyro":  {"x": 0.0,     "y": 0.0,     "z": 0.0},
                "timestamp_ms": get_timestamp_ms()
            },
            "timestamp_ms": get_timestamp_ms()
        }

    # ── Helper: build a fake obstacle result ─────────────────────────────────
    def make_obstacle_result(danger=False, warning=False):
        """
        Creates a minimal fake obstacle detection result.
        """
        danger_track = [{
            "label": "chair", "distance_mm": 500,
            "urgency": "DANGER", "angle_deg": 0.0
        }] if danger else []

        warning_track = [{
            "label": "table", "distance_mm": 1500,
            "urgency": "WARNING", "angle_deg": 10.0
        }] if warning else []

        return {
            "tracks":  danger_track + warning_track,
            "danger":  danger_track,
            "warning": warning_track,
            "safe":    [],
        }

    # ── Test 1: Start in NAVIGATION ───────────────────────────────────────────
    print("Test 1: Initial mode")
    bundle   = make_bundle()
    obs      = make_obstacle_result()
    mode     = sm.update(bundle, obs)
    print(f"  Mode: {mode} (expected: NAVIGATION)")
    assert mode == MODE.NAVIGATION
    print("  PASSED\n")

    # ── Test 2: OCR trigger — text appears for 1 frame (not yet stable) ───────
    print("Test 2: OCR trigger — 1 frame (should NOT switch yet)")
    mode = sm.update(
        make_bundle(), obs,
        ocr_text_distance=600.0   # text at 600mm, within OCR_TRIGGER_DIST_MM
    )
    print(f"  Mode: {mode} (expected: NAVIGATION — need 2 frames)")
    assert mode == MODE.NAVIGATION
    print("  PASSED\n")

    # ── Test 3: Wait for dwell time, then trigger OCR for 2 frames ───────────
    print("Test 3: OCR trigger — 2 consecutive frames + dwell time passed")

    # Simulate dwell time passing by temporarily backdating the entry time.
    # This is only for testing — in production, real time passes naturally.
    sm._mode_entered_at = time.time() - 2.0   # pretend we entered 2 seconds ago

    # First frame with text — ocr_frames goes to 1 (or 2 since test 2 ran)
    # Actually test 2 already gave us 1 frame, so this gives us 2.
    mode = sm.update(
        make_bundle(), obs,
        ocr_text_distance=600.0
    )
    print(f"  Mode: {mode} (expected: OCR)")
    assert mode == MODE.OCR, f"Expected OCR but got {mode}"
    print("  PASSED\n")

    # ── Test 4: Emergency override — DANGER appears while in OCR ─────────────
    print("Test 4: Emergency override — DANGER obstacle while in OCR mode")

    # Make sure we're stable in OCR first.
    sm._mode_entered_at = time.time() - 5.0

    mode = sm.update(
        make_bundle(),
        make_obstacle_result(danger=True),  # DANGER obstacle!
        ocr_text_distance=600.0
    )
    print(f"  Mode: {mode} (expected: NAVIGATION — emergency override)")
    assert mode == MODE.NAVIGATION
    print("  PASSED\n")

    # ── Test 5: Fast motion suppresses OCR ───────────────────────────────────
    print("Test 5: Fast motion — OCR should NOT trigger")

    # Full reset between tests — clear ALL state to avoid cross-test pollution.
    # In production this never happens because tests don't share state.
    sm._mode_entered_at = time.time() - 2.0
    sm._ocr_frames = 0
    sm._face_frames = 0
    sm._banknote_frames = 0
    sm._interaction_frames = 0

    # Make sure we are back in NAVIGATION before this test.
    if sm._current_mode != MODE.NAVIGATION:
        sm.force_mode(MODE.NAVIGATION, reason="test reset")
        sm._mode_entered_at = time.time() - 2.0

    # Simulate walking — high acceleration in X direction.
    # accel_x=4.0 gives magnitude = sqrt(4² + 0.5² + 9.81²) ≈ 10.6 m/s²
    # motion = |10.6 - 9.81| = 0.79...
    # Wait — that's below MAX_MOTION_FOR_STILL_MODES (2.0).
    # We need stronger acceleration to simulate real walking.
    # accel_x=6.0 gives magnitude ≈ sqrt(36 + 0 + 96.2) ≈ 11.5 m/s²
    # motion = |11.5 - 9.81| = 1.69 — still below 2.0.
    # Use accel_x=8.0: magnitude ≈ sqrt(64 + 0 + 96.2) ≈ 12.67 m/s²
    # motion = |12.67 - 9.81| = 2.86 — above MAX_MOTION_FOR_STILL_MODES (2.0). ✓
    walking_bundle = make_bundle(accel_x=8.0, accel_y=0.0, accel_z=9.81)

    for i in range(5):
        mode = sm.update(
            walking_bundle, obs,
            ocr_text_distance=600.0
        )

    print(f"  Motion level: {sm._motion_level:.2f} m/s²")
    print(f"  OCR frames:   {sm._ocr_frames}")
    print(f"  Mode: {mode} (expected: NAVIGATION — user moving too fast)")
    assert mode == MODE.NAVIGATION
    print("  PASSED\n")
    # ── Test 6: Face ID trigger ────────────────────────────────────────────────
    print("Test 6: Face ID trigger — stable face detection")

    sm._mode_entered_at = time.time() - 2.0

    # Simulate 3 consecutive frames with a high-confidence face.
    for i in range(3):
        mode = sm.update(
            make_bundle(), obs,
            face_confidence=0.92
        )
        print(f"  Frame {i+1}: mode={mode}, face_frames={sm._face_frames}")

    print(f"  Final mode: {mode} (expected: FACE_ID)")
    assert mode == MODE.FACE_ID
    print("  PASSED\n")

    # ── Test 7: History log ───────────────────────────────────────────────────
    print("Test 7: Mode transition history")
    history = sm.get_history()
    print(f"  Last {len(history)} transitions:")
    for t in history:
        ms = int(t.timestamp_ms)
        print(f"    {t.from_mode:12s} → {t.to_mode:12s}  ({t.reason})")
    print("  PASSED\n")

    # ── Test 8: Stats ─────────────────────────────────────────────────────────
    print("Test 8: Stats")
    stats = sm.get_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")
    print("  PASSED\n")

    print("=== All tests passed ===")