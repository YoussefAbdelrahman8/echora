# =============================================================================
# control_unit.py — ECHORA Central Brain
# =============================================================================
# Data flow every frame:
#   camera → obstacle_detection → state_machine → [mode handler] → audio/haptic
#
# Two top-level operating modes:
#
#   AUTO MODE   (default) — production wearable mode
#     State machine decides which mode is active automatically.
#     Emergency override is active.
#
#   MANUAL MODE — developer testing mode
#     Keyboard controls which mode is active.
#     State machine is NOT called — zero side effects, zero callbacks.
#     Emergency override is disabled — mode is fully locked.
#     Bounding boxes only shown in NAVIGATION mode.
#
# Keyboard controls (always shown in debug window):
#   TAB  — toggle AUTO / MANUAL mode
#   1    — force NAVIGATION   (manual only)
#   2    — force OCR          (manual only)
#   3    — force INTERACTION  (manual only)
#   4    — force FACE_ID      (manual only)
#   5    — force BANKNOTE     (manual only)
#   Q    — quit
# =============================================================================


# =============================================================================
# WINDOWS PROCESS PRIORITY
# =============================================================================
import sys
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(),
        0x00000080
    )


# =============================================================================
# IMPORTS
# =============================================================================

import cv2
import numpy as np
import time
import threading
from typing import Optional, Dict

from config import (
    MODE,
    CAMERA_RGB_WIDTH,
    CAMERA_RGB_HEIGHT,
    MAX_FRAME_TIME_MS,
    LOG_LEVEL,
)
from utils import logger, get_timestamp_ms, draw_overlay, mm_to_spoken
from camera import EchoraCamera
from obstacle_detection import ObstacleDetector
from state_machine import StateMachine
from audio_feedback import AudioFeedback, SpeechPriority

import ocr
from interaction_detection import InteractionDetector
import banknote
import echora_face as face_recognition


# =============================================================================
# DEBUG CONFIGURATION
# =============================================================================

SHOW_DEBUG_WINDOW       = True
PERF_LOG_EVERY_N_FRAMES = 30


# =============================================================================
# CONTROL UNIT CLASS
# =============================================================================

class ControlUnit:

    def __init__(self, start_in_manual: bool = False):
        """
        Arguments:
            start_in_manual: if True, ECHORA starts in MANUAL mode.
                             Passed from main.py via --manual flag.
        """

        # Sub-modules
        self._camera:               Optional[EchoraCamera]        = None
        self._detector:             Optional[ObstacleDetector]    = None
        self._state_machine:        Optional[StateMachine]        = None
        self._audio:                Optional[AudioFeedback]       = None
        self._interaction_detector: Optional[InteractionDetector] = None

        # State tracking
        self._started:     bool = False
        self._running:     bool = False
        self._frame_count: int  = 0

        # Performance monitoring
        self._frame_times: list  = []
        self._slow_frames: int   = 0
        self._start_time:  float = 0.0

        # Mode-specific state
        self._last_mode:         str = MODE.NAVIGATION
        self._last_scene_desc:   str = ""
        self._last_ocr_text:     str = ""
        self._last_face_name:    str = ""
        self._last_denomination: str = ""

        # Rate-limited signal cache
        self._last_ocr_dist:      float = 0.0
        self._last_face_conf:     float = 0.0
        self._last_note_visible:  bool  = False
        self._last_interact_dist: float = 0.0

        # Background thread state
        self._ocr_running:     bool          = False
        self._face_id_running: bool          = False
        self._face_id_result:  Optional[Dict] = None

        # ── AUTO / MANUAL control ──────────────────────────────────────────────
        # _auto_mode = True  -> state machine decides mode (production)
        # _auto_mode = False -> keyboard decides mode (developer testing)
        #
        # KEY DESIGN:
        #   In MANUAL mode, state_machine.update() is NOT called at all.
        #   This prevents all state machine side effects:
        #     - No on_enter / on_exit callbacks firing
        #     - No audio announcements from state machine
        #     - No console mode-switch log messages
        #     - No emergency override
        #   The mode is 100% controlled by keyboard only.
        self._auto_mode: bool = not start_in_manual

        # Which mode is locked when _auto_mode is False.
        # Always starts on NAVIGATION — safe default.
        self._manual_mode: str = MODE.NAVIGATION

        # Maps keyboard key codes to mode names.
        self._key_to_mode: Dict[int, str] = {
            ord('1'): MODE.NAVIGATION,
            ord('2'): MODE.OCR,
            ord('3'): MODE.INTERACTION,
            ord('4'): MODE.FACE_ID,
            ord('5'): MODE.BANKNOTE,
        }

        logger.info(
            f"ControlUnit created. "
            f"Starting in {'AUTO' if self._auto_mode else 'MANUAL'} mode."
        )


    # =========================================================================
    # STARTUP
    # =========================================================================

    def startup(self):

        logger.info("=" * 60)
        logger.info("ECHORA Control Unit starting up...")
        logger.info("=" * 60)

        self._start_time = time.time()

        logger.info("Step 1: Starting camera...")
        self._camera = EchoraCamera()
        self._camera.init_pipeline()
        logger.info("Camera ready.")

        logger.info("Step 2: Loading obstacle detector...")
        self._detector = ObstacleDetector()
        self._detector.load_model()
        logger.info("Obstacle detector ready.")

        logger.info("Step 2b: Loading interaction detector...")
        self._interaction_detector = InteractionDetector()
        self._interaction_detector.load_model()
        logger.info("Interaction detector ready.")

        logger.info("Step 2c: Initialising OCR...")
        import ocr as ocr_module
        ocr_module.init_ocr()
        logger.info("OCR ready.")

        logger.info("Step 2d: Initialising banknote detector...")
        import banknote as banknote_module
        banknote_module.init_banknote()
        logger.info("Banknote detector ready.")

        logger.info("Step 2e: Initialising database...")
        from database import init_database
        init_database()
        logger.info("Database ready.")

        logger.info("Step 2f: Initialising face recognition...")
        from echora_face import init_face_recognition
        init_face_recognition()
        logger.info("Face recognition ready.")

        logger.info("Step 2g: Initialising haptic feedback...")
        from haptic_feedback import init_haptic
        init_haptic()
        logger.info("Haptic feedback ready.")

        logger.info("Step 3: Starting audio system...")
        self._audio = AudioFeedback()
        self._audio.init_audio()
        logger.info("Audio ready.")

        logger.info("Step 4: Initialising state machine...")
        self._state_machine = StateMachine()
        logger.info("State machine ready.")

        logger.info("Step 5: Registering callbacks...")
        self._register_callbacks()
        logger.info("Callbacks registered.")

        time.sleep(0.3)

        if self._auto_mode:
            self._audio.speak(
                "ECHORA online. Auto mode active.",
                priority=SpeechPriority.NORMAL
            )
        else:
            self._audio.speak(
                "ECHORA online. Manual testing mode. "
                "Press 1 to 5 to select a mode.",
                priority=SpeechPriority.NORMAL
            )

        self._started = True
        logger.info("=" * 60)
        logger.info("ECHORA startup complete. Entering main loop.")
        logger.info("=" * 60)


    # =========================================================================
    # AUTO / MANUAL TOGGLE
    # =========================================================================

    def _toggle_auto_manual(self):
        """
        Toggles between AUTO and MANUAL mode when TAB is pressed.

        Switching TO manual:
          - Locks mode to whatever is currently active
          - State machine will NOT be called next frame
          - All callbacks, audio, and mode-switching from state machine stop

        Switching TO auto:
          - State machine resumes full control
          - First update() call will re-evaluate the current scene
        """

        self._auto_mode = not self._auto_mode

        if self._auto_mode:
            logger.info("Switched to AUTO mode.")
            self._audio.speak(
                "Auto mode.",
                priority=SpeechPriority.HIGH,
                ttl_sec=2.0
            )
        else:
            self._manual_mode = self._last_mode
            logger.info(
                f"Switched to MANUAL mode. "
                f"Locked on: {self._manual_mode}."
            )
            self._audio.speak(
                f"Manual mode. "
                f"{self._manual_mode.lower().replace('_', ' ')} locked.",
                priority=SpeechPriority.HIGH,
                ttl_sec=3.0
            )


    def _set_manual_mode(self, new_mode: str):
        """
        Forces a specific mode in MANUAL operation.
        Called when keys 1-5 are pressed.
        Silently ignored in AUTO mode.
        """

        if self._auto_mode:
            return

        if new_mode == self._manual_mode:
            return

        old_mode          = self._manual_mode
        self._manual_mode = new_mode
        self._last_mode   = new_mode

        # Reset mode-specific state so new mode starts fresh.
        if new_mode == MODE.OCR:
            self._reset_ocr_state()
        elif new_mode == MODE.FACE_ID:
            self._reset_face_state()
            face_recognition.reset_face()
        elif new_mode == MODE.BANKNOTE:
            self._reset_banknote_state()
        elif new_mode == MODE.INTERACTION:
            self._reset_interaction_state()

        logger.info(f"Manual mode: {old_mode} -> {new_mode}")

        # Announce new mode — this is the ONLY audio call for mode change
        # in MANUAL mode. State machine callbacks are suppressed entirely.
        self._audio.announce_mode_change(new_mode)


    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def _on_enter_interaction(self):
        target = self._interaction_detector._target_object
        if target:
            label    = target.get("label", "object")
            dist     = target.get("distance_mm", 0)
            dist_str = mm_to_spoken(dist)
            self._audio.speak(
                f"{label} detected. {dist_str}. Raise your hand to reach it.",
                priority=SpeechPriority.HIGH
            )
            logger.info(f"INTERACTION: target={label} at {dist:.0f}mm")
        else:
            self._audio.announce_mode_change(MODE.INTERACTION)

    def _reset_ocr_state(self):
        self._last_ocr_text = ""

    def _reset_face_state(self):
        self._last_face_name = ""

    def _reset_banknote_state(self):
        self._last_denomination = ""

    def _reset_interaction_state(self):
        if self._interaction_detector:
            self._interaction_detector.reset()

    def _register_callbacks(self):

        sm = self._state_machine

        sm.register_callback(
            mode     = MODE.NAVIGATION,
            on_enter = lambda: self._audio.announce_mode_change(MODE.NAVIGATION)
        )
        sm.register_callback(
            mode    = MODE.NAVIGATION,
            on_exit = self._detector.reset_tracker
        )
        sm.register_callback(
            mode     = MODE.OCR,
            on_enter = lambda: (
                self._audio.announce_mode_change(MODE.OCR),
                self._audio.stop_all()
            )
        )
        sm.register_callback(
            mode    = MODE.OCR,
            on_exit = lambda: (
                self._reset_ocr_state(),
                __import__('ocr').reset_ocr()
            )
        )
        sm.register_callback(
            mode     = MODE.INTERACTION,
            on_enter = self._on_enter_interaction
        )
        sm.register_callback(
            mode    = MODE.INTERACTION,
            on_exit = self._reset_interaction_state
        )
        sm.register_callback(
            mode     = MODE.FACE_ID,
            on_enter = lambda: self._audio.announce_mode_change(MODE.FACE_ID)
        )
        sm.register_callback(
            mode    = MODE.FACE_ID,
            on_exit = lambda: (
                self._reset_face_state(),
                __import__('echora_face').reset_face()
            )
        )
        sm.register_callback(
            mode     = MODE.BANKNOTE,
            on_enter = lambda: self._audio.announce_mode_change(MODE.BANKNOTE)
        )
        sm.register_callback(
            mode    = MODE.BANKNOTE,
            on_exit = lambda: (
                self._reset_banknote_state(),
                __import__('banknote').reset_banknote()
            )
        )


    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self):

        if not self._started:
            logger.error("Cannot run — call startup() first.")
            return

        self._running = True
        logger.info("Main loop started.")
        logger.info("TAB = toggle AUTO/MANUAL | 1-5 = set mode | Q = quit")

        try:
            while self._running:

                bundle = self._camera.get_synced_bundle()
                if bundle is None:
                    continue

                frame_start    = get_timestamp_ms()
                debug_frame    = self._process_frame(bundle)
                frame_duration = get_timestamp_ms() - frame_start

                self._frame_times.append(frame_duration)
                self._frame_times = self._frame_times[-30:]

                if frame_duration > MAX_FRAME_TIME_MS:
                    self._slow_frames += 1

                if SHOW_DEBUG_WINDOW and debug_frame is not None:
                    cv2.imshow("ECHORA — Debug", debug_frame)
                    key = cv2.waitKey(1)
                    self._handle_key(key)

                self._frame_count += 1
                if self._frame_count % PERF_LOG_EVERY_N_FRAMES == 0:
                    self._log_performance()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt.")

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)

        finally:
            if SHOW_DEBUG_WINDOW:
                cv2.destroyAllWindows()
            self.shutdown()


    def _handle_key(self, key: int):
        """Processes keyboard input every frame. key=-1 means no key pressed."""

        if key == -1:
            return

        if key == ord('q') or key == ord('Q'):
            logger.info("Q pressed — quitting.")
            self._running = False
            return

        if key == 9:
            # TAB key — ASCII code 9
            self._toggle_auto_manual()
            return

        if key in self._key_to_mode:
            # Keys 1-5
            self._set_manual_mode(self._key_to_mode[key])
            return


    # =========================================================================
    # FRAME PROCESSING
    # =========================================================================

    def _process_frame(self, bundle: Dict) -> Optional[np.ndarray]:

        try:
            rgb_frame = bundle["rgb"]
            depth_map = bundle["depth"]

            # ── PRIORITY ZERO: Face ID result from background thread ───────────
            # Checked every frame regardless of current mode.
            # pyttsx3 must always be called from the main thread.
            if self._face_id_result is not None:
                result               = self._face_id_result
                self._face_id_result = None

                name = result.get("name", "")

                if name and name != self._last_face_name:
                    self._last_face_name = name
                    self._audio.speak(
                        f"This is {name}.",
                        priority = SpeechPriority.HIGH,
                        ttl_sec  = 8.0
                    )
                    logger.info(f"Face announced: {name}")

                elif not name and self._last_face_name == "":
                    self._last_face_name = "unknown"
                    self._audio.speak(
                        "Unknown person.",
                        priority = SpeechPriority.NORMAL,
                        ttl_sec  = 4.0
                    )

            # ── Step 1: YOLO — every frame ─────────────────────────────────────
            # YOLO always runs regardless of mode.
            # We need obstacle_result for the debug overlay and for NAVIGATION
            # mode even in MANUAL. In other modes we just don't act on it.
            obstacle_result = self._detector.update(bundle)

            # ── Step 2: Rate-limited perception signals ────────────────────────

            # OCR — background thread, every 15 frames
            if self._frame_count % 20 == 0 and not self._ocr_running:
                self._ocr_running = True
                rgb_copy   = rgb_frame.copy()
                depth_copy = depth_map.copy()

                def _ocr_worker(f=rgb_copy, d=depth_copy):
                    self._last_ocr_dist = ocr.get_text_distance(f, d)
                    self._ocr_running   = False

                threading.Thread(target=_ocr_worker, daemon=True).start()

            ocr_dist = self._last_ocr_dist

            # Face detection — every 5 frames, skip in FACE_ID mode
            if self._frame_count % 5 == 0 and self._last_mode != MODE.FACE_ID:
                self._last_face_conf = face_recognition.detect_face(rgb_frame)
            face_conf = self._last_face_conf

            # Banknote detection — every 5 frames
            if self._frame_count % 5 == 0:
                self._last_note_visible = banknote.detect_banknote(rgb_frame)
            note_visible = self._last_note_visible

            # Interactable scan — every 5 frames
            if self._frame_count % 5 == 0:
                current_tracks = obstacle_result.get("tracks", [])
                self._last_interact_dist = (
                    self._interaction_detector.scan_for_interactables(
                        detections = current_tracks,
                        depth_map  = depth_map
                    )
                )
            interact_dist = self._last_interact_dist

            # ── Step 3: Determine current mode ────────────────────────────────
            #
            # AUTO MODE:
            #   State machine evaluates all signals and decides the mode.
            #   Its callbacks fire — audio announcements, resets, etc.
            #   Emergency override is fully active.
            #
            # MANUAL MODE:
            #   State machine is NOT called at all.
            #   No callbacks, no audio from state machine, no mode switches.
            #   No emergency override.
            #   The mode you set with 1-5 is the mode. Period.

            if self._auto_mode:
                # Full production mode — state machine in complete control.
                current_mode = self._state_machine.update(
                    bundle                = bundle,
                    obstacle_result       = obstacle_result,
                    ocr_text_distance     = ocr_dist,
                    face_confidence       = face_conf,
                    interactable_distance = interact_dist,
                    banknote_visible      = note_visible,
                )

                # Log AUTO mode changes.
                if current_mode != self._last_mode:
                    logger.info(f"Mode: {self._last_mode} -> {current_mode} [AUTO]")
                    self._last_mode = current_mode

            else:
                # Manual testing mode — state machine NOT called.
                # _manual_mode was set by keyboard and does not change here.
                current_mode = self._manual_mode

                # _last_mode is updated by _set_manual_mode() when a key
                # is pressed. No update needed here.

            # ── Step 4: Mode handler ───────────────────────────────────────────
            if current_mode == MODE.NAVIGATION:
                self._handle_navigation(bundle, obstacle_result)
            elif current_mode == MODE.OCR:
                self._handle_ocr(bundle)
            elif current_mode == MODE.INTERACTION:
                self._handle_interaction(bundle, obstacle_result)
            elif current_mode == MODE.FACE_ID:
                self._handle_face_id(bundle)
            elif current_mode == MODE.BANKNOTE:
                self._handle_banknote(bundle)

            # ── Step 5: Build debug frame ──────────────────────────────────────
            debug_frame = rgb_frame.copy()

            # ── FIX: Only draw bounding boxes in NAVIGATION mode ───────────────
            # In other modes (OCR, FACE_ID, BANKNOTE) the boxes are distracting
            # and irrelevant. In INTERACTION mode the interaction overlay handles
            # its own visuals.
            if current_mode == MODE.NAVIGATION and obstacle_result["tracks"]:
                debug_frame = draw_overlay(debug_frame, obstacle_result["tracks"])

            if current_mode == MODE.INTERACTION:
                if self._interaction_detector._last_grid is not None:
                    interaction_result = {
                        "phase":          self._interaction_detector._phase,
                        "hand":           None,
                        "target":         self._interaction_detector._target_object,
                        "electrode_grid": self._interaction_detector._last_grid,
                    }
                    debug_frame = self._interaction_detector.draw_debug_overlay(
                        debug_frame, interaction_result
                    )

            debug_frame = self._draw_debug_overlay(
                debug_frame, obstacle_result, current_mode
            )

            return debug_frame

        except Exception as e:
            logger.error(f"Frame processing error: {e}", exc_info=True)
            return bundle.get("rgb")


    # =========================================================================
    # MODE HANDLERS
    # =========================================================================

    def _handle_navigation(self, bundle: Dict, obstacle_result: Dict):

        danger_tracks  = obstacle_result.get("danger",  [])
        warning_tracks = obstacle_result.get("warning", [])
        scene_desc     = obstacle_result.get("scene_desc", "")

        for track in danger_tracks:
            self._audio.announce_obstacle(track)
        for track in warning_tracks:
            self._audio.announce_obstacle(track)

        if (scene_desc
                and scene_desc != self._last_scene_desc
                and len(scene_desc) > 10):
            self._last_scene_desc = scene_desc
            self._audio.announce_scene(scene_desc)
            logger.info(f"Scene: {scene_desc[:60]}...")


    def _handle_ocr(self, bundle: Dict):

        rgb_frame = bundle["rgb"]
        text      = ocr.read_text(rgb_frame)

        if not text or not text.strip():
            return

        if text.strip() != self._last_ocr_text:
            self._last_ocr_text = text.strip()
            self._audio.announce_ocr(text)
            logger.info(f"OCR: '{text[:60]}'")


    def _handle_interaction(self, bundle: Dict, obstacle_result: Dict):

        rgb_frame      = bundle["rgb"]
        depth_map      = bundle["depth"]
        current_tracks = obstacle_result.get("tracks", [])

        result = self._interaction_detector.update(
            rgb_frame  = rgb_frame,
            depth_map  = depth_map,
            detections = current_tracks
        )

        if result.get("on_target"):
            self._audio.speak("Object reached.", priority=SpeechPriority.HIGH)
            logger.info("Interaction SUCCESS.")
            if self._auto_mode:
                # In AUTO mode, return to NAVIGATION after success.
                # In MANUAL mode, stay in INTERACTION — developer controls flow.
                self._state_machine.force_mode(
                    MODE.NAVIGATION, reason="object reached"
                )


    def _handle_face_id(self, bundle: Dict):
        """
        Launches face ID background thread every 5 frames.
        Result is consumed at PRIORITY ZERO at the top of _process_frame().
        """

        if self._frame_count % 5 != 0:
            return

        if self._face_id_running:
            return

        self._face_id_running = True
        rgb_copy = bundle["rgb"].copy()

        def _face_worker(f=rgb_copy):
            try:
                name, details        = face_recognition.identify_face(f)
                self._face_id_result = {"name": name, "details": details}
            except Exception as e:
                logger.error(f"Face worker error: {e}")
                self._face_id_result = {"name": "", "details": ""}
            finally:
                self._face_id_running = False

        threading.Thread(target=_face_worker, daemon=True).start()


    def _handle_banknote(self, bundle: Dict):

        rgb_frame    = bundle["rgb"]
        denomination = banknote.classify_denomination(rgb_frame)

        if denomination and denomination != self._last_denomination:
            self._last_denomination = denomination
            self._audio.announce_banknote(denomination)
            logger.info(f"Banknote: {denomination}")


    # =========================================================================
    # DEBUG OVERLAY
    # =========================================================================

    def _draw_debug_overlay(
        self,
        frame:           np.ndarray,
        obstacle_result: Dict,
        current_mode:    str
    ) -> np.ndarray:

        h, w = frame.shape[:2]

        mode_colours = {
            MODE.NAVIGATION:  (0,   200,  80),
            MODE.OCR:         (255, 165,   0),
            MODE.INTERACTION: (0,   165, 255),
            MODE.FACE_ID:     (180,   0, 255),
            MODE.BANKNOTE:    (0,   215, 255),
        }
        mode_colour = mode_colours.get(current_mode, (200, 200, 200))

        # ── Top left: mode name + AUTO/MANUAL indicator ────────────────────────
        cv2.rectangle(frame, (0, 0), (370, 58), (0, 0, 0), -1)

        cv2.putText(
            frame, f"MODE: {current_mode}",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, mode_colour, 2, cv2.LINE_AA
        )

        if self._auto_mode:
            indicator_text   = "AUTO  (TAB to switch to manual)"
            indicator_colour = (0, 200, 80)
        else:
            indicator_text   = "MANUAL  (TAB: auto | 1-5: mode)"
            indicator_colour = (0, 165, 255)

        cv2.putText(
            frame, indicator_text,
            (8, 48), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, indicator_colour, 1, cv2.LINE_AA
        )

        # ── Top right: FPS ─────────────────────────────────────────────────────
        if self._frame_times:
            avg_ms = sum(self._frame_times) / len(self._frame_times)
            fps    = 1000.0 / max(avg_ms, 1)
        else:
            fps = 0.0

        fps_text = f"FPS: {fps:.1f}  F:{self._frame_count}"
        (text_w, _), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        cv2.rectangle(frame, (w - text_w - 16, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(
            frame, fps_text,
            (w - text_w - 8, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (200, 200, 200), 1, cv2.LINE_AA
        )

        # ── Bottom right: track counts (only relevant in NAVIGATION) ───────────
        if current_mode == MODE.NAVIGATION:
            n_danger  = len(obstacle_result.get("danger",  []))
            n_warning = len(obstacle_result.get("warning", []))
            n_tracks  = len(obstacle_result.get("tracks",  []))

            for i, line in enumerate(reversed([
                f"Tracks:  {n_tracks}",
                f"Danger:  {n_danger}",
                f"Warning: {n_warning}",
            ])):
                if   "Danger"  in line and n_danger  > 0: colour = (0,   0, 220)
                elif "Warning" in line and n_warning > 0: colour = (0, 165, 255)
                else:                                      colour = (180, 180, 180)
                cv2.putText(
                    frame, line,
                    (w - 140, h - 10 - (i * 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colour, 1, cv2.LINE_AA
                )

        # ── Bottom left: state machine stats (AUTO only) ───────────────────────
        if self._auto_mode:
            sm_stats = self._state_machine.get_stats()
            for i, line in enumerate([
                f"Motion: {sm_stats['motion_level']:.2f} m/s2",
                f"Stable: {'yes' if sm_stats['is_stable'] else 'no'}",
                f"In mode: {sm_stats['mode_duration_s']:.1f}s",
            ]):
                cv2.putText(
                    frame, line,
                    (8, h - 10 - (i * 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (160, 160, 160), 1, cv2.LINE_AA
                )
        else:
            # In MANUAL mode show a simple hint at bottom left instead.
            cv2.putText(
                frame,
                "State machine: BYPASSED",
                (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (100, 100, 100), 1, cv2.LINE_AA
            )

        # ── Center top: most urgent obstacle (NAVIGATION only) ─────────────────
        if current_mode == MODE.NAVIGATION:
            most_urgent = self._detector.get_most_urgent_obstacle()
            if most_urgent:
                urgent_text = (
                    f"{most_urgent['label']}  "
                    f"{most_urgent['distance_mm']:.0f}mm  "
                    f"{most_urgent['angle_deg']:+.0f}deg"
                )
                urgency_colour = {
                    "DANGER":  (0,   0, 220),
                    "WARNING": (0, 165, 255),
                    "SAFE":    (0, 200,  80),
                }.get(most_urgent["urgency"], (180, 180, 180))

                (tw, _), _ = cv2.getTextSize(
                    urgent_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                x_pos = (w - tw) // 2
                cv2.rectangle(
                    frame, (x_pos - 6, 60), (x_pos + tw + 6, 86),
                    (0, 0, 0), -1
                )
                cv2.putText(
                    frame, urgent_text,
                    (x_pos, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, urgency_colour, 1, cv2.LINE_AA
                )

        # ── Manual key hint strip (only in MANUAL mode) ────────────────────────
        if not self._auto_mode:
            hint = "1:NAVIGATION  2:OCR  3:INTERACTION  4:FACE_ID  5:BANKNOTE"
            (hw, _), _ = cv2.getTextSize(
                hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
            )
            hx = (w - hw) // 2
            cv2.rectangle(
                frame, (hx - 6, h - 80), (hx + hw + 6, h - 60),
                (0, 0, 0), -1
            )
            cv2.putText(
                frame, hint,
                (hx, h - 64), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (0, 165, 255), 1, cv2.LINE_AA
            )

        return frame


    # =========================================================================
    # PERFORMANCE MONITORING
    # =========================================================================

    def _log_performance(self):

        if not self._frame_times:
            return

        avg_ms = sum(self._frame_times) / len(self._frame_times)
        fps    = 1000.0 / max(avg_ms, 1)
        uptime = time.time() - self._start_time

        tracker_stats = self._detector.get_stats()["tracker"]

        if self._auto_mode:
            sm_stats = self._state_machine.get_stats()
            mode_str = sm_stats['current_mode']
        else:
            mode_str = f"MANUAL:{self._manual_mode}"

        logger.info(
            f"Performance | Frame {self._frame_count} | "
            f"FPS: {fps:.1f} | Avg: {avg_ms:.1f}ms | "
            f"Slow: {self._slow_frames} | Uptime: {uptime:.0f}s | "
            f"{'AUTO' if self._auto_mode else 'MANUAL'} | "
            f"Mode: {mode_str} | "
            f"Tracks: {tracker_stats['confirmed']}"
        )


    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def shutdown(self):

        if not self._started:
            return

        logger.info("Shutting down ECHORA...")

        if self._audio and self._audio._ready:
            self._audio.speak(
                "ECHORA shutting down.", priority=SpeechPriority.HIGH
            )
            time.sleep(2.0)

        if self._audio:
            self._audio.release()

        if self._interaction_detector:
            self._interaction_detector.release()

        if self._camera:
            self._camera.release()

        from haptic_feedback import get_haptic
        h = get_haptic()
        if h:
            h.disconnect()

        from database import get_db
        db = get_db()
        if db:
            db.close()

        uptime = time.time() - self._start_time
        logger.info(
            f"ECHORA stopped. Frames: {self._frame_count} | "
            f"Uptime: {uptime:.1f}s | Slow frames: {self._slow_frames}"
        )
        self._started = False
        logger.info("Shutdown complete.")