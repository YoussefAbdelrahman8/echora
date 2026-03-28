# =============================================================================
# control_unit.py — ECHORA Central Brain
# =============================================================================
# The conductor. Knows about every module. Makes all decisions.
# Runs the main loop. Routes perception outputs to feedback outputs.
#
# Data flow every frame:
#   camera → obstacle_detection → state_machine → [mode handler] → audio/haptic
#
# Run ECHORA with: python control_unit.py
# =============================================================================


# =============================================================================
# WINDOWS PROCESS PRIORITY — must be first, before any other import
# =============================================================================
# Sets this Python process to HIGH priority in the Windows scheduler.
# Prevents Windows from pausing ECHORA to service background apps.
# This alone can recover 5-10 FPS on Windows machines.
import sys

from sympy import false

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(),
        0x00000080   # HIGH_PRIORITY_CLASS
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

    def __init__(self):

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

        # Rate-limited signal cache — updated on schedule, not every frame.
        # Initialised here so _process_frame never gets AttributeError
        # on the very first frame before the rate limiter fires.
        self._last_ocr_dist:      float = 0.0
        self._last_face_conf:     float = 0.0
        self._last_note_visible:  bool  = False
        self._last_interact_dist: float = 0.0

        # OCR background thread state.
        # _ocr_running flag prevents two OCR threads running simultaneously.
        self._ocr_running: bool = False

        logger.info("ControlUnit created. Call startup() to begin.")


    # =========================================================================
    # STARTUP
    # =========================================================================

    def startup(self):

        logger.info("=" * 60)
        logger.info("ECHORA Control Unit starting up...")
        logger.info("=" * 60)

        self._start_time = time.time()

        # Step 1: Camera
        logger.info("Step 1: Starting camera...")
        self._camera = EchoraCamera()
        self._camera.init_pipeline()
        logger.info("Camera ready.")

        # Step 2: Obstacle detector
        logger.info("Step 2: Loading obstacle detector...")
        self._detector = ObstacleDetector()
        self._detector.load_model()
        logger.info("Obstacle detector ready.")

        # Step 2b: Interaction detector
        logger.info("Step 2b: Loading interaction detector...")
        self._interaction_detector = InteractionDetector()
        self._interaction_detector.load_model()
        logger.info("Interaction detector ready.")

        # Step 2c: OCR
        logger.info("Step 2c: Initialising OCR...")
        import ocr as ocr_module
        ocr_module.init_ocr()
        logger.info("OCR ready.")

        # Step 2d: Banknote detector
        logger.info("Step 2d: Initialising banknote detector...")
        import banknote as banknote_module
        banknote_module.init_banknote()
        logger.info("Banknote detector ready.")

        # ── Step 2e: Initialise database ──────────────────────────────────────────
        logger.info("Step 2e: Initialising database...")
        from database import init_database
        init_database()
        logger.info("Database ready.")

        # ── Step 2f: Initialise face recognition ──────────────────────────────────
        logger.info("Step 2f: Initialising face recognition...")
        from echora_face import init_face_recognition
        init_face_recognition()
        logger.info("Face recognition ready.")

        # Step 3: Audio
        logger.info("Step 3: Starting audio system...")
        self._audio = AudioFeedback()
        self._audio.init_audio()
        logger.info("Audio ready.")

        # Step 4: State machine
        logger.info("Step 4: Initialising state machine...")
        self._state_machine = StateMachine()
        logger.info("State machine ready.")

        # Step 5: Callbacks
        logger.info("Step 5: Registering mode callbacks...")
        self._register_callbacks()
        logger.info("Callbacks registered.")

        time.sleep(0.3)
        self._audio.speak(
            "ECHORA online. Navigation mode active.",
            priority=SpeechPriority.NORMAL
        )

        self._started = True
        logger.info("=" * 60)
        logger.info("ECHORA startup complete. Entering main loop.")
        logger.info("=" * 60)


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
            logger.info(
                f"INTERACTION mode entered. Target: {label} at {dist:.0f}mm"
            )
        else:
            self._audio.announce_mode_change(MODE.INTERACTION)

    def _reset_ocr_state(self):
        self._last_ocr_text = ""
        logger.debug("OCR state reset.")

    def _reset_face_state(self):
        self._last_face_name = ""
        logger.debug("Face state reset.")

    def _reset_banknote_state(self):
        self._last_denomination = ""
        logger.debug("Banknote state reset.")

    def _reset_interaction_state(self):
        if self._interaction_detector:
            self._interaction_detector.reset()
        logger.debug("Interaction state reset.")

    def _register_callbacks(self):

        sm = self._state_machine

        # NAVIGATION
        sm.register_callback(
            mode     = MODE.NAVIGATION,
            on_enter = lambda: self._audio.announce_mode_change(MODE.NAVIGATION)
        )
        sm.register_callback(
            mode    = MODE.NAVIGATION,
            on_exit = self._detector.reset_tracker
        )

        # OCR
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

        # INTERACTION
        sm.register_callback(
            mode     = MODE.INTERACTION,
            on_enter = self._on_enter_interaction
        )
        sm.register_callback(
            mode    = MODE.INTERACTION,
            on_exit = self._reset_interaction_state
        )

        # FACE_ID
        sm.register_callback(
            mode     = MODE.FACE_ID,
            on_enter = lambda: self._audio.announce_mode_change(MODE.FACE_ID)
        )
        sm.register_callback(
            mode=MODE.FACE_ID,
            on_exit=lambda: (
                self._reset_face_state(),
                __import__('echora_face').reset_face()
            )
        )
        # BANKNOTE
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
        logger.info("Main loop started. Press Q in the debug window to quit.")

        try:
            while self._running:

                bundle = self._camera.get_synced_bundle()
                if bundle is None:
                    continue

                frame_start = get_timestamp_ms()
                debug_frame = self._process_frame(bundle)
                frame_duration = get_timestamp_ms() - frame_start

                self._frame_times.append(frame_duration)
                self._frame_times = self._frame_times[-30:]

                if frame_duration > MAX_FRAME_TIME_MS:
                    self._slow_frames += 1

                if SHOW_DEBUG_WINDOW and debug_frame is not None:
                    cv2.imshow("ECHORA — Debug", debug_frame)
                    key = cv2.waitKey(1)
                    if key == ord('q') or key == ord('Q'):
                        logger.info("Q key pressed — exiting.")
                        self._running = False

                self._frame_count += 1
                if self._frame_count % PERF_LOG_EVERY_N_FRAMES == 0:
                    self._log_performance()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — exiting.")

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)

        finally:
            if SHOW_DEBUG_WINDOW:
                cv2.destroyAllWindows()
            self.shutdown()


    # =========================================================================
    # FRAME PROCESSING
    # =========================================================================

    def _process_frame(self, bundle: Dict) -> Optional[np.ndarray]:

        try:
            rgb_frame = bundle["rgb"]
            depth_map = bundle["depth"]

            # ── Step 1: YOLO — runs EVERY frame ───────────────────────────────
            # With FP16 + CUDA this takes ~2-5ms. Always current.
            obstacle_result = self._detector.update(bundle)

            # ── Step 2: Rate-limited perception signals ────────────────────────
            # Signals that change slowly — computed on a schedule, not every
            # frame. Cached values used on all other frames.

            # ── OCR — background thread, NEVER blocks main loop ────────────────
            # EasyOCR takes 30-136ms even with GPU. Running it in the main
            # loop would halve our FPS. Background thread fires every 15 frames.
            # The thread updates _last_ocr_dist when done.
            # _ocr_running ensures only one OCR thread runs at a time.
            if self._frame_count % 15 == 0 and not self._ocr_running:
                self._ocr_running = True
                rgb_copy   = rgb_frame.copy()
                depth_copy = depth_map.copy()

                def _ocr_worker(f=rgb_copy, d=depth_copy):
                    self._last_ocr_dist = ocr.get_text_distance(f, d)
                    self._ocr_running   = False

                threading.Thread(
                    target = _ocr_worker,
                    daemon = True
                ).start()

            ocr_dist = self._last_ocr_dist

            # ── Face detection — every 5 frames ───────────────────────────────
            if self._frame_count % 5 == 0:
                self._last_face_conf = face_recognition.detect_face(rgb_frame)
            face_conf = self._last_face_conf

            # ── Banknote detection — every 5 frames ───────────────────────────
            if self._frame_count % 5 == 0:
                self._last_note_visible = banknote.detect_banknote(rgb_frame)
            note_visible = self._last_note_visible

            # ── Interactable scan — every 5 frames ────────────────────────────
            if self._frame_count % 5 == 0:
                current_tracks = obstacle_result.get("tracks", [])
                self._last_interact_dist = (
                    self._interaction_detector.scan_for_interactables(
                        detections = current_tracks,
                        depth_map  = depth_map
                    )
                )
            interact_dist = self._last_interact_dist

            # ── Step 3: State machine update ───────────────────────────────────
            current_mode = self._state_machine.update(
                bundle                = bundle,
                obstacle_result       = obstacle_result,
                ocr_text_distance     = ocr_dist,
                face_confidence       = face_conf,
                interactable_distance = interact_dist,
                banknote_visible      = note_visible,
            )

            # ── Step 4: Log mode changes ───────────────────────────────────────
            if current_mode != self._last_mode:
                logger.info(f"Mode: {self._last_mode} → {current_mode}")
                self._last_mode = current_mode

            # ── Step 5: Run mode handler ───────────────────────────────────────
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

            # ── Step 6: Build debug frame ──────────────────────────────────────
            debug_frame = rgb_frame.copy()

            if obstacle_result["tracks"]:
                debug_frame = draw_overlay(
                    debug_frame,
                    obstacle_result["tracks"]
                )

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
        text = ocr.read_text(rgb_frame)

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

        phase  = result.get("phase")
        target = result.get("target")

        if target:
            logger.debug(
                f"Interaction phase={phase} | "
                f"target={target.get('label')} at "
                f"{target.get('distance_mm', 0):.0f}mm"
            )

        if result.get("on_target"):
            self._audio.speak(
                "Object reached.", priority=SpeechPriority.HIGH
            )
            logger.info("Interaction SUCCESS.")
            self._state_machine.force_mode(
                MODE.NAVIGATION, reason="object reached"
            )


    def _handle_face_id(self, bundle: Dict):

        rgb_frame     = bundle["rgb"]
        name, details = face_recognition.identify_face(rgb_frame)

        if name and name != self._last_face_name:
            self._last_face_name = name
            self._audio.announce_face(name, details)
            logger.info(f"Face identified: {name}")

        elif not name and self._last_face_name == "":
            self._last_face_name = "unknown"
            self._audio.speak(
                "Unknown person.", priority=SpeechPriority.NORMAL
            )


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

        cv2.rectangle(frame, (0, 0), (280, 36), (0, 0, 0), -1)
        cv2.putText(
            frame, f"MODE: {current_mode}",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, mode_colour, 2, cv2.LINE_AA
        )

        # FPS — top right
        if self._frame_times:
            avg_ms = sum(self._frame_times) / len(self._frame_times)
            fps    = 1000.0 / max(avg_ms, 1)
        else:
            fps = 0.0

        fps_text = f"FPS: {fps:.1f}  F:{self._frame_count}"
        (text_w, _), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        cv2.rectangle(
            frame, (w - text_w - 16, 0), (w, 28), (0, 0, 0), -1
        )
        cv2.putText(
            frame, fps_text,
            (w - text_w - 8, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (200, 200, 200), 1, cv2.LINE_AA
        )

        # Danger / warning counts — bottom right
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

        # State machine stats — bottom left
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

        # Most urgent track — center top
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
                frame,
                (x_pos - 6, 38), (x_pos + tw + 6, 64),
                (0, 0, 0), -1
            )
            cv2.putText(
                frame, urgent_text,
                (x_pos, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, urgency_colour, 1, cv2.LINE_AA
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

        sm_stats      = self._state_machine.get_stats()
        tracker_stats = self._detector.get_stats()["tracker"]

        logger.info(
            f"Performance | Frame {self._frame_count} | "
            f"FPS: {fps:.1f} | Avg: {avg_ms:.1f}ms | "
            f"Slow: {self._slow_frames} | Uptime: {uptime:.0f}s | "
            f"Mode: {sm_stats['current_mode']} | "
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

        uptime = time.time() - self._start_time
        logger.info(
            f"ECHORA stopped. Frames: {self._frame_count} | "
            f"Uptime: {uptime:.1f}s | Slow frames: {self._slow_frames}"
        )
        # ── Close database ─────────────────────────────────────────────────────────
        from database import get_db
        db = get_db()
        if db:
            db.close()


        self._started = False
        logger.info("Shutdown complete.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  ECHORA — AI-Powered Sensory Substitution System")
    print("  Press Q in the debug window to quit.")
    print("=" * 60)

    cu = ControlUnit()

    try:
        cu.startup()
        cu.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        cu.shutdown()