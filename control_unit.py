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
# IMPORTS
# =============================================================================

# cv2 for the debug window — showing bounding boxes and system state.
import cv2

# numpy for frame operations in the debug overlay.
import numpy as np

# time for measuring frame duration and performance monitoring.
import time

# Type hints.
from typing import Optional, Dict

# ── Our completed modules ─────────────────────────────────────────────────────
from config import (
    MODE,
    CAMERA_RGB_WIDTH,
    CAMERA_RGB_HEIGHT,
    MAX_FRAME_TIME_MS,
    LOG_LEVEL,
)
from utils import logger, get_timestamp_ms, draw_overlay
from camera import EchoraCamera
from obstacle_detection import ObstacleDetector
from state_machine import StateMachine
from audio_feedback import AudioFeedback, SpeechPriority

# ── Stub modules (real implementations coming later) ──────────────────────────
# These imports work because we just created the stub files above.
# When the real modules are built, we just fill in the stubs —
# control_unit.py doesn't need to change at all.
import ocr
from interaction_detection import InteractionDetector
import banknote
import face_recognition


# =============================================================================
# DEBUG CONFIGURATION
# =============================================================================

# Whether to show the debug window with bounding boxes and system state.
# Set to False for the final wearable — no screen needed in production.
SHOW_DEBUG_WINDOW = True

# Whether to print a performance summary every N frames.
PERF_LOG_EVERY_N_FRAMES = 30


# =============================================================================
# CONTROL UNIT CLASS
# =============================================================================

class ControlUnit:
    """
    The central brain of ECHORA.

    Creates and manages all sub-modules.
    Runs the main perception-decision-feedback loop.
    Routes outputs from perception modules to feedback modules.

    Usage:
        cu = ControlUnit()
        cu.startup()
        cu.run()       ← blocks until user presses Q or error occurs
        cu.shutdown()  ← called automatically after run() exits
    """

    def __init__(self):
        """
        Creates all sub-module objects.

        Does NOT start any hardware or load any models yet.
        Call startup() to do that.

        We create objects here but start them in startup() so that
        if object creation fails, we get a clean error before any
        hardware is touched.
        """

        # ── Sub-modules ────────────────────────────────────────────────────────
        # The camera — reads RGB, depth, and IMU every frame.
        self._camera: Optional[EchoraCamera] = None

        # The obstacle detector — runs YOLO + Kalman tracker + VLM.
        self._detector: Optional[ObstacleDetector] = None

        # The state machine — manages mode switching.
        self._state_machine: Optional[StateMachine] = None

        # The audio system — speaks alerts and plays spatial sounds.
        self._audio: Optional[AudioFeedback] = None

        # ── State tracking ─────────────────────────────────────────────────────
        # Whether the system has been successfully started.
        self._started: bool = False

        # Whether the main loop should keep running.
        # Set to False to exit cleanly from anywhere.
        self._running: bool = False

        # Frame counter — total frames processed since startup.
        self._frame_count: int = 0

        # ── Performance monitoring ─────────────────────────────────────────────
        # List of the last 30 frame durations in milliseconds.
        # Used to calculate rolling average FPS.
        self._frame_times: list = []

        # How many frames exceeded MAX_FRAME_TIME_MS (too slow).
        self._slow_frames: int = 0

        # Timestamp of when the system started — for total uptime calculation.
        self._start_time: float = 0.0

        # ── Mode-specific state ────────────────────────────────────────────────
        # Tracks the last mode we were in — used to detect mode changes
        # and trigger announcements when the mode switches.
        self._last_mode: str = MODE.NAVIGATION

        # Tracks whether we already announced the current scene description.
        # Prevents re-announcing the same VLM description every frame.
        self._last_scene_desc: str = ""

        # Tracks the last OCR text we spoke — prevents re-reading the same text.
        self._last_ocr_text: str = ""

        # Tracks the last face we identified — prevents re-announcing same person.
        self._last_face_name: str = ""

        # Tracks the last banknote denomination — prevents re-announcing.
        self._last_denomination: str = ""
        self._interaction_detector: Optional[InteractionDetector] = None

        logger.info("ControlUnit created. Call startup() to begin.")

    def _on_enter_interaction(self):
        """
        Called by state machine when entering INTERACTION mode.
        Announces the target object name so the user knows what to reach for.
        """
        # Get the cached target from the interaction detector.
        target = self._interaction_detector._target_object

        if target:
            label = target.get("label", "object")
            dist = target.get("distance_mm", 0)
            dist_str = __import__('utils').mm_to_spoken(dist)

            # Tell the user what object was found and how far it is.
            self._audio.speak(
                f"{label} detected. {dist_str}. Raise your hand to reach it.",
                priority=SpeechPriority.HIGH
            )
            logger.info(f"INTERACTION mode entered. Target: {label} at {dist:.0f}mm")
        else:
            self._audio.announce_mode_change(MODE.INTERACTION)

    # =========================================================================
    # STARTUP
    # =========================================================================


    def startup(self):
        """
        Starts all sub-modules in the correct dependency order.

        Order matters:
          1. Camera must start first — everything else needs frames.
          2. Detector needs camera to be running before it can process frames.
          3. Audio can start independently — doesn't need camera.
          4. State machine is pure logic — no hardware dependency.
          5. Callbacks registered last — after all modules exist.

        Raises an exception if any critical module fails to start.
        """

        logger.info("=" * 60)
        logger.info("ECHORA Control Unit starting up...")
        logger.info("=" * 60)

        self._start_time = time.time()

        # ── Step 1: Start camera ───────────────────────────────────────────────
        logger.info("Step 1/5: Starting camera...")
        self._camera = EchoraCamera()
        self._camera.init_pipeline()
        logger.info("Camera ready.")

        # ── Step 2: Start obstacle detector ───────────────────────────────────
        logger.info("Step 2/5: Loading obstacle detector...")
        self._detector = ObstacleDetector()
        self._detector.load_model()
        logger.info("Obstacle detector ready.")

        logger.info("Step 2b: Loading interaction detector...")
        self._interaction_detector = InteractionDetector()
        self._interaction_detector.load_model()

        # ── Step 2c: Initialise OCR ────────────────────────────────────────────────
        logger.info("Step 2c: Initialising OCR...")
        import ocr as ocr_module
        ocr_module.init_ocr()
        logger.info("OCR ready.")

        # ── Step 2d: Initialise banknote detector ──────────────────────────────────
        logger.info("Step 2d: Initialising banknote detector...")
        import banknote as banknote_module
        banknote_module.init_banknote()
        logger.info("Banknote detector ready.")

        # ── Step 3: Start audio system ─────────────────────────────────────────
        logger.info("Step 3/5: Starting audio system...")
        self._audio = AudioFeedback()
        self._audio.init_audio()
        logger.info("Audio ready.")

        # ── Step 4: Create state machine ───────────────────────────────────────
        logger.info("Step 4/5: Initialising state machine...")
        self._state_machine = StateMachine()
        logger.info("State machine ready.")

        # ── Step 5: Register callbacks ─────────────────────────────────────────
        logger.info("Step 5/5: Registering mode callbacks...")
        self._register_callbacks()
        logger.info("Callbacks registered.")

        # ── Announce startup ───────────────────────────────────────────────────
        # Let the user know the system is ready.
        # We use a short delay so the TTS engine has time to fully initialise.
        time.sleep(0.3)
        self._audio.speak(
            "ECHORA online. Navigation mode active.",
            priority=SpeechPriority.NORMAL
        )

        self._started = True

        logger.info("=" * 60)
        logger.info("ECHORA startup complete. Entering main loop.")
        logger.info("=" * 60)

    def _reset_interaction_state(self):
        """Resets interaction state when leaving INTERACTION mode."""
        if self._interaction_detector:
            self._interaction_detector.reset()
        logger.debug("Interaction state reset.")
    def _register_callbacks(self):
        """
        Registers on_enter and on_exit callbacks with the state machine.

        Each callback is a function that runs automatically when a mode
        is entered or exited. This keeps mode-change logic in one place.

        We define the callback functions as small lambdas or direct
        method references. Lambda is a way to create a tiny one-line
        function inline: lambda: expression
        """

        sm = self._state_machine

        # ── NAVIGATION callbacks ───────────────────────────────────────────────

        # When entering NAVIGATION: announce the mode change.
        sm.register_callback(
            mode     = MODE.NAVIGATION,
            on_enter = lambda: self._audio.announce_mode_change(MODE.NAVIGATION)
        )

        # When exiting NAVIGATION: reset the obstacle tracker.
        # Stale tracked objects from navigation are irrelevant in other modes.
        sm.register_callback(
            mode    = MODE.NAVIGATION,
            on_exit = self._detector.reset_tracker
        )

        # ── OCR callbacks ──────────────────────────────────────────────────────────
        sm.register_callback(
            mode=MODE.OCR,
            on_enter=lambda: (
                self._audio.announce_mode_change(MODE.OCR),
                self._audio.stop_all()
            )
        )

        sm.register_callback(
            mode=MODE.OCR,
            on_exit=lambda: (
                self._reset_ocr_state(),
                __import__('ocr').reset_ocr()
            )
        )
        # ← nothing else for OCR — remove the duplicate _reset_ocr_state registration

        # ── INTERACTION callbacks ──────────────────────────────────────────────────

        # When entering INTERACTION: announce the target object name.
        sm.register_callback(
            mode=MODE.INTERACTION,
            on_enter=self._on_enter_interaction
        )

        # When exiting INTERACTION: reset interaction state.
        # Clears the target object and denomination history so the next
        # interaction starts fresh.
        sm.register_callback(
            mode=MODE.INTERACTION,
            on_exit=self._reset_interaction_state
        )

        # ── FACE_ID callbacks ──────────────────────────────────────────────────

        sm.register_callback(
            mode     = MODE.FACE_ID,
            on_enter = lambda: self._audio.announce_mode_change(MODE.FACE_ID)
        )

        # When exiting FACE_ID: reset face state so same person can be
        # re-identified if we enter FACE_ID mode again.
        sm.register_callback(
            mode    = MODE.FACE_ID,
            on_exit = self._reset_face_state
        )

        # ── BANKNOTE callbacks ─────────────────────────────────────────────────

        sm.register_callback(
            mode     = MODE.BANKNOTE,
            on_enter = lambda: self._audio.announce_mode_change(MODE.BANKNOTE)
        )

        sm.register_callback(
            mode=MODE.BANKNOTE,
            on_exit=lambda: (
                self._reset_banknote_state(),
                __import__('banknote').reset_banknote()
            )
        )

    # =========================================================================
    # STATE RESET HELPERS
    # =========================================================================
    # These are called by the state machine's on_exit callbacks.
    # They reset module-specific state when leaving a mode.

    def _reset_ocr_state(self):
        """Resets OCR tracking state when leaving OCR mode."""
        self._last_ocr_text = ""
        logger.debug("OCR state reset.")

    def _reset_face_state(self):
        """Resets face recognition state when leaving FACE_ID mode."""
        self._last_face_name = ""
        logger.debug("Face state reset.")

    def _reset_banknote_state(self):
        """Resets banknote state when leaving BANKNOTE mode."""
        self._last_denomination = ""
        logger.debug("Banknote state reset.")


    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self):
        """
        The main loop — runs ECHORA until the user presses Q.

        This function blocks — it does not return until the user exits.
        After it returns, call shutdown() for clean cleanup.

        The loop structure:
          1. Get camera bundle
          2. Process the frame
          3. Show debug window
          4. Check for quit key
          5. Repeat
        """

        if not self._started:
            logger.error("Cannot run — call startup() first.")
            return

        self._running = True
        logger.info("Main loop started. Press Q in the debug window to quit.")

        try:
            while self._running:

                # ── Get camera bundle ──────────────────────────────────────────
                bundle = self._camera.get_synced_bundle()

                # If no frame is ready yet, skip this iteration.
                # This is normal — the camera might not have a new frame ready.
                if bundle is None:
                    continue

                # ── Process the frame ──────────────────────────────────────────
                # Record the start time for performance measurement.
                frame_start = get_timestamp_ms()

                # Core processing — all the logic happens here.
                debug_frame = self._process_frame(bundle)

                # Record frame duration.
                frame_duration = get_timestamp_ms() - frame_start
                self._frame_times.append(frame_duration)

                # Keep only the last 30 frame times for rolling average.
                # [-30:] is a Python slice that keeps the last 30 items.
                self._frame_times = self._frame_times[-30:]

                # Count slow frames.
                if frame_duration > MAX_FRAME_TIME_MS:
                    self._slow_frames += 1

                # ── Show debug window ──────────────────────────────────────────
                if SHOW_DEBUG_WINDOW and debug_frame is not None:
                    cv2.imshow("ECHORA — Debug", debug_frame)

                    # cv2.waitKey(1) waits 1ms for a keypress.
                    # Returns the ASCII value of the key pressed, or -1 if none.
                    # ord('q') converts the character 'q' to its ASCII value (113).
                    # Pressing Q exits the main loop.
                    key = cv2.waitKey(1)
                    if key == ord('q') or key == ord('Q'):
                        logger.info("Q key pressed — exiting main loop.")
                        self._running = False

                # ── Log performance ────────────────────────────────────────────
                self._frame_count += 1
                if self._frame_count % PERF_LOG_EVERY_N_FRAMES == 0:
                    self._log_performance()

        except KeyboardInterrupt:
            # Ctrl+C in terminal — exit cleanly.
            logger.info("Keyboard interrupt — exiting.")

        except Exception as e:
            # Any unexpected error — log it and exit.
            logger.error(f"Main loop error: {e}", exc_info=True)

        finally:
            # Always runs whether we exited normally, via Q, Ctrl+C, or error.
            # 'finally' guarantees cleanup even if an exception occurred.
            if SHOW_DEBUG_WINDOW:
                cv2.destroyAllWindows()

            self.shutdown()


    # =========================================================================
    # FRAME PROCESSING — THE CORE LOGIC
    # =========================================================================

    def _process_frame(self, bundle: Dict) -> Optional[np.ndarray]:
        """
        Processes one frame — the heart of ECHORA's logic.

        Called 30 times per second by run().

        Steps:
          1. Run obstacle detection
          2. Gather perception signals for the state machine
          3. Update the state machine → get current mode
          4. Handle the current mode (navigation, OCR, etc.)
          5. Build and return the debug frame

        Arguments:
            bundle: camera bundle from EchoraCamera.get_synced_bundle()

        Returns:
            Debug frame (numpy array) for display, or None on error.
        """

        try:
            rgb_frame = bundle["rgb"]
            depth_map = bundle["depth"]

            # ── Step 1: Run obstacle detection ────────────────────────────────────────────
            # YOLO runs every frame — gives us all confirmed tracks.
            obstacle_result = self._detector.update(bundle)

            # ── Step 2: Gather perception signals ─────────────────────────────────────────
            ocr_dist = ocr.get_text_distance(rgb_frame, depth_map)
            face_conf = face_recognition.detect_face(rgb_frame)
            note_visible = banknote.detect_banknote(rgb_frame)

            # Run lightweight interactable scan every frame — feeds the state machine.
            # This is separate from full interaction mode (which includes hand tracking).
            # It runs regardless of current mode so the state machine always has a
            # fresh distance value to decide whether to switch to INTERACTION mode.
            current_tracks = obstacle_result.get("tracks", [])
            interact_dist = self._interaction_detector.scan_for_interactables(
                detections=current_tracks,
                depth_map=depth_map
            )

            # ── Step 3: Update state machine ──────────────────────────────────
            # Pass all perception signals to the state machine.
            # It evaluates switching rules and returns the current mode.
            current_mode = self._state_machine.update(
                bundle               = bundle,
                obstacle_result      = obstacle_result,
                ocr_text_distance    = ocr_dist,
                face_confidence      = face_conf,
                interactable_distance = interact_dist,
                banknote_visible     = note_visible,
            )

            # ── Step 4: Detect mode changes ───────────────────────────────────
            # If the mode changed since last frame, log it.
            # The actual announcements are handled by state machine callbacks.
            if current_mode != self._last_mode:
                logger.info(
                    f"Active mode: {self._last_mode} → {current_mode}"
                )
                self._last_mode = current_mode

            # ── Step 5: Handle the current mode ───────────────────────────────
            # Delegate to the appropriate mode handler.
            # Each handler runs the logic specific to that mode.

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
            # Make a copy of the RGB frame so we can draw on it without
            # modifying the original (which other modules might still use).
            debug_frame = rgb_frame.copy()

            # Draw bounding boxes and labels on the debug frame.
            if obstacle_result["tracks"]:
                debug_frame = draw_overlay(
                    debug_frame,
                    obstacle_result["tracks"]
                )

                # Draw interaction overlay when in INTERACTION mode.
                if current_mode == MODE.INTERACTION:
                    last_result = getattr(self._interaction_detector, '_last_grid', None)
                    if last_result is not None:
                        # Build a minimal result dict for the overlay.
                        interaction_result = {
                            "phase": self._interaction_detector._phase,
                            "hand": None,
                            "target": self._interaction_detector._target_object,
                            "electrode_grid": self._interaction_detector._last_grid,
                        }
                        debug_frame = self._interaction_detector.draw_debug_overlay(
                            debug_frame, interaction_result
                        )

            # Draw the system status overlay (mode, FPS, state machine info).
            debug_frame = self._draw_debug_overlay(
                debug_frame,
                obstacle_result,
                current_mode
            )

            return debug_frame

        except Exception as e:
            logger.error(f"Frame processing error: {e}", exc_info=True)
            # Return the raw frame without any overlay if processing failed.
            return bundle.get("rgb")


    # =========================================================================
    # MODE HANDLERS
    # =========================================================================

    def _handle_navigation(self, bundle: Dict, obstacle_result: Dict):
        """
        Handles all logic for NAVIGATION mode.

        This is the most important mode — runs every frame when the user
        is walking and needs obstacle awareness.

        Actions:
          1. Announce DANGER obstacles with spatial audio + speech
          2. Play spatial audio for WARNING obstacles (no speech unless new)
          3. Speak new VLM scene descriptions
        """

        danger_tracks  = obstacle_result.get("danger",  [])
        warning_tracks = obstacle_result.get("warning", [])
        scene_desc     = obstacle_result.get("scene_desc", "")

        # ── Handle DANGER tracks ───────────────────────────────────────────────
        # announce_obstacle() handles both spatial audio AND speech.
        # The AlertCooldown inside audio_feedback prevents repetition.
        for track in danger_tracks:
            self._audio.announce_obstacle(track)

        # ── Handle WARNING tracks ──────────────────────────────────────────────
        # For WARNING tracks, we announce them too but at lower priority.
        # The cooldown in announce_obstacle() prevents flooding.
        for track in warning_tracks:
            self._audio.announce_obstacle(track)

        # ── Handle VLM scene description ───────────────────────────────────────
        # Only speak the description if it's NEW (different from last time).
        # We don't want to repeat the same description every frame.
        if (scene_desc
                and scene_desc != self._last_scene_desc
                and len(scene_desc) > 10):

            # Update our record of the last description we spoke.
            self._last_scene_desc = scene_desc

            # Speak it at NORMAL priority — informational, not urgent.
            self._audio.announce_scene(scene_desc)
            logger.info(f"New scene description: {scene_desc[:60]}...")


    def _handle_ocr(self, bundle: Dict):
        """
        Handles all logic for OCR mode.

        The user is looking at text. We read it and speak it.

        Actions:
          1. Run OCR on the current frame
          2. If new text found: speak it
          3. If same text as before: don't repeat
        """

        rgb_frame = bundle["rgb"]

        # Run the OCR module on the current frame.
        # This is a stub for now — returns "" until ocr.py is implemented.
        text = ocr.read_text(rgb_frame)

        if not text or not text.strip():
            # No text found this frame — nothing to do.
            return

        # Only speak if the text is different from last time.
        # This prevents re-reading the same sign every frame while it's visible.
        if text.strip() != self._last_ocr_text:
            self._last_ocr_text = text.strip()
            self._audio.announce_ocr(text)
            logger.info(f"OCR text: '{text[:60]}'")

        # Keep spatial danger audio active even in OCR mode — safety first.
        # If a DANGER obstacle appears while reading text, still beep.
        # Note: we don't get obstacle_result here because OCR mode doesn't
        # run the full detector. The state machine's emergency override
        # handles the case where a DANGER appears — it forces NAVIGATION.

    def _handle_interaction(self, bundle: Dict, obstacle_result: Dict):
            """
            Handles all logic for INTERACTION mode.

            Uses the real InteractionDetector — hand guidance + edge rendering.
            The InteractionDetector internally sends patterns to HapticBridge.
            """

            rgb_frame = bundle["rgb"]
            depth_map = bundle["depth"]

            # Get current obstacle detections — we pass them to the interaction
            # detector so it can filter for interactable classes internally.
            current_tracks = obstacle_result.get("tracks", [])

            # Run the full interaction update — hand detection + guidance + edge rendering.
            # This also sends the electrode grid to HapticBridge automatically.
            result = self._interaction_detector.update(
                rgb_frame=rgb_frame,
                depth_map=depth_map,
                detections=current_tracks
            )

            phase = result.get("phase")
            target = result.get("target")

            # Log what is happening.
            if target:
                logger.debug(
                    f"Interaction phase={phase} | "
                    f"target={target.get('label')} at {target.get('distance_mm'):.0f}mm"
                )

            # If the user successfully touched the object — announce it and return to NAV.
            if result.get("on_target"):
                self._audio.speak(
                    f"Object reached.",
                    priority=SpeechPriority.HIGH
                )
                logger.info("Interaction SUCCESS — object reached.")
                # Force return to navigation after successful interaction.
                self._state_machine.force_mode(MODE.NAVIGATION, reason="object reached")

            # TODO: send haptic guidance pattern when haptic_feedback.py is ready
            # haptic.guide_hand(nearest, hands)


    def _handle_face_id(self, bundle: Dict):
        """
        Handles all logic for FACE_ID mode.

        The user is looking at a person. We identify them if possible.

        Actions:
          1. Run face identification
          2. If person identified and new: speak their name
          3. If unknown: announce "unknown person"
        """

        rgb_frame = bundle["rgb"]

        # Run face identification using the stub.
        # Returns (name, details) — both empty strings if not recognised.
        name, details = face_recognition.identify_face(rgb_frame)

        if name and name != self._last_face_name:
            # New person identified — announce them.
            self._last_face_name = name
            self._audio.announce_face(name, details)
            logger.info(f"Face identified: {name}")

        elif not name and self._last_face_name == "":
            # Face detected but not in our database.
            # Only announce "unknown" once per mode entry.
            self._last_face_name = "unknown"
            self._audio.speak(
                "Unknown person.",
                priority=SpeechPriority.NORMAL
            )


    def _handle_banknote(self, bundle: Dict):
        """
        Handles all logic for BANKNOTE mode.

        The user is holding up a banknote. We identify the denomination.

        Actions:
          1. Run banknote classification
          2. If denomination identified and new: speak it
        """

        rgb_frame = bundle["rgb"]

        # Run banknote classification using the stub.
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
        frame: np.ndarray,
        obstacle_result: Dict,
        current_mode: str
    ) -> np.ndarray:
        """
        Draws the developer debug overlay on the frame.

        Shows:
          - Current mode (top left, colour-coded)
          - FPS and frame count (top right)
          - State machine stats (bottom left)
          - Danger/warning counts (bottom right)

        This overlay is only visible during development.
        In the final wearable, SHOW_DEBUG_WINDOW is False so this
        function is never called.

        Arguments:
            frame:           the RGB frame to draw on (already has bboxes)
            obstacle_result: the result dict from obstacle_detection.py
            current_mode:    the current STATE_MACHINE mode string

        Returns:
            The frame with overlay drawn on it.
        """

        h, w = frame.shape[:2]

        # ── Mode indicator — top left ──────────────────────────────────────────
        # Each mode gets its own colour for quick visual identification.
        mode_colours = {
            MODE.NAVIGATION:  (0, 200, 80),    # green
            MODE.OCR:         (255, 165, 0),   # orange
            MODE.INTERACTION: (0, 165, 255),   # blue
            MODE.FACE_ID:     (180, 0, 255),   # purple
            MODE.BANKNOTE:    (0, 215, 255),   # gold
        }
        mode_colour = mode_colours.get(current_mode, (200, 200, 200))

        # Draw a filled rectangle behind the mode text for readability.
        # cv2.rectangle(frame, top-left, bottom-right, colour, -1)
        # -1 as thickness means fill the rectangle.
        cv2.rectangle(frame, (0, 0), (280, 36), (0, 0, 0), -1)

        # Draw the mode name text.
        cv2.putText(
            frame,
            f"MODE: {current_mode}",
            (8, 24),                      # position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,                           # font scale
            mode_colour,
            2,                             # thickness
            cv2.LINE_AA                    # anti-aliased for smooth text
        )

        # ── FPS — top right ────────────────────────────────────────────────────
        # Calculate rolling average FPS from the last 30 frame times.
        if self._frame_times:
            # Average frame time in ms.
            avg_frame_ms = sum(self._frame_times) / len(self._frame_times)

            # FPS = 1000ms / average frame time.
            # Avoid division by zero with max(..., 1).
            fps = 1000.0 / max(avg_frame_ms, 1)
        else:
            fps = 0.0

        fps_text = f"FPS: {fps:.1f}  F:{self._frame_count}"

        # Calculate text width so we can right-align it.
        # cv2.getTextSize returns ((width, height), baseline).
        (text_w, _), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )

        # Draw background rectangle.
        cv2.rectangle(frame, (w - text_w - 16, 0), (w, 28), (0, 0, 0), -1)

        # Draw FPS text.
        cv2.putText(
            frame,
            fps_text,
            (w - text_w - 8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),   # light grey
            1,
            cv2.LINE_AA
        )

        # ── Danger / warning counts — bottom right ─────────────────────────────
        n_danger  = len(obstacle_result.get("danger",  []))
        n_warning = len(obstacle_result.get("warning", []))
        n_tracks  = len(obstacle_result.get("tracks",  []))

        count_lines = [
            f"Tracks:  {n_tracks}",
            f"Danger:  {n_danger}",
            f"Warning: {n_warning}",
        ]

        # Draw each line from the bottom upward.
        # enumerate gives us (index, value) pairs.
        for i, line in enumerate(reversed(count_lines)):

            # Choose colour based on content.
            if "Danger" in line and n_danger > 0:
                colour = (0, 0, 220)      # red
            elif "Warning" in line and n_warning > 0:
                colour = (0, 165, 255)    # orange
            else:
                colour = (180, 180, 180)  # grey

            # y position counts up from the bottom.
            # i=0 is the bottom line, i=1 is one line above, etc.
            y_pos = h - 10 - (i * 22)

            cv2.putText(
                frame, line,
                (w - 140, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, colour, 1, cv2.LINE_AA
            )

        # ── State machine stats — bottom left ──────────────────────────────────
        sm_stats    = self._state_machine.get_stats()
        motion_str  = f"Motion: {sm_stats['motion_level']:.2f} m/s2"
        stable_str  = f"Stable: {'yes' if sm_stats['is_stable'] else 'no'}"
        dur_str     = f"In mode: {sm_stats['mode_duration_s']:.1f}s"

        for i, line in enumerate([motion_str, stable_str, dur_str]):
            y_pos = h - 10 - (i * 22)
            cv2.putText(
                frame, line,
                (8, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (160, 160, 160), 1, cv2.LINE_AA
            )

        # ── Most urgent track info — center top ────────────────────────────────
        most_urgent = self._detector.get_most_urgent_obstacle()
        if most_urgent:
            urgent_text = (
                f"{most_urgent['label']}  "
                f"{most_urgent['distance_mm']:.0f}mm  "
                f"{most_urgent['angle_deg']:+.0f}deg"
            )

            # Colour based on urgency.
            urgency_colour = {
                "DANGER":  (0, 0, 220),
                "WARNING": (0, 165, 255),
                "SAFE":    (0, 200, 80),
            }.get(most_urgent["urgency"], (180, 180, 180))

            # Centre the text horizontally.
            (tw, _), _ = cv2.getTextSize(
                urgent_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            x_pos = (w - tw) // 2

            # Background strip.
            cv2.rectangle(frame, (x_pos - 6, 38), (x_pos + tw + 6, 64), (0,0,0), -1)

            cv2.putText(
                frame, urgent_text,
                (x_pos, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, urgency_colour, 1, cv2.LINE_AA
            )

        return frame


    # =========================================================================
    # PERFORMANCE MONITORING
    # =========================================================================

    def _log_performance(self):
        """
        Logs a performance summary every PERF_LOG_EVERY_N_FRAMES frames.

        Shows FPS, average frame time, slow frame count, and uptime.
        This helps identify performance bottlenecks during development.
        """

        if not self._frame_times:
            return

        # Calculate average frame time and FPS.
        avg_ms = sum(self._frame_times) / len(self._frame_times)
        fps    = 1000.0 / max(avg_ms, 1)

        # Total uptime in seconds.
        uptime = time.time() - self._start_time

        # Get state machine and tracker stats.
        sm_stats      = self._state_machine.get_stats()
        tracker_stats = self._detector.get_stats()["tracker"]

        logger.info(
            f"Performance | Frame {self._frame_count} | "
            f"FPS: {fps:.1f} | "
            f"Avg: {avg_ms:.1f}ms | "
            f"Slow frames: {self._slow_frames} | "
            f"Uptime: {uptime:.0f}s | "
            f"Mode: {sm_stats['current_mode']} | "
            f"Tracks: {tracker_stats['confirmed']}"
        )


    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def shutdown(self):
        """
        Cleanly shuts down all sub-modules.

        Called automatically by run() when the main loop exits.
        Also safe to call manually.

        Order: audio first (so user hears the goodbye), then camera,
        detector can just be garbage collected.
        """

        if not self._started:
            return

        logger.info("Shutting down ECHORA...")

        # ── Announce shutdown ──────────────────────────────────────────────────
        if self._audio and self._audio._ready:
            self._audio.speak("ECHORA shutting down.", priority=SpeechPriority.HIGH)
            # Give Karen time to say goodbye before we kill the audio thread.
            time.sleep(2.0)

        # ── Release audio ──────────────────────────────────────────────────────
        if self._audio:
            self._audio.release()

        # ── Release interaction detector ───────────────────────────────────────────
        if self._interaction_detector:
            self._interaction_detector.release()

        # ── Release camera ─────────────────────────────────────────────────────
        if self._camera:
            self._camera.release()

        # ── Final stats ────────────────────────────────────────────────────────
        uptime = time.time() - self._start_time
        logger.info(
            f"ECHORA stopped. "
            f"Frames: {self._frame_count} | "
            f"Uptime: {uptime:.1f}s | "
            f"Slow frames: {self._slow_frames}"
        )

        self._started = False
        logger.info("Shutdown complete.")


# =============================================================================
# ENTRY POINT — RUN ECHORA
# =============================================================================
# This is how you start the full ECHORA system:
#   python control_unit.py
#
# This will:
#   1. Start the OAK-D camera
#   2. Load the YOLO model
#   3. Start the audio system (Karen will say "ECHORA online")
#   4. Open the debug window
#   5. Begin real-time obstacle detection and audio alerts
#
# Press Q in the debug window to exit cleanly.

if __name__ == "__main__":

    print("=" * 60)
    print("  ECHORA — AI-Powered Sensory Substitution System")
    print("  Press Q in the debug window to quit.")
    print("=" * 60)

    # Create the control unit.
    cu = ControlUnit()

    try:
        # Start everything.
        cu.startup()

        # Run the main loop — blocks until Q is pressed.
        cu.run()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

    finally:
        # Ensure clean shutdown even if startup() failed halfway.
        # shutdown() checks self._started so it is safe to call even
        # if startup() never completed.
        cu.shutdown()
'''

---

Make sure your folder looks like this before running:

echora/
├── config.py
├── utils.py
├── camera.py
├── kalman_tracker.py
├── obstacle_detection.py
├── state_machine.py
├── audio_feedback.py
├── control_unit.py        ← new
├── ocr.py                 ← new stub
├── interaction_detection.py  ← new stub
├── banknote.py            ← new stub
├── face_recognition.py    ← new stub
└── models/
    └── yolov8n.pt


'''