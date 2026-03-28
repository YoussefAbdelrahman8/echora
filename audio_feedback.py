# =============================================================================
# audio_feedback.py — ECHORA Audio Output System
# =============================================================================
# Two jobs:
#   Job 1 — Text to Speech (TTS): speaks obstacle names, distances,
#            OCR text, face names, banknote denominations.
#   Job 2 — 3D Spatial Audio: plays directional warning beeps panned
#            left/right based on the real-world angle of the obstacle.
#
# Completely non-blocking — all speech runs in a background thread.
# The main loop never waits for audio to finish.
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

# pyttsx3 is the text-to-speech library.
# It uses the OS built-in TTS engine — no internet needed.
# On Mac it uses the built-in voices (Samantha, Alex, etc.)
import pyttsx3

# pygame.mixer handles audio playback and stereo panning.
# We only use the mixer part of pygame, not the full game library.
import pygame
import pygame.mixer
import pygame.sndarray

# numpy for generating tones mathematically when no sound files exist.
import numpy as np

# queue.PriorityQueue — a thread-safe queue that serves highest-priority
# items first. Used to manage multiple speech requests.
import queue

# threading for the background speech worker thread.
import threading

# time for timestamps and rate limiting.
import time

# dataclass for clean SpeechRequest objects.
from dataclasses import dataclass, field

# Type hints.
from typing import Optional, Dict

# Our config and utils.
from config import (
    TTS_RATE,
    TTS_VOLUME,
    ALERT_VOLUME,
    CAMERA_HFOV_DEG,
    SOUND_DANGER_PATH,
    SOUND_WARNING_PATH,
    SOUND_CHIME_PATH,
    ALERT_COOLDOWN_SEC,
)
from utils import logger, mm_to_spoken, get_timestamp_ms, AlertCooldown


# =============================================================================
# PRIORITY LEVELS
# =============================================================================
# These integers define the priority of speech requests.
# LOWER number = higher priority = spoken first.
# We use negative numbers so that higher priority = more negative = first out.

class SpeechPriority:
    """
    Priority levels for the speech queue.

    URGENT = most important, spoken immediately even if others are waiting.
    LOW    = least important, only spoken when nothing else is queued.
    """

    # DANGER obstacle — must be spoken immediately.
    # User is about to collide with something.
    URGENT = -3

    # WARNING obstacle or important event.
    # User should know soon but not a collision risk yet.
    HIGH   = -2

    # Normal information — scene descriptions, OCR text, face names.
    NORMAL = -1

    # Low-priority informational messages.
    # Spoken only when nothing else is waiting.
    LOW    = 0


# =============================================================================
# SPEECH REQUEST DATACLASS
# =============================================================================

@dataclass(order=True)
class SpeechRequest:
    """
    Represents one item in the speech priority queue.

    @dataclass(order=True) automatically generates comparison methods
    (__lt__, __gt__, etc.) based on the fields in order.
    PriorityQueue uses these comparisons to sort items.

    The 'sort_index' field is compared first — lower sort_index = higher priority.
    'field(compare=False)' means that field is NOT used in comparisons,
    which prevents errors when comparing non-comparable types like dicts.
    """

    # The primary sort key. PriorityQueue uses this to order items.
    # Lower value = higher priority = dequeued first.
    # We set this to the priority level (e.g. -3 for URGENT).
    sort_index: int

    # The text to speak. compare=False means this field is ignored
    # when two SpeechRequests are compared for ordering.
    text: str = field(compare=False)

    # When this request was created — milliseconds since epoch.
    # compare=False — not used for ordering.
    timestamp_ms: float = field(compare=False, default_factory=get_timestamp_ms)

    # How long this request remains valid — seconds.
    # Requests older than this are discarded without being spoken.
    # This prevents a backlog of old alerts from playing after a delay.
    # Default: 3 seconds. URGENT alerts last 5 seconds.
    ttl_sec: float = field(compare=False, default=3.0)


# =============================================================================
# AUDIO FEEDBACK CLASS
# =============================================================================

class AudioFeedback:
    """
    Manages all audio output for ECHORA.

    Non-blocking: speak() returns immediately. Speech happens in background.
    Priority-aware: urgent alerts interrupt or skip past lower-priority speech.
    Spatial: warning sounds are panned left/right based on obstacle angle.

    Usage:
        audio = AudioFeedback()
        audio.init_audio()

        # These return immediately — speech happens in background:
        audio.announce_obstacle(track)
        audio.announce_scene("Hallway ahead, door on the left.")
        audio.speak("Hello world", priority=SpeechPriority.NORMAL)

        audio.release()  # call when shutting down
    """

    def __init__(self):
        """
        Creates the AudioFeedback object. Does NOT start audio yet.
        Call init_audio() to start the audio system.
        """

        # The pyttsx3 TTS engine object.
        # None until init_audio() runs.
        self._engine: Optional[pyttsx3.Engine] = None

        # Whether the audio system has been successfully initialised.
        self._ready: bool = False

        # The speech priority queue.
        # SpeechRequest objects are put in here from any thread.
        # The background speech worker reads from here in priority order.
        self._speech_queue: queue.PriorityQueue = queue.PriorityQueue()

        # Flag to control the background speech thread.
        # True = keep running. False = shut down.
        self._running: bool = False

        # The background speech thread object.
        # None until init_audio() starts it.
        self._speech_thread: Optional[threading.Thread] = None

        # Alert cooldown — prevents the same obstacle from being announced
        # every single frame. Uses AlertCooldown from utils.py.
        self._cooldown = AlertCooldown()

        # pygame Sound objects for each alert type.
        # None until init_audio() loads or generates them.
        self._sound_danger:  Optional[pygame.mixer.Sound] = None
        self._sound_warning: Optional[pygame.mixer.Sound] = None
        self._sound_chime:   Optional[pygame.mixer.Sound] = None

        # Master volume level 0.0 to 1.0.
        # Loaded from config.py.
        self._volume: float = TTS_VOLUME

        # Whether TTS is currently speaking.
        # Used to prevent overlapping TTS calls.
        self._is_speaking: bool = False

        # Lock for thread-safe access to _is_speaking.
        self._speaking_lock = threading.Lock()

        # Track the last time a spatial alert was played per urgency level.
        # Prevents continuous beeping every frame for the same obstacle.
        # Key = urgency string, Value = timestamp of last alert.
        self._last_spatial_alert: Dict[str, float] = {}

        # Minimum seconds between spatial alert sounds for the same urgency.
        # DANGER beeps every 0.5 seconds, WARNING every 2 seconds.
        self._spatial_cooldown = {
            "DANGER":  0.5,
            "WARNING": 2.0,
        }

        logger.info("AudioFeedback created. Call init_audio() to start.")


    # =========================================================================
    # INITIALISATION
    # =========================================================================

    def init_audio(self):
        """
        Starts the audio system.

        Steps:
          1. Initialise pygame mixer for spatial audio playback
          2. Load or generate alert sounds
          3. Initialise pyttsx3 TTS engine
          4. Configure TTS voice, rate, and volume
          5. Start the background speech worker thread
        """

        logger.info("Initialising audio system...")

        # ── Step 1: Initialise pygame mixer ───────────────────────────────────
        # pygame.mixer.init() prepares pygame's audio system.
        # frequency=22050: audio sample rate in Hz (22050 is standard).
        # size=-16: 16-bit signed audio samples (negative = signed).
        # channels=2: stereo (2 channels = left + right).
        # buffer=512: audio buffer size in samples.
        #   Smaller buffer = lower latency but more CPU usage.
        #   512 samples at 22050Hz = ~23ms latency — acceptable for real-time.
        try:
            pygame.mixer.init(
                frequency=22050,
                size=-16,
                channels=2,
                buffer=512
            )
            logger.info("pygame mixer initialised.")
        except Exception as e:
            logger.error(f"pygame mixer init failed: {e}")
            # We continue even if pygame fails — TTS will still work.

        # ── Step 2: Load or generate alert sounds ─────────────────────────────
        self._load_sounds()

        # ── Step 3: Initialise pyttsx3 TTS engine ─────────────────────────────
        try:
            # pyttsx3.init() creates the TTS engine using the OS default.
            # On Mac: uses NSSpeechSynthesizer (built-in Mac TTS).
            # On Windows: uses SAPI5.
            # On Linux: uses espeak.
            self._engine = pyttsx3.init()
            logger.info("TTS engine initialised.")
        except Exception as e:
            logger.error(f"TTS engine init failed: {e}")
            self._engine = None

        # ── Step 4: Configure TTS ──────────────────────────────────────────────
        if self._engine:
            # setProperty('rate', N) sets speech speed in words per minute.
            # TTS_RATE = 150 from config.py. 150 wpm is natural pace.
            self._engine.setProperty('rate', TTS_RATE)

            # setProperty('volume', N) sets volume 0.0 to 1.0.
            self._engine.setProperty('volume', self._volume)

            # Select a voice — try to find a clear English voice.
            # engine.getProperty('voices') returns a list of available voices.
            voices = self._engine.getProperty('voices')

            # Try to find the best voice for clarity.
            # We prefer female voices on Mac as they tend to be clearer
            # for assistive applications — Samantha, Karen, etc.
            # If none found, we use whatever the default is.
            selected_voice = None
            preferred_names = ['samantha', 'karen', 'victoria', 'alex']

            for voice in voices:
                # voice.name is the voice name string.
                # .lower() converts to lowercase for case-insensitive comparison.
                # any() returns True if the voice name contains any preferred name.
                if any(name in voice.name.lower() for name in preferred_names):
                    selected_voice = voice.id
                    logger.info(f"Selected TTS voice: {voice.name}")
                    break

            if selected_voice:
                self._engine.setProperty('voice', selected_voice)
            else:
                logger.info("Using default TTS voice.")

        # ── Step 5: Start background speech thread ─────────────────────────────
        self._running = True
        self._speech_thread = threading.Thread(
            target=self._speech_worker,
            daemon=True   # auto-killed when main program exits
        )
        self._speech_thread.start()
        logger.info("Speech worker thread started.")

        self._ready = True
        logger.info("Audio system ready.")


    def _load_sounds(self):
        """
        Loads alert sounds from files, or generates them mathematically
        if the sound files don't exist yet.

        This means ECHORA works even before you have created sound files.
        The generated tones are simple beeps — functional but not pretty.
        You can replace them later with proper recorded sounds.
        """

        # ── Try to load from files first ──────────────────────────────────────
        # SOUND_DANGER_PATH etc. are Path objects from config.py.
        # .exists() returns True if the file is there.

        if SOUND_DANGER_PATH.exists():
            # pygame.mixer.Sound() loads an audio file into memory.
            self._sound_danger = pygame.mixer.Sound(str(SOUND_DANGER_PATH))
            logger.info(f"Loaded danger sound: {SOUND_DANGER_PATH.name}")
        else:
            # Generate a 880Hz beep (high pitch = urgent)
            self._sound_danger = self._generate_tone(
                frequency=880,    # Hz — A5 note, high and attention-grabbing
                duration=0.3,     # seconds
                volume=0.9        # loud
            )
            logger.info("Generated danger tone (880Hz).")

        if SOUND_WARNING_PATH.exists():
            self._sound_warning = pygame.mixer.Sound(str(SOUND_WARNING_PATH))
            logger.info(f"Loaded warning sound: {SOUND_WARNING_PATH.name}")
        else:
            # Generate a 440Hz beep (medium pitch = caution)
            self._sound_warning = self._generate_tone(
                frequency=440,    # Hz — A4 note, standard reference pitch
                duration=0.4,
                volume=0.7
            )
            logger.info("Generated warning tone (440Hz).")

        if SOUND_CHIME_PATH.exists():
            self._sound_chime = pygame.mixer.Sound(str(SOUND_CHIME_PATH))
            logger.info(f"Loaded chime sound: {SOUND_CHIME_PATH.name}")
        else:
            # Generate a soft 523Hz chime (C5 note = informational)
            self._sound_chime = self._generate_tone(
                frequency=523,
                duration=0.2,
                volume=0.5
            )
            logger.info("Generated chime tone (523Hz).")


    def _generate_tone(
        self,
        frequency: float,
        duration: float,
        volume: float = 0.8
    ) -> Optional[pygame.mixer.Sound]:
        """
        Generates a simple sine wave tone as a pygame Sound object.

        A sine wave is the purest possible sound — a smooth oscillation
        at a single frequency. Used as a fallback when no sound files exist.

        Arguments:
            frequency: pitch in Hz (440 = A4, 880 = A5, 523 = C5)
            duration:  length in seconds
            volume:    amplitude 0.0 to 1.0

        Returns:
            A pygame.mixer.Sound object ready to play, or None on error.
        """

        try:
            # Sample rate — must match what we initialised pygame.mixer with.
            sample_rate = 22050

            # Total number of audio samples in this tone.
            # duration seconds × samples per second = total samples.
            n_samples = int(sample_rate * duration)

            # np.linspace creates an evenly-spaced array of time values.
            # From 0 to duration, with n_samples steps.
            # This gives us the time value for each sample.
            t = np.linspace(0, duration, n_samples, endpoint=False)

            # Generate a sine wave at the given frequency.
            # np.sin(2π × frequency × t) creates one full cycle per 1/frequency seconds.
            # Multiply by volume to control amplitude.
            # Multiply by 32767 to scale to 16-bit integer range (0 to 32767).
            wave = (np.sin(2 * np.pi * frequency * t) * volume * 32767).astype(np.int16)

            # Apply a short fade-in and fade-out to avoid clicking sounds.
            # Clicking happens when the sound starts or ends abruptly at non-zero amplitude.
            # Fade over the first and last 10ms (220 samples at 22050Hz).
            fade_samples = min(220, n_samples // 4)

            # np.linspace(0, 1, fade_samples) creates a ramp from 0 to 1.
            # Multiplying the wave start by this ramp fades it in smoothly.
            wave[:fade_samples] = (
                wave[:fade_samples] * np.linspace(0, 1, fade_samples)
            ).astype(np.int16)

            # np.linspace(1, 0, fade_samples) creates a ramp from 1 to 0.
            # Multiplying the wave end by this ramp fades it out smoothly.
            wave[-fade_samples:] = (
                wave[-fade_samples:] * np.linspace(1, 0, fade_samples)
            ).astype(np.int16)

            # Convert mono to stereo by stacking the wave twice as columns.
            # np.column_stack creates a 2D array: [[left, right], [left, right], ...]
            # Both channels start identical — panning is applied later at play time.
            stereo_wave = np.column_stack([wave, wave])

            # np.ascontiguousarray ensures the array memory layout is
            # compatible with pygame — pygame requires contiguous memory.
            stereo_wave = np.ascontiguousarray(stereo_wave)

            # pygame.sndarray.make_sound() converts a numpy array into
            # a pygame Sound object that can be played with .play().
            sound = pygame.sndarray.make_sound(stereo_wave)

            return sound

        except Exception as e:
            logger.error(f"Tone generation failed: {e}")
            return None


    # =========================================================================
    # TEXT TO SPEECH
    # =========================================================================

    def speak(
        self,
        text: str,
        priority: int = SpeechPriority.NORMAL,
        ttl_sec: float = 3.0
    ):
        """
        Adds a speech request to the priority queue.

        Returns IMMEDIATELY — does not wait for speech to finish.
        The background thread picks it up and speaks it.

        Arguments:
            text:     the text to speak
            priority: SpeechPriority constant (default: NORMAL)
            ttl_sec:  how many seconds this request stays valid.
                      Older requests are discarded so stale alerts don't pile up.
        """

        # Don't accept speech requests if audio isn't ready yet.
        if not self._ready:
            logger.warning(f"Audio not ready. Discarding speech: '{text[:40]}'")
            return

        # Don't queue empty strings.
        if not text or not text.strip():
            return

        # Create a SpeechRequest dataclass object.
        # sort_index = priority level (lower = higher priority = dequeued first).
        request = SpeechRequest(
            sort_index   = priority,
            text         = text.strip(),
            timestamp_ms = get_timestamp_ms(),
            ttl_sec      = ttl_sec
        )

        # Put the request in the priority queue.
        # PriorityQueue.put() is thread-safe — safe to call from any thread.
        self._speech_queue.put(request)
        logger.debug(f"Queued speech [priority={priority}]: '{text[:50]}'")


    def _speech_worker(self):
        """
        Background thread — continuously reads and speaks from the queue.

        This runs in a separate thread forever until _running is False.
        It reads the highest-priority SpeechRequest, checks if it's still
        valid (not expired), and speaks it using pyttsx3.

        The underscore prefix means this is private — only called internally.
        """

        logger.info("Speech worker thread running.")

        while self._running:

            try:
                # queue.get(timeout=0.1) waits up to 0.1 seconds for an item.
                # If nothing arrives in 0.1s, it raises queue.Empty.
                # This prevents the thread from spinning at 100% CPU when idle.
                request = self._speech_queue.get(timeout=0.1)

            except queue.Empty:
                # Nothing in the queue — loop back and wait again.
                continue

            # ── Check if this request has expired ─────────────────────────────
            # Calculate how old this request is in seconds.
            # get_timestamp_ms() returns milliseconds — divide by 1000 for seconds.
            age_sec = (get_timestamp_ms() - request.timestamp_ms) / 1000.0

            if age_sec > request.ttl_sec:
                # This request is too old — the situation may have changed.
                # Discard it without speaking.
                logger.debug(
                    f"Discarded expired speech request: '{request.text[:40]}' "
                    f"(age={age_sec:.1f}s > ttl={request.ttl_sec}s)"
                )
                # Mark the queue task as done so get() works correctly.
                self._speech_queue.task_done()
                continue

            # ── Speak the text ─────────────────────────────────────────────────
            if self._engine:
                try:
                    # Mark that we are currently speaking.
                    with self._speaking_lock:
                        self._is_speaking = True

                    logger.debug(f"Speaking: '{request.text[:60]}'")

                    # engine.say() queues the text for speaking.
                    self._engine.say(request.text)

                    # engine.runAndWait() blocks until the current speech is done.
                    # This is inside the thread so it blocks the THREAD, not the
                    # main loop. The main loop keeps running.
                    self._engine.runAndWait()

                except Exception as e:
                    logger.error(f"TTS speech error: {e}")

                finally:
                    # Always mark speaking as done, even if an error occurred.
                    with self._speaking_lock:
                        self._is_speaking = False

            # Mark this task as done in the queue.
            self._speech_queue.task_done()

        logger.info("Speech worker thread stopped.")


    # =========================================================================
    # SPATIAL AUDIO
    # =========================================================================

    def _angle_to_pan(self, angle_deg: float) -> float:
        """
        Converts a real-world angle in degrees to a stereo pan value.

        Pan value: -1.0 = full left, 0.0 = center, +1.0 = full right.

        The camera has a horizontal field of view of CAMERA_HFOV_DEG degrees.
        An angle of -HFOV/2 degrees = leftmost edge of frame.
        An angle of +HFOV/2 degrees = rightmost edge of frame.

        We normalise this range to -1.0 to +1.0.

        Example with HFOV = 73 degrees:
          angle = -36.5° → pan = -1.0 (far left)
          angle =   0.0° → pan =  0.0 (center)
          angle = +36.5° → pan = +1.0 (far right)
        """

        # Half the field of view — the maximum angle from center.
        half_fov = CAMERA_HFOV_DEG / 2.0

        # Divide by half_fov to normalise to -1.0 to +1.0 range.
        pan = angle_deg / half_fov

        # np.clip ensures the value stays within -1.0 to +1.0.
        # Angles outside the FOV (from Kalman prediction) could exceed ±1.0.
        pan = float(np.clip(pan, -1.0, 1.0))

        return pan


    def play_spatial_alert(self, angle_deg: float, urgency: str):
        """
        Plays a directional warning beep panned to match the obstacle's angle.

        The beep frequency and volume depend on urgency:
          DANGER  → 880Hz, loud, short cooldown (every 0.5 seconds)
          WARNING → 440Hz, medium, longer cooldown (every 2 seconds)

        The stereo pan position gives the user a sense of direction.

        Arguments:
            angle_deg: horizontal angle of the obstacle in degrees
                       (negative = left, 0 = ahead, positive = right)
            urgency:   "DANGER" or "WARNING"
        """

        if not self._ready:
            return

        # ── Rate limiting for spatial alerts ───────────────────────────────────
        # Check when we last played a spatial alert for this urgency level.
        now = time.time()
        last_time = self._last_spatial_alert.get(urgency, 0.0)
        cooldown  = self._spatial_cooldown.get(urgency, 1.0)

        if now - last_time < cooldown:
            # Too soon — skip this alert.
            return

        # Record this alert time.
        self._last_spatial_alert[urgency] = now

        # ── Select the appropriate sound ───────────────────────────────────────
        if urgency == "DANGER":
            sound = self._sound_danger
        elif urgency == "WARNING":
            sound = self._sound_warning
        else:
            sound = self._sound_chime

        if sound is None:
            return

        # ── Calculate stereo pan ───────────────────────────────────────────────
        pan = self._angle_to_pan(angle_deg)

        # ── Calculate left and right channel volumes ───────────────────────────
        # We use a simple equal-power panning law:
        #   left_vol  = cos((pan + 1) / 2 × π/2)
        #   right_vol = sin((pan + 1) / 2 × π/2)
        #
        # This gives smooth, natural-sounding stereo separation.
        # At pan=0: left=0.707, right=0.707 (equal volume, -3dB each)
        # At pan=-1: left=1.0, right=0.0 (full left)
        # At pan=+1: left=0.0, right=1.0 (full right)
        #
        # np.pi / 2 = 90 degrees in radians
        angle_rad  = (pan + 1) / 2 * (np.pi / 2)
        left_vol   = float(np.cos(angle_rad)) * ALERT_VOLUME
        right_vol  = float(np.sin(angle_rad)) * ALERT_VOLUME

        # ── Set the volume and play ────────────────────────────────────────────
        # pygame Sound.set_volume() sets a single volume for both channels.
        # For stereo panning we need to set left and right independently.
        # We do this by setting the overall volume and using a channel approach.

        # Get a free channel from pygame's mixer.
        # pygame.mixer.find_channel() returns a Channel object or None.
        # True = force-find a channel even if all are busy (stops the oldest).
        channel = pygame.mixer.find_channel(True)

        if channel:
            # channel.set_volume(left, right) sets stereo volumes independently.
            # This is the key function for spatial audio.
            channel.set_volume(left_vol, right_vol)

            # Play the sound on this specific channel (with our custom panning).
            channel.play(sound)

            logger.debug(
                f"Spatial alert: {urgency} at {angle_deg:+.1f}° "
                f"(pan={pan:+.2f}, L={left_vol:.2f}, R={right_vol:.2f})"
            )


    # =========================================================================
    # HIGH-LEVEL ANNOUNCEMENT FUNCTIONS
    # =========================================================================

    def announce_obstacle(self, track: Dict):

        label = track.get("label", "object")
        distance_mm = track.get("distance_mm", 0)
        angle_deg = track.get("angle_deg", 0.0)
        urgency = track.get("urgency", "UNKNOWN")

        if urgency in ("DANGER", "WARNING"):
            self.play_spatial_alert(angle_deg, urgency)

        if not self._cooldown.can_alert(label):
            return

        distance_str = mm_to_spoken(distance_mm)
        direction = self._angle_to_direction(angle_deg)

        # Glasses-mounted phrasing — calm and navigational, not panicked
        if urgency == "DANGER":
            text = f"Stop. {label}. {distance_str}. {direction}."
            priority = SpeechPriority.URGENT
            ttl = 5.0

        elif urgency == "WARNING":
            text = f"{label}. {distance_str}. {direction}."
            priority = SpeechPriority.HIGH
            ttl = 3.0

        else:
            text = f"{label}. {distance_str}."
            priority = SpeechPriority.LOW
            ttl = 2.0

        self.speak(text, priority=priority, ttl_sec=ttl)

    def announce_scene(self, description: str):
        """
        Speaks a VLM scene description.

        Called by control_unit.py when the VLM produces a new description.

        Arguments:
            description: plain English scene description from the VLM
        """

        if not description or not description.strip():
            return

        # VLM descriptions are NORMAL priority — informational, not urgent.
        self.speak(description, priority=SpeechPriority.NORMAL, ttl_sec=4.0)


    def announce_ocr(self, text: str):
        """
        Speaks text read by the OCR module.

        Called by control_unit.py when OCR reads text in the scene.

        Arguments:
            text: the text that was read from the scene
        """

        if not text or not text.strip():
            return

        # Prepend "Text reads:" so the user knows this is OCR output.
        spoken = f"Text reads: {text}"
        self.speak(spoken, priority=SpeechPriority.NORMAL, ttl_sec=5.0)


    def announce_face(self, name: str, details: str = ""):
        """
        Speaks a recognised face identification.

        Called by control_unit.py when face recognition identifies someone.

        Arguments:
            name:    the recognised person's name
            details: optional additional info (e.g. "your colleague")
        """

        if not name:
            return

        if details:
            text = f"{name}. {details}."
        else:
            text = f"This is {name}."

        # Face ID is HIGH priority — the user specifically looked at someone.
        self.speak(text, priority=SpeechPriority.HIGH, ttl_sec=4.0)


    def announce_banknote(self, denomination: str):
        """
        Speaks a recognised banknote denomination.

        Called by control_unit.py when banknote recognition identifies currency.

        Arguments:
            denomination: e.g. "50 Egyptian pounds" or "20 dollars"
        """

        if not denomination:
            return

        text = f"This is a {denomination} note."

        # Banknote is HIGH priority — user deliberately held up the note.
        self.speak(text, priority=SpeechPriority.HIGH, ttl_sec=4.0)


    def announce_mode_change(self, new_mode: str):
        """
        Speaks a brief confirmation when the system switches modes.

        Helps the user understand what the system is now doing.

        Arguments:
            new_mode: MODE constant string e.g. MODE.OCR
        """

        # Map mode names to user-friendly spoken phrases.
        mode_phrases = {
            "NAVIGATION":  "Navigation mode.",
            "OCR":         "Reading text.",
            "INTERACTION": "Interaction mode. Reach for the object.",
            "FACE_ID":     "Identifying person.",
            "BANKNOTE":    "Scanning banknote.",
        }

        phrase = mode_phrases.get(new_mode, f"{new_mode} mode.")
        # Mode change is NORMAL priority — informational.
        self.speak(phrase, priority=SpeechPriority.NORMAL, ttl_sec=2.0)


    # =========================================================================
    # DIRECTION HELPER
    # =========================================================================

    def _angle_to_direction(self, angle_deg: float) -> str:
        """
        Converts a horizontal angle to a spoken direction phrase.

        Calibrated for glasses-mounted camera — directions describe
        where to TURN THE HEAD, not where to move the hand.

        Examples:
          -36° → "sharp left"
           -15° → "turn left"
            -5° → "slightly left"
             0° → "straight ahead"
            +5° → "slightly right"
           +15° → "turn right"
           +36° → "sharp right"
        """

        abs_angle = abs(angle_deg)

        if abs_angle <= 5:
            return "straight ahead"
        elif abs_angle <= 15:
            direction = "slightly left" if angle_deg < 0 else "slightly right"
        elif abs_angle <= 30:
            direction = "turn left" if angle_deg < 0 else "turn right"
        else:
            direction = "sharp left" if angle_deg < 0 else "sharp right"

        return direction


    # =========================================================================
    # CONTROL FUNCTIONS
    # =========================================================================

    def stop_all(self):
        """
        Immediately stops all audio output.

        Stops:
          - All pygame spatial sounds
          - Clears the speech queue
          - Stops current TTS if possible

        Called by the state machine's emergency override — when a DANGER
        obstacle appears, we need to clear the audio slate immediately
        so the danger announcement is heard without delay.
        """

        # Stop all pygame sounds immediately.
        pygame.mixer.stop()

        # Clear the speech queue by draining it.
        # We can't directly clear a PriorityQueue, so we drain it.
        # The while loop reads and discards items until the queue is empty.
        drained = 0
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
                drained += 1
            except queue.Empty:
                break

        if drained > 0:
            logger.debug(f"Cleared {drained} queued speech requests.")

        # Stop current TTS if the engine supports it.
        # pyttsx3's stop() method may not work on all platforms.
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass   # Not all platforms support mid-speech stopping.

        logger.debug("Audio stopped.")


    def set_volume(self, level: float):
        """
        Sets the master volume for both TTS and spatial alerts.

        Arguments:
            level: float 0.0 (silent) to 1.0 (maximum)
        """

        # np.clip ensures value stays in valid range.
        self._volume = float(np.clip(level, 0.0, 1.0))

        # Update TTS engine volume.
        if self._engine:
            self._engine.setProperty('volume', self._volume)

        # Update pygame master volume.
        pygame.mixer.music.set_volume(self._volume)

        logger.info(f"Volume set to {self._volume:.1f}")


    def is_speaking(self) -> bool:
        """
        Returns True if TTS is currently speaking.

        Used by control_unit.py to avoid queuing more speech while
        something important is already being said.
        """
        with self._speaking_lock:
            return self._is_speaking


    def release(self):
        """
        Cleanly shuts down all audio resources.

        Always call this when ECHORA exits — in main.py's shutdown().
        """

        logger.info("Releasing audio resources...")

        # Signal the speech thread to stop.
        self._running = False

        # Wait for the speech thread to finish.
        if self._speech_thread and self._speech_thread.is_alive():
            self._speech_thread.join(timeout=3.0)

        # Stop all pygame sounds.
        try:
            pygame.mixer.stop()
            pygame.mixer.quit()
        except Exception:
            pass

        # Stop the TTS engine.
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass

        self._ready = False
        logger.info("Audio released cleanly.")


# =============================================================================
# SELF-TEST
# =============================================================================
# Tests audio WITHOUT the camera or obstacle detection.
# You should HEAR the output through your speakers/earphones.
#
# Run with: python audio_feedback.py

if __name__ == "__main__":

    import time

    print("=== ECHORA audio_feedback.py self-test ===")
    print("You should hear audio through your speakers/earphones.\n")

    # Create and initialise the audio system.
    audio = AudioFeedback()
    audio.init_audio()

    # Give the system a moment to fully start.
    time.sleep(0.5)

    # ── Test 1: Basic speech ──────────────────────────────────────────────────
    print("Test 1: Basic speech")
    audio.speak("ECHORA audio system online.", priority=SpeechPriority.NORMAL)
    time.sleep(2.5)
    print("  Done\n")

    # ── Test 2: Priority ordering ──────────────────────────────────────────────
    print("Test 2: Priority — LOW queued first, URGENT should play first")
    audio.speak("This is a low priority message.", priority=SpeechPriority.LOW)
    audio.speak("Danger. Person. 40 centimetres.", priority=SpeechPriority.URGENT)
    time.sleep(5.0)
    print("  Done\n")

    # ── Test 3: Spatial audio panning ─────────────────────────────────────────
    print("Test 3: Spatial alerts — listen for direction")
    print("  Far left...")
    audio.play_spatial_alert(-35.0, "DANGER")
    time.sleep(1.0)

    print("  Center...")
    audio.play_spatial_alert(0.0, "WARNING")
    time.sleep(1.0)

    print("  Far right...")
    audio.play_spatial_alert(35.0, "DANGER")
    time.sleep(1.0)
    print("  Done\n")

    # ── Test 4: announce_obstacle ─────────────────────────────────────────────
    print("Test 4: Obstacle announcements")
    fake_danger_track = {
        "label":       "person",
        "distance_mm": 500,
        "angle_deg":   -20.0,
        "urgency":     "DANGER",
    }
    fake_warning_track = {
        "label":       "chair",
        "distance_mm": 1500,
        "angle_deg":   15.0,
        "urgency":     "WARNING",
    }
    audio.announce_obstacle(fake_danger_track)
    time.sleep(0.1)
    audio.announce_obstacle(fake_warning_track)
    time.sleep(4.0)
    print("  Done\n")

    # ── Test 5: Mode change ────────────────────────────────────────────────────
    print("Test 5: Mode change announcements")
    audio.announce_mode_change("OCR")
    time.sleep(2.0)
    audio.announce_mode_change("NAVIGATION")
    time.sleep(2.0)
    print("  Done\n")

    # ── Test 6: OCR and banknote ──────────────────────────────────────────────
    print("Test 6: OCR and banknote")
    audio.announce_ocr("Emergency Exit")
    time.sleep(3.0)
    audio.announce_banknote("50 Egyptian pounds")
    time.sleep(3.0)
    print("  Done\n")

    # ── Clean shutdown ────────────────────────────────────────────────────────
    print("Shutting down audio...")
    audio.release()
    print("\n=== Self-test complete ===")