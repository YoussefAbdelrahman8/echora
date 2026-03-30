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
#
# Fixes applied:
#   - self._ready = True added after watchdog starts
#   - _last_speak_time updated after each successful speech
#   - Watchdog monitors and restarts speech thread on crash
#   - _reinit_engine() restores TTS engine after crash
# =============================================================================


import pyttsx3
import pygame
import pygame.mixer
import pygame.sndarray
import numpy as np
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict

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

class SpeechPriority:
    URGENT = -3
    HIGH   = -2
    NORMAL = -1
    LOW    =  0


# =============================================================================
# SPEECH REQUEST DATACLASS
# =============================================================================

@dataclass(order=True)
class SpeechRequest:

    sort_index:   int
    text:         str   = field(compare=False)
    timestamp_ms: float = field(compare=False, default_factory=get_timestamp_ms)
    ttl_sec:      float = field(compare=False, default=3.0)


# =============================================================================
# AUDIO FEEDBACK CLASS
# =============================================================================

class AudioFeedback:

    def __init__(self):

        # TTS engine
        self._engine: Optional[pyttsx3.Engine] = None

        # ── CRITICAL: _ready starts False, set True at end of init_audio() ────
        self._ready: bool = False

        # Speech queue
        self._speech_queue: queue.PriorityQueue = queue.PriorityQueue()

        # Thread control
        self._running:        bool                    = False
        self._speech_thread:  Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None

        # Alert cooldown
        self._cooldown = AlertCooldown()

        # Pygame sounds
        self._sound_danger:  Optional[pygame.mixer.Sound] = None
        self._sound_warning: Optional[pygame.mixer.Sound] = None
        self._sound_chime:   Optional[pygame.mixer.Sound] = None

        # Volume
        self._volume: float = TTS_VOLUME

        # Speaking state
        self._is_speaking:  bool             = False
        self._speaking_lock = threading.Lock()

        # Spatial alert rate limiting
        self._last_spatial_alert: Dict[str, float] = {}
        self._spatial_cooldown = {
            "DANGER":  0.5,
            "WARNING": 2.0,
        }

        # Keep-alive timestamp — used to detect TTS engine inactivity
        self._last_speak_time: float = time.time()

        logger.info("AudioFeedback created. Call init_audio() to start.")


    # =========================================================================
    # INITIALISATION
    # =========================================================================

    def init_audio(self):

        logger.info("Initialising audio system...")

        # ── Step 1: pygame mixer ───────────────────────────────────────────────
        try:
            pygame.mixer.init(
                frequency = 22050,
                size      = -16,
                channels  = 2,
                buffer    = 512
            )
            logger.info("pygame mixer initialised.")
        except Exception as e:
            logger.error(f"pygame mixer init failed: {e}")

        # ── Step 2: Load sounds ────────────────────────────────────────────────
        self._load_sounds()

        # ── Step 3: Initialise TTS engine ──────────────────────────────────────
        try:
            self._engine = pyttsx3.init()
            logger.info("TTS engine initialised.")
        except Exception as e:
            logger.error(f"TTS engine init failed: {e}")
            self._engine = None

        # ── Step 4: Configure TTS ──────────────────────────────────────────────
        if self._engine:
            self._engine.setProperty('rate',   TTS_RATE)
            self._engine.setProperty('volume', self._volume)

            voices          = self._engine.getProperty('voices')
            preferred_names = ['samantha', 'karen', 'victoria', 'alex', 'arabic', 'naayf', 'hoda']
            selected_voice  = None

            for voice in voices:
                if any(n in voice.name.lower() for n in preferred_names):
                    selected_voice = voice.id
                    logger.info(f"Selected TTS voice: {voice.name}")
                    break

            if selected_voice:
                self._engine.setProperty('voice', selected_voice)
            else:
                logger.info("Using default TTS voice.")

        # ── Step 5: Start speech worker thread ─────────────────────────────────
        self._running = True

        self._speech_thread = threading.Thread(
            target = self._speech_worker,
            daemon = True
        )
        self._speech_thread.start()
        logger.info("Speech worker thread started.")

        # ── Step 6: Start watchdog thread ──────────────────────────────────────
        # Monitors speech thread and restarts it if it crashes.
        self._watchdog_thread = threading.Thread(
            target = self._speech_watchdog,
            daemon = True
        )
        self._watchdog_thread.start()
        logger.info("Speech watchdog started.")

        # ── CRITICAL: Mark system as ready ────────────────────────────────────
        # Without this line, speak() silently discards every request.
        self._ready = True

        logger.info("Audio system ready.")


    def _reinit_engine(self):
        """
        Reinitialises the pyttsx3 TTS engine after a crash.
        Called by the speech worker when an error occurs,
        and by the watchdog after restarting the thread.
        """

        logger.info("Reinitialising TTS engine...")

        try:
            if self._engine:
                try:
                    self._engine.stop()
                except Exception:
                    pass

            self._engine = pyttsx3.init()
            self._engine.setProperty('rate',   TTS_RATE)
            self._engine.setProperty('volume', self._volume)

            voices          = self._engine.getProperty('voices')
            preferred_names = ['samantha', 'karen', 'victoria', 'alex']
            selected_voice  = None

            for voice in voices:
                if any(n in voice.name.lower() for n in preferred_names):
                    selected_voice = voice.id
                    break

            if selected_voice:
                self._engine.setProperty('voice', selected_voice)

            logger.info("TTS engine reinitialised successfully.")

        except Exception as e:
            logger.error(f"Failed to reinitialise TTS engine: {e}")
            self._engine = None


    # =========================================================================
    # SOUND LOADING
    # =========================================================================

    def _load_sounds(self):

        if SOUND_DANGER_PATH.exists():
            self._sound_danger = pygame.mixer.Sound(str(SOUND_DANGER_PATH))
            logger.info(f"Loaded danger sound: {SOUND_DANGER_PATH.name}")
        else:
            self._sound_danger = self._generate_tone(880, 0.3, 0.9)
            logger.info("Generated danger tone (880Hz).")

        if SOUND_WARNING_PATH.exists():
            self._sound_warning = pygame.mixer.Sound(str(SOUND_WARNING_PATH))
            logger.info(f"Loaded warning sound: {SOUND_WARNING_PATH.name}")
        else:
            self._sound_warning = self._generate_tone(440, 0.4, 0.7)
            logger.info("Generated warning tone (440Hz).")

        if SOUND_CHIME_PATH.exists():
            self._sound_chime = pygame.mixer.Sound(str(SOUND_CHIME_PATH))
            logger.info(f"Loaded chime sound: {SOUND_CHIME_PATH.name}")
        else:
            self._sound_chime = self._generate_tone(523, 0.2, 0.5)
            logger.info("Generated chime tone (523Hz).")


    def _generate_tone(
        self,
        frequency: float,
        duration:  float,
        volume:    float = 0.8
    ) -> Optional[pygame.mixer.Sound]:

        try:
            sample_rate = 22050
            n_samples   = int(sample_rate * duration)
            t           = np.linspace(0, duration, n_samples, endpoint=False)
            wave        = (
                np.sin(2 * np.pi * frequency * t) * volume * 32767
            ).astype(np.int16)

            fade_samples = min(220, n_samples // 4)
            wave[:fade_samples] = (
                wave[:fade_samples] * np.linspace(0, 1, fade_samples)
            ).astype(np.int16)
            wave[-fade_samples:] = (
                wave[-fade_samples:] * np.linspace(1, 0, fade_samples)
            ).astype(np.int16)

            stereo_wave = np.ascontiguousarray(np.column_stack([wave, wave]))
            return pygame.sndarray.make_sound(stereo_wave)

        except Exception as e:
            logger.error(f"Tone generation failed: {e}")
            return None


    # =========================================================================
    # TEXT TO SPEECH
    # =========================================================================

    def speak(
        self,
        text:     str,
        priority: int   = SpeechPriority.NORMAL,
        ttl_sec:  float = 3.0
    ):
        """
        Adds a speech request to the priority queue.
        Returns immediately — speech happens in background thread.
        """

        if not self._ready:
            logger.warning(f"Audio not ready. Discarding: '{text[:40]}'")
            return

        if not text or not text.strip():
            return

        request = SpeechRequest(
            sort_index   = priority,
            text         = text.strip(),
            timestamp_ms = get_timestamp_ms(),
            ttl_sec      = ttl_sec
        )

        self._speech_queue.put(request)
        logger.debug(f"Queued [p={priority}]: '{text[:50]}'")


    def _speech_worker(self):
        """
        Background thread — reads and speaks from the priority queue.

        Handles:
          - TTL expiry (discard stale requests)
          - Engine crashes (reinitialise automatically)
          - Keep-alive ping (prevents pyttsx3 COM timeout on Windows)
        """

        logger.info("Speech worker thread running.")

        while self._running:

            try:
                request = self._speech_queue.get(timeout=0.1)

            except queue.Empty:
                # ── Keep-alive ping ────────────────────────────────────────────
                # pyttsx3 on Windows uses a COM object that times out after
                # ~30 seconds of inactivity. A periodic runAndWait() ping
                # keeps it alive and prevents silent crashes.
                if self._engine:
                    if time.time() - self._last_speak_time > 25:
                        try:
                            self._engine.runAndWait()
                            self._last_speak_time = time.time()
                        except Exception:
                            pass
                continue

            # ── Check TTL ──────────────────────────────────────────────────────
            age_sec = (get_timestamp_ms() - request.timestamp_ms) / 1000.0
            if age_sec > request.ttl_sec:
                logger.debug(f"Expired: '{request.text[:40]}'")
                self._speech_queue.task_done()
                continue

            # ── Speak ──────────────────────────────────────────────────────────
            if self._engine:
                try:
                    with self._speaking_lock:
                        self._is_speaking = True

                    logger.debug(f"Speaking: '{request.text[:60]}'")
                    self._engine.say(request.text)
                    self._engine.runAndWait()

                    # ── CRITICAL: update keep-alive timestamp ──────────────────
                    self._last_speak_time = time.time()

                except Exception as e:
                    logger.warning(f"TTS error: {e} — reinitialising...")
                    self._reinit_engine()

                finally:
                    with self._speaking_lock:
                        self._is_speaking = False

            self._speech_queue.task_done()

        logger.info("Speech worker thread stopped.")


    def _speech_watchdog(self):
        """
        Monitors the speech thread every 3 seconds.
        Restarts it automatically if it dies.
        Handles the pyttsx3 silent crash on Windows.
        """

        while self._running:
            time.sleep(3.0)

            if (self._running
                    and self._speech_thread is not None
                    and not self._speech_thread.is_alive()):

                logger.warning("Speech thread died — restarting...")

                # Reinitialise engine before restarting thread.
                self._reinit_engine()

                self._speech_thread = threading.Thread(
                    target = self._speech_worker,
                    daemon = True
                )
                self._speech_thread.start()
                logger.info("Speech thread restarted.")


    # =========================================================================
    # SPATIAL AUDIO
    # =========================================================================

    def _angle_to_pan(self, angle_deg: float) -> float:
        half_fov = CAMERA_HFOV_DEG / 2.0
        pan      = angle_deg / half_fov
        return float(np.clip(pan, -1.0, 1.0))


    def play_spatial_alert(self, angle_deg: float, urgency: str):

        if not self._ready:
            return

        now       = time.time()
        last_time = self._last_spatial_alert.get(urgency, 0.0)
        cooldown  = self._spatial_cooldown.get(urgency, 1.0)

        if now - last_time < cooldown:
            return

        self._last_spatial_alert[urgency] = now

        if urgency == "DANGER":
            sound = self._sound_danger
        elif urgency == "WARNING":
            sound = self._sound_warning
        else:
            sound = self._sound_chime

        if sound is None:
            return

        pan       = self._angle_to_pan(angle_deg)
        angle_rad = (pan + 1) / 2 * (np.pi / 2)
        left_vol  = float(np.cos(angle_rad)) * ALERT_VOLUME
        right_vol = float(np.sin(angle_rad)) * ALERT_VOLUME

        channel = pygame.mixer.find_channel(True)
        if channel:
            channel.set_volume(left_vol, right_vol)
            channel.play(sound)
            logger.debug(
                f"Spatial: {urgency} {angle_deg:+.1f}deg "
                f"L={left_vol:.2f} R={right_vol:.2f}"
            )


    # =========================================================================
    # HIGH-LEVEL ANNOUNCEMENT FUNCTIONS
    # =========================================================================

    def announce_obstacle(self, track: Dict):

        label       = track.get("label",       "object")
        distance_mm = track.get("distance_mm", 0)
        angle_deg   = track.get("angle_deg",   0.0)
        urgency     = track.get("urgency",     "UNKNOWN")

        if urgency in ("DANGER", "WARNING"):
            self.play_spatial_alert(angle_deg, urgency)

        if not self._cooldown.can_alert(label):
            return

        distance_str = mm_to_spoken(distance_mm)
        direction    = self._angle_to_direction(angle_deg)

        if urgency == "DANGER":
            text     = f"Stop. {label}. {distance_str}. {direction}."
            priority = SpeechPriority.URGENT
            ttl      = 5.0
        elif urgency == "WARNING":
            text     = f"{label}. {distance_str}. {direction}."
            priority = SpeechPriority.HIGH
            ttl      = 3.0
        else:
            text     = f"{label}. {distance_str}."
            priority = SpeechPriority.LOW
            ttl      = 2.0

        self.speak(text, priority=priority, ttl_sec=ttl)


    def announce_scene(self, description: str):
        if not description or not description.strip():
            return
        self.speak(description, priority=SpeechPriority.NORMAL, ttl_sec=4.0)


    def announce_ocr(self, text: str):
        if not text or not text.strip():
            return
        self.speak(
            f"Text reads: {text}",
            priority = SpeechPriority.NORMAL,
            ttl_sec  = 5.0
        )


    def announce_face(self, name: str, details: str = ""):
        if not name:
            return
        text = f"{name}. {details}." if details else f"This is {name}."
        self.speak(text, priority=SpeechPriority.HIGH, ttl_sec=4.0)


    def announce_banknote(self, denomination: str):
        if not denomination:
            return
        self.speak(
            f"This is a {denomination} note.",
            priority = SpeechPriority.HIGH,
            ttl_sec  = 4.0
        )


    def announce_mode_change(self, new_mode: str):
        mode_phrases = {
            "NAVIGATION":  "Navigation mode.",
            "OCR":         "Reading text.",
            "INTERACTION": "Interaction mode. Reach for the object.",
            "FACE_ID":     "Identifying person.",
            "BANKNOTE":    "Scanning banknote.",
        }
        phrase = mode_phrases.get(new_mode, f"{new_mode} mode.")
        self.speak(phrase, priority=SpeechPriority.NORMAL, ttl_sec=2.0)


    # =========================================================================
    # DIRECTION HELPER
    # =========================================================================

    def _angle_to_direction(self, angle_deg: float) -> str:

        abs_angle = abs(angle_deg)

        if abs_angle <= 5:
            return "straight ahead"
        elif abs_angle <= 15:
            return "slightly left"  if angle_deg < 0 else "slightly right"
        elif abs_angle <= 30:
            return "turn left"      if angle_deg < 0 else "turn right"
        else:
            return "sharp left"     if angle_deg < 0 else "sharp right"


    # =========================================================================
    # CONTROL FUNCTIONS
    # =========================================================================

    def stop_all(self):

        pygame.mixer.stop()

        drained = 0
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
                drained += 1
            except queue.Empty:
                break

        if drained > 0:
            logger.debug(f"Cleared {drained} queued requests.")

        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass

        logger.debug("Audio stopped.")


    def set_volume(self, level: float):
        self._volume = float(np.clip(level, 0.0, 1.0))
        if self._engine:
            self._engine.setProperty('volume', self._volume)
        logger.info(f"Volume: {self._volume:.1f}")


    def is_speaking(self) -> bool:
        with self._speaking_lock:
            return self._is_speaking


    def release(self):

        logger.info("Releasing audio resources...")

        self._running = False

        if self._speech_thread and self._speech_thread.is_alive():
            self._speech_thread.join(timeout=3.0)

        try:
            pygame.mixer.stop()
            pygame.mixer.quit()
        except Exception:
            pass

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

if __name__ == "__main__":

    print("=== ECHORA audio_feedback.py self-test ===")
    print("You should hear audio through your speakers/earphones.\n")

    audio = AudioFeedback()
    audio.init_audio()
    time.sleep(0.5)

    print("Test 1: Basic speech")
    audio.speak("ECHORA audio system online.", priority=SpeechPriority.NORMAL)
    time.sleep(2.5)
    print("  Done\n")

    print("Test 2: Priority — LOW first, URGENT should play first")
    audio.speak("This is a low priority message.", priority=SpeechPriority.LOW)
    audio.speak("Danger. Person. 40 centimetres.",  priority=SpeechPriority.URGENT)
    time.sleep(5.0)
    print("  Done\n")

    print("Test 3: Spatial alerts")
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

    print("Test 4: Obstacle announcements")
    audio.announce_obstacle({
        "label": "person", "distance_mm": 500,
        "angle_deg": -20.0, "urgency": "DANGER"
    })
    time.sleep(0.1)
    audio.announce_obstacle({
        "label": "chair", "distance_mm": 1500,
        "angle_deg": 15.0, "urgency": "WARNING"
    })
    time.sleep(4.0)
    print("  Done\n")

    print("Test 5: Mode changes")
    audio.announce_mode_change("OCR")
    time.sleep(2.0)
    audio.announce_mode_change("NAVIGATION")
    time.sleep(2.0)
    print("  Done\n")

    print("Test 6: OCR and banknote")
    audio.announce_ocr("Emergency Exit")
    time.sleep(3.0)
    audio.announce_banknote("50 Egyptian pounds")
    time.sleep(3.0)
    print("  Done\n")

    print("Shutting down...")
    audio.release()
    print("\n=== Self-test complete ===")