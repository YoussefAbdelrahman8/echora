import sys
import cv2
import numpy as np
import time
import threading
from typing import Optional, Dict

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080
    )

from src.core.config import settings, MODE
from src.core.utils import logger, get_timestamp_ms, draw_overlay, mm_to_spoken
from src.hardware.camera import EchoraCamera
from src.perception.obstacle_detection import ObstacleDetector
from src.core.state_machine import StateMachine
from src.hardware.audio_feedback import AudioFeedback, SpeechPriority

from src.perception import ocr
from src.perception.interaction_detection import InteractionDetector
from src.perception import banknote
from src.perception import echora_face as face_recognition

from src.storage.database import init_database
from src.perception.echora_face import init_face_recognition
from src.hardware.haptic_feedback import init_haptic

SHOW_DEBUG_WINDOW = True
PERF_LOG_EVERY_N_FRAMES = 30

class ControlUnit:
    def __init__(self, start_in_manual: bool = False):
        self._camera: Optional[EchoraCamera] = None
        self._detector: Optional[ObstacleDetector] = None
        self._state_machine: Optional[StateMachine] = None
        self._audio: Optional[AudioFeedback] = None
        self._interaction_detector: Optional[InteractionDetector] = None

        self._started = False
        self._running = False
        self._frame_count = 0

        self._frame_times = []
        self._slow_frames = 0
        self._start_time = 0.0

        self._last_mode = MODE.NAVIGATION
        self._last_scene_desc = ""
        self._last_ocr_text = ""
        self._last_face_name = ""
        self._last_denomination = ""

        self._last_ocr_dist = 0.0
        self._last_face_conf = 0.0
        self._last_note_visible = False
        self._last_interact_dist = 0.0

        self._ocr_running = False
        self._face_id_running = False
        self._face_id_result: Optional[Dict] = None

        self._auto_mode = not start_in_manual
        self._manual_mode = MODE.NAVIGATION

        self._key_to_mode = {
            ord('1'): MODE.NAVIGATION,
            ord('2'): MODE.OCR,
            ord('3'): MODE.INTERACTION,
            ord('4'): MODE.FACE_ID,
            ord('5'): MODE.BANKNOTE,
        }

        logger.info(f"ControlUnit created. Starting in {'AUTO' if self._auto_mode else 'MANUAL'} mode.")

    def startup(self):
        logger.info("=" * 60)
        logger.info("ECHORA Control Unit starting up...")
        logger.info("=" * 60)

        self._start_time = time.time()

        self._camera = EchoraCamera()
        self._camera.init_pipeline()
        
        self._detector = ObstacleDetector()
        self._detector.load_model()
        
        self._interaction_detector = InteractionDetector()
        self._interaction_detector.load_model()
        
        ocr.init_ocr()
        banknote.init_banknote()
        
        init_database()
        init_face_recognition()
        init_haptic()
        
        self._audio = AudioFeedback()
        self._audio.init_audio()
        
        self._state_machine = StateMachine()
        self._register_callbacks()
        
        time.sleep(0.3)

        if self._auto_mode:
            self._audio.speak("ECHORA online. Auto mode active.", priority=SpeechPriority.NORMAL)
        else:
            self._audio.speak("ECHORA online. Manual testing mode. Press 1 to 5 to select a mode.", priority=SpeechPriority.NORMAL)

        self._started = True
        logger.info("=" * 60)
        logger.info("ECHORA startup complete. Entering main loop.")
        logger.info("=" * 60)

    def _toggle_auto_manual(self):
        self._auto_mode = not self._auto_mode
        if self._auto_mode:
            logger.info("Switched to AUTO mode.")
            self._audio.speak("Auto mode.", priority=SpeechPriority.HIGH, ttl_sec=2.0)
        else:
            self._manual_mode = self._last_mode
            logger.info(f"Switched to MANUAL mode. Locked on: {self._manual_mode}.")
            self._audio.speak(f"Manual mode. {self._manual_mode.lower().replace('_', ' ')} locked.", priority=SpeechPriority.HIGH, ttl_sec=3.0)

    def _set_manual_mode(self, new_mode: str):
        if self._auto_mode or new_mode == self._manual_mode:
            return

        old_mode = self._manual_mode
        self._manual_mode = new_mode
        self._last_mode = new_mode

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
        self._audio.announce_mode_change(new_mode)

    def _on_enter_interaction(self):
        target = self._interaction_detector._target_object
        if target:
            label = target.get("label", "object")
            dist = target.get("distance_mm", 0)
            self._audio.speak(
                f"{label} detected. {mm_to_spoken(dist)}. Raise your hand to reach it.",
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
        sm.register_callback(mode=MODE.NAVIGATION, on_enter=lambda: self._audio.announce_mode_change(MODE.NAVIGATION), on_exit=self._detector.reset_tracker)
        sm.register_callback(mode=MODE.OCR, on_enter=lambda: (self._audio.announce_mode_change(MODE.OCR), self._audio.stop_all()),
                             on_exit=lambda: (self._reset_ocr_state(), __import__('src.perception.ocr').reset_ocr()))
        sm.register_callback(mode=MODE.INTERACTION, on_enter=self._on_enter_interaction, on_exit=self._reset_interaction_state)
        sm.register_callback(mode=MODE.FACE_ID, on_enter=lambda: self._audio.announce_mode_change(MODE.FACE_ID),
                             on_exit=lambda: (self._reset_face_state(), __import__('src.perception.echora_face').reset_face()))
        sm.register_callback(mode=MODE.BANKNOTE, on_enter=lambda: self._audio.announce_mode_change(MODE.BANKNOTE),
                             on_exit=lambda: (self._reset_banknote_state(), __import__('src.perception.banknote').reset_banknote()))

    def run(self):
        if not self._started:
            logger.error("Cannot run — call startup() first.")
            return

        self._running = True
        logger.info("Main loop started. TAB = toggle AUTO/MANUAL | 1-5 = set mode | Q = quit")

        try:
            while self._running:
                bundle = self._camera.get_synced_bundle()
                if not bundle: continue

                frame_start = get_timestamp_ms()
                debug_frame = self._process_frame(bundle)
                frame_duration = get_timestamp_ms() - frame_start

                self._frame_times.append(frame_duration)
                self._frame_times = self._frame_times[-30:]
                if frame_duration > settings.MAX_FRAME_TIME_MS:
                    self._slow_frames += 1

                if SHOW_DEBUG_WINDOW and debug_frame is not None:
                    cv2.imshow("ECHORA - Debug", debug_frame)
                    self._handle_key(cv2.waitKey(1))

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
        if key == -1: return
        if key in (ord('q'), ord('Q')):
            logger.info("Q pressed — quitting.")
            self._running = False
            return
        if key == 9:
            self._toggle_auto_manual()
            return
        if key in self._key_to_mode:
            self._set_manual_mode(self._key_to_mode[key])

    def _process_frame(self, bundle: Dict) -> Optional[np.ndarray]:
        try:
            rgb_frame = bundle["rgb"]
            depth_map = bundle["depth"]

            if self._face_id_result is not None:
                result = self._face_id_result
                self._face_id_result = None
                name = result.get("name", "")
                if name and name != self._last_face_name:
                    self._last_face_name = name
                    self._audio.speak(f"This is {name}.", priority=SpeechPriority.HIGH, ttl_sec=8.0)
                    logger.info(f"Face announced: {name}")
                elif not name and self._last_face_name == "":
                    self._last_face_name = "unknown"
                    self._audio.speak("Unknown person.", priority=SpeechPriority.NORMAL, ttl_sec=4.0)

            obstacle_result = self._detector.update(bundle)

            if self._frame_count % 20 == 0 and not self._ocr_running:
                self._ocr_running = True
                def _ocr_worker(f=rgb_frame.copy(), d=depth_map.copy()):
                    self._last_ocr_dist = ocr.get_text_distance(f, d)
                    self._ocr_running = False
                threading.Thread(target=_ocr_worker, daemon=True).start()

            if self._frame_count % 5 == 0 and self._last_mode != MODE.FACE_ID:
                self._last_face_conf = face_recognition.detect_face(rgb_frame)

            if self._frame_count % 5 == 0:
                self._last_note_visible = banknote.detect_banknote(rgb_frame)

            if self._frame_count % 5 == 0:
                self._last_interact_dist = self._interaction_detector.scan_for_interactables(
                    detections=obstacle_result.get("tracks", []), depth_map=depth_map
                )

            current_mode = self._manual_mode
            if self._auto_mode:
                current_mode = self._state_machine.update(
                    bundle=bundle, obstacle_result=obstacle_result,
                    ocr_text_distance=self._last_ocr_dist, face_confidence=self._last_face_conf,
                    interactable_distance=self._last_interact_dist, banknote_visible=self._last_note_visible
                )
                if current_mode != self._last_mode:
                    logger.info(f"Mode: {self._last_mode} -> {current_mode} [AUTO]")
                    self._last_mode = current_mode

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

            debug_frame = rgb_frame.copy()
            if current_mode == MODE.NAVIGATION and obstacle_result.get("tracks"):
                debug_frame = draw_overlay(debug_frame, obstacle_result["tracks"])
            if current_mode == MODE.INTERACTION and self._interaction_detector._last_grid is not None:
                debug_frame = self._interaction_detector.draw_debug_overlay(debug_frame, {
                    "phase": self._interaction_detector._phase, "hand": None,
                    "target": self._interaction_detector._target_object, "electrode_grid": self._interaction_detector._last_grid
                })
            debug_frame = self._draw_debug_overlay(debug_frame, obstacle_result, current_mode)

            return debug_frame

        except Exception as e:
            logger.error(f"Frame processing error: {e}", exc_info=True)
            return bundle.get("rgb")

    def _handle_navigation(self, bundle: Dict, obstacle_result: Dict):
        for track in obstacle_result.get("danger", []): self._audio.announce_obstacle(track)
        for track in obstacle_result.get("warning", []): self._audio.announce_obstacle(track)
        scene_desc = obstacle_result.get("scene_desc", "")
        if scene_desc and scene_desc != self._last_scene_desc and len(scene_desc) > 10:
            self._last_scene_desc = scene_desc
            self._audio.announce_scene(scene_desc)
            logger.info(f"Scene: {scene_desc[:60]}...")

    def _handle_ocr(self, bundle: Dict):
        text = ocr.read_text(bundle["rgb"])
        if text and text.strip() and text.strip() != self._last_ocr_text:
            self._last_ocr_text = text.strip()
            self._audio.announce_ocr(text)
            logger.info(f"OCR: '{text[:60]}'")

    def _handle_interaction(self, bundle: Dict, obstacle_result: Dict):
        result = self._interaction_detector.update(bundle["rgb"], bundle["depth"], obstacle_result.get("tracks", []))
        if result.get("on_target"):
            self._audio.speak("Object reached.", priority=SpeechPriority.HIGH)
            logger.info("Interaction SUCCESS.")
            if self._auto_mode:
                self._state_machine.force_mode(MODE.NAVIGATION, reason="object reached")

    def _handle_face_id(self, bundle: Dict):
        if self._frame_count % 5 != 0 or self._face_id_running: return
        self._face_id_running = True
        def _face_worker(f=bundle["rgb"].copy()):
            try:
                name, details = face_recognition.identify_face(f)
                self._face_id_result = {"name": name, "details": details}
            except Exception as e:
                logger.error(f"Face worker error: {e}")
                self._face_id_result = {"name": "", "details": ""}
            finally:
                self._face_id_running = False
        threading.Thread(target=_face_worker, daemon=True).start()

    def _handle_banknote(self, bundle: Dict):
        denomination = banknote.classify_denomination(bundle["rgb"])
        if denomination and denomination != self._last_denomination:
            self._last_denomination = denomination
            self._audio.announce_banknote(denomination)
            logger.info(f"Banknote: {denomination}")

    def _draw_debug_overlay(self, frame: np.ndarray, obstacle_result: Dict, current_mode: str) -> np.ndarray:
        h, w = frame.shape[:2]
        mode_colors = {
            MODE.NAVIGATION: (0, 200, 80), MODE.OCR: (255, 165, 0),
            MODE.INTERACTION: (0, 165, 255), MODE.FACE_ID: (180, 0, 255), MODE.BANKNOTE: (0, 215, 255)
        }
        color = mode_colors.get(current_mode, (200, 200, 200))

        cv2.rectangle(frame, (0, 0), (370, 58), (0, 0, 0), -1)
        cv2.putText(frame, f"MODE: {current_mode}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
        indicator = "AUTO  (TAB to switch to manual)" if self._auto_mode else "MANUAL  (TAB: auto | 1-5: mode)"
        ind_color = (0, 200, 80) if self._auto_mode else (0, 165, 255)
        cv2.putText(frame, indicator, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ind_color, 1, cv2.LINE_AA)

        fps = 1000.0 / max(sum(self._frame_times) / len(self._frame_times), 1) if self._frame_times else 0.0
        fps_text = f"FPS: {fps:.1f}  F:{self._frame_count}"
        (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (w - tw - 16, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(frame, fps_text, (w - tw - 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        if current_mode == MODE.NAVIGATION:
            n_danger, n_warn, n_tracks = len(obstacle_result.get("danger", [])), len(obstacle_result.get("warning", [])), len(obstacle_result.get("tracks", []))
            for i, (text, col) in enumerate(reversed([
                (f"Tracks: {n_tracks}", (180, 180, 180)),
                (f"Danger: {n_danger}", (0, 0, 220) if n_danger > 0 else (180, 180, 180)),
                (f"Warning: {n_warn}", (0, 165, 255) if n_warn > 0 else (180, 180, 180))
            ])):
                cv2.putText(frame, text, (w - 140, h - 10 - i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

            most_urgent = self._detector.get_most_urgent_obstacle()
            if most_urgent:
                u_text = f"{most_urgent['label']}  {most_urgent['distance_mm']:.0f}mm  {most_urgent['angle_deg']:+.0f}deg"
                u_col = {"DANGER": (0, 0, 220), "WARNING": (0, 165, 255), "SAFE": (0, 200, 80)}.get(most_urgent["urgency"], (180, 180, 180))
                (utw, _), _ = cv2.getTextSize(u_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                x = (w - utw) // 2
                cv2.rectangle(frame, (x - 6, 60), (x + utw + 6, 86), (0, 0, 0), -1)
                cv2.putText(frame, u_text, (x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, u_col, 1, cv2.LINE_AA)

        if self._auto_mode:
            stats = self._state_machine.get_stats()
            for i, text in enumerate([
                f"Motion: {stats['motion_level']:.2f} m/s2", f"Stable: {'yes' if stats['is_stable'] else 'no'}",
                f"In mode: {stats['mode_duration_s']:.1f}s"
            ]):
                cv2.putText(frame, text, (8, h - 10 - i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "State machine: BYPASSED", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
            hint = "1:NAV  2:OCR  3:INTERACT  4:FACE  5:BANKNOTE"
            (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            hx = (w - hw) // 2
            cv2.rectangle(frame, (hx - 6, h - 80), (hx + hw + 6, h - 60), (0, 0, 0), -1)
            cv2.putText(frame, hint, (hx, h - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1, cv2.LINE_AA)

        return frame

    def _log_performance(self):
        if not self._frame_times: return
        fps = 1000.0 / max(sum(self._frame_times) / len(self._frame_times), 1)
        uptime = time.time() - self._start_time
        t_stats = self._detector.get_stats()["tracker"]
        m_str = self._state_machine.get_stats()['current_mode'] if self._auto_mode else f"MANUAL:{self._manual_mode}"
        logger.info(f"Performance | F:{self._frame_count} | FPS:{fps:.1f} | Slow:{self._slow_frames} | "
                    f"Up:{uptime:.0f}s | Mode:{m_str} | Tracks:{t_stats['confirmed']}")

    def shutdown(self):
        if not self._started: return
        logger.info("Shutting down ECHORA...")
        if self._audio and self._audio._ready:
            self._audio.speak("ECHORA shutting down.", priority=SpeechPriority.HIGH)
            time.sleep(2.0)
            self._audio.release()
        if self._interaction_detector: self._interaction_detector.release()
        if self._camera: self._camera.release()
        
        from src.hardware.haptic_feedback import get_haptic
        if h := get_haptic(): h.disconnect()
        from src.storage.database import get_db
        if db := get_db(): db.close()

        logger.info(f"ECHORA stopped. Frames: {self._frame_count} | Uptime: {time.time() - self._start_time:.1f}s | Slow:{self._slow_frames}")
        self._started = False