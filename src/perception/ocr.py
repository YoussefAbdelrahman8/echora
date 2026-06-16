# import numpy as np
# import cv2
# import time
# import threading
# import torch
# from collections import deque, Counter
# from typing import List, Dict, Optional, Tuple
#
# from paddleocr import PaddleOCR
#
# from src.core.config import settings
# from src.core.utils import logger, get_timestamp_ms, depth_in_region
#
# MIN_WORD_CONFIDENCE   = 0.4   # raised from EasyOCR's 0.3 — PaddleOCR is more precise
# MIN_TEXT_LENGTH       = 2
# MAX_SPEAK_CHARS       = 150
# TEXT_STABILITY_FRAMES = 3
# OCR_SCALE_FACTOR      = 1.0   # full resolution — GPU handles it (was 0.75 for CPU)
# BBOX_PADDING          = 8
# INFERENCE_CACHE_MS    = 200
# MERGE_IOU_THRESHOLD   = 0.3   # IoU above this → duplicate; Arabic result wins
#
# # ── Preprocessing tuning ───────────────────────────────────────────────
# MAX_SKEW_DEG    = 15.0   # maximum tilt to auto-correct; beyond this PaddleOCR's
#                           # angle classifier takes over
# MIN_SKEW_DEG    = 0.8    # below this angle correction is noise — skip it
# HOUGH_MIN_VOTES = 80     # Hough accumulator threshold; higher = fewer but more
#                           # confident line detections
# OVEREXPOSURE_GAMMA = 0.7  # applied when mean brightness > 200; < 1.0 darkens
#                            # highlights and recovers text contrast in bright sun
# BILATERAL_D        = 5    # bilateral filter neighbourhood diameter
# BILATERAL_SIGMA    = 15   # colour and spatial sigma (kept low to preserve edges)
#
#
# def _use_gpu() -> bool:
#     if torch.cuda.is_available():
#         logger.info("OCR: CUDA GPU detected — using GPU.")
#         return True
#     logger.info("OCR: no CUDA GPU — using CPU.")
#     return False
#
#
# def _bbox_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
#     """IoU for axis-aligned bounding boxes (x1, y1, x2, y2)."""
#     ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
#     ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
#     inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
#     if inter == 0:
#         return 0.0
#     a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
#     a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
#     return inter / (a1 + a2 - inter)
#
#
# class OCRReader:
#     """
#     Reads Arabic and English text from camera frames using PaddleOCR PP-OCRv4.
#
#     Two-model strategy:
#       - Arabic model (primary)  — covers Arabic and mixed Arabic/Latin content.
#       - English model (secondary) — adds Latin-only detections not found by Arabic.
#     Results are IoU-deduplicated so Arabic regions are never duplicated by the
#     English pass.
#
#     Inference lock ensures only one PaddleOCR call runs at a time across
#     the probe thread and the OCR-mode text thread.
#     """
#
#     def __init__(self):
#         self._ocr_ar: Optional[PaddleOCR] = None
#         self._ocr_en: Optional[PaddleOCR] = None
#         self._ready:  bool = False
#         self._use_english: bool = 'en' in settings.OCR_LANGUAGE
#
#         self._text_history:     deque = deque(maxlen=TEXT_STABILITY_FRAMES)
#         self._last_spoken_text: str   = ""
#
#         self._last_boxes:        List[Dict] = []
#         self._last_inference_ts: float      = 0.0
#         self._last_dist_mm:      float      = 0.0
#
#         self._read_count:    int   = 0
#         self._success_count: int   = 0
#         self._avg_ocr_ms:    float = 0.0
#
#         self._inference_lock = threading.Lock()
#
#     # ── Lifecycle ──────────────────────────────────────────────────────
#
#     def load_model(self):
#         logger.info("Loading PaddleOCR PP-OCRv4...")
#         logger.info("  First run downloads model weights (~200 MB) automatically.")
#         t0      = time.time()
#         use_gpu = _use_gpu()
#
#         # Base kwargs shared by both models
#         _base = dict(
#             use_angle_cls    = True,
#             use_gpu          = use_gpu,
#             show_log         = False,
#             det_db_unclip_ratio = 1.6,
#             drop_score       = MIN_WORD_CONFIDENCE,
#             use_mp           = False,
#         )
#
#         # Arabic model — use fine-tuned weights if configured, else default PP-OCRv4
#         ar_kwargs = dict(_base)
#         ar_kwargs["lang"] = "ar"
#         if settings.OCR_CUSTOM_AR_MODEL:
#             ar_kwargs["rec_model_dir"] = settings.OCR_CUSTOM_AR_MODEL
#             logger.info(f"  Arabic model: fine-tuned at {settings.OCR_CUSTOM_AR_MODEL}")
#         else:
#             logger.info("  Arabic model: default PP-OCRv4")
#
#         try:
#             self._ocr_ar = PaddleOCR(**ar_kwargs)
#             logger.info(f"PaddleOCR Arabic model ready in {(time.time()-t0)*1000:.0f}ms.")
#         except Exception as e:
#             logger.error(f"PaddleOCR Arabic model failed to load: {e}")
#             raise
#
#         # English model — optional, use fine-tuned weights if configured
#         if self._use_english:
#             en_kwargs = dict(_base)
#             en_kwargs["lang"] = "en"
#             if settings.OCR_CUSTOM_EN_MODEL:
#                 en_kwargs["rec_model_dir"] = settings.OCR_CUSTOM_EN_MODEL
#                 logger.info(f"  English model: fine-tuned at {settings.OCR_CUSTOM_EN_MODEL}")
#             try:
#                 t1 = time.time()
#                 self._ocr_en = PaddleOCR(**en_kwargs)
#                 logger.info(f"PaddleOCR English model ready in {(time.time()-t1)*1000:.0f}ms.")
#             except Exception as e:
#                 logger.warning(f"PaddleOCR English model failed ({e}) — Arabic-only mode.")
#                 self._ocr_en = None
#
#         self._ready = True
#         ft_tag = "fine-tuned" if settings.OCR_CUSTOM_AR_MODEL else "default"
#         logger.info(
#             f"PaddleOCR ready. Total load: {(time.time()-t0)*1000:.0f}ms  "
#             f"gpu={use_gpu}  arabic={ft_tag}  english={'yes' if self._ocr_en else 'no'}"
#         )
#
#     def reset(self):
#         self._text_history.clear()
#         self._last_spoken_text = ""
#         self._last_boxes       = []
#         self._last_inference_ts = 0.0
#
#     # ── Public API ─────────────────────────────────────────────────────
#
#     def get_text_distance(self, frame: np.ndarray, depth_map: np.ndarray) -> float:
#         """
#         Lightweight probe: returns depth (mm) of the nearest text region.
#         Used by the state machine to decide whether to enter OCR mode.
#         """
#         detections = self._run_ocr_cached(frame)
#         self._last_dist_mm = 0.0
#
#         if not detections:
#             return 0.0
#
#         min_dist = float("inf")
#         for det in detections:
#             x1, y1, x2, y2 = det["bbox"]
#             d = depth_in_region(depth_map, x1, y1, x2, y2)
#             if 0 < d < min_dist:
#                 min_dist = d
#
#         self._last_dist_mm = 0.0 if min_dist == float("inf") else min_dist
#         return self._last_dist_mm
#
#     def read_text(self, frame: np.ndarray) -> str:
#         """
#         Full OCR pass: run inference (or use cache), apply stability filter,
#         return the cleaned text string ready for TTS.
#         Returns "" when text is not yet stable or unchanged since last call.
#         """
#         if not self._ready:
#             return ""
#
#         self._read_count += 1
#         t0 = get_timestamp_ms()
#
#         detections = self._run_ocr_cached(frame)
#
#         if not detections:
#             self._text_history.append("")
#             return ""
#
#         h, w      = frame.shape[:2]
#         detections = self._prioritise(detections, w, h)
#         combined  = self._clean_text([d["text"] for d in detections])
#
#         self._text_history.append(combined)
#
#         stable_text = self._stable_text()
#         if not stable_text:
#             return ""
#         if stable_text == self._last_spoken_text:
#             return ""
#
#         self._last_spoken_text = stable_text
#         self._success_count   += 1
#         elapsed = get_timestamp_ms() - t0
#         self._avg_ocr_ms = self._avg_ocr_ms * 0.8 + elapsed * 0.2
#         logger.info(f"OCR confirmed: '{stable_text[:80]}'")
#         return stable_text
#
#     # ── Core inference ─────────────────────────────────────────────────
#
#     def _run_ocr_cached(self, frame: np.ndarray) -> List[Dict]:
#         """
#         Returns cached inference result if younger than INFERENCE_CACHE_MS.
#         Blurry frames (motion blur while walking) skip inference entirely and
#         return the previous cache unchanged — avoids garbage results mid-stride.
#         Double-checked locking prevents redundant inference when two threads race.
#         """
#         age = get_timestamp_ms() - self._last_inference_ts
#         if age < INFERENCE_CACHE_MS and self._last_boxes:
#             return self._last_boxes
#
#         # Blur gate: check BEFORE acquiring the lock so we don't block the
#         # inference thread on a frame that won't produce useful results anyway.
#         if not self._is_frame_sharp_enough(frame):
#             return self._last_boxes
#
#         with self._inference_lock:
#             age = get_timestamp_ms() - self._last_inference_ts
#             if age < INFERENCE_CACHE_MS and self._last_boxes:
#                 return self._last_boxes
#
#             detections = self._run_ocr_on_frame(frame)
#             self._last_boxes        = detections
#             self._last_inference_ts = get_timestamp_ms()
#             return detections
#
#     def _run_ocr_on_frame(self, frame: np.ndarray) -> List[Dict]:
#         if not self._ready or self._ocr_ar is None:
#             return []
#         try:
#             h, w = frame.shape[:2]
#             if OCR_SCALE_FACTOR != 1.0:
#                 frame = cv2.resize(
#                     frame,
#                     (int(w * OCR_SCALE_FACTOR), int(h * OCR_SCALE_FACTOR)),
#                 )
#
#             preprocessed   = self._preprocess(frame)
#             orig_shape     = frame.shape[:2]
#
#             ar_dets = self._run_paddle_pass(self._ocr_ar, preprocessed, orig_shape)
#
#             if self._ocr_en is not None:
#                 en_dets = self._run_paddle_pass(self._ocr_en, preprocessed, orig_shape)
#                 return self._merge_detections(ar_dets, en_dets)
#
#             return ar_dets
#
#         except Exception as e:
#             logger.error(f"OCR frame error: {e}")
#             return []
#
#     def _run_paddle_pass(
#         self,
#         ocr:            PaddleOCR,
#         frame:          np.ndarray,
#         original_shape: Tuple[int, int],
#     ) -> List[Dict]:
#         """
#         One PaddleOCR inference pass.
#         Normalises bbox coordinates back to original (pre-scale) dimensions,
#         filters by confidence, size, and alphanumeric density.
#
#         PaddleOCR result layout:
#           result[0]  — first (only) page
#           result[0][i] — [quad_bbox_points, (text, confidence)]
#           quad_bbox_points — [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
#         """
#         try:
#             result = ocr.ocr(frame, cls=True)
#             if not result or result[0] is None:
#                 return []
#
#             h_orig, w_orig = original_shape
#             h_inp,  w_inp  = frame.shape[:2]
#             h_scale = h_orig / max(h_inp, 1)
#             w_scale = w_orig / max(w_inp, 1)
#
#             detections = []
#             for line in result[0]:
#                 if line is None:
#                     continue
#
#                 bbox_points, (text, confidence) = line
#
#                 text = text.strip()
#                 if not text or len(text) < MIN_TEXT_LENGTH:
#                     continue
#
#                 alpha = sum(1 for c in text if c.isalnum())
#                 if alpha / max(len(text), 1) < 0.4:
#                     continue
#
#                 xs = [int(pt[0] * w_scale) for pt in bbox_points]
#                 ys = [int(pt[1] * h_scale) for pt in bbox_points]
#                 x1 = max(0,      min(xs) - BBOX_PADDING)
#                 y1 = max(0,      min(ys) - BBOX_PADDING)
#                 x2 = min(w_orig, max(xs) + BBOX_PADDING)
#                 y2 = min(h_orig, max(ys) + BBOX_PADDING)
#
#                 if x2 <= x1 or y2 <= y1:
#                     continue
#                 if (y2 - y1) < settings.OCR_MIN_TEXT_HEIGHT_PX:
#                     continue
#
#                 detections.append({
#                     "text":       text,
#                     "confidence": float(confidence),
#                     "bbox":       (x1, y1, x2, y2),
#                 })
#
#             return detections
#
#         except Exception as e:
#             logger.error(f"PaddleOCR pass error: {e}")
#             return []
#
#     def _merge_detections(
#         self, primary: List[Dict], secondary: List[Dict]
#     ) -> List[Dict]:
#         """
#         Merge Arabic (primary) and English (secondary) detection lists.
#         A secondary detection is included only when it does not substantially
#         overlap any primary detection (IoU ≤ MERGE_IOU_THRESHOLD).
#         """
#         if not secondary:
#             return primary
#
#         merged = list(primary)
#         for sec in secondary:
#             if not any(
#                 _bbox_iou(sec["bbox"], p["bbox"]) > MERGE_IOU_THRESHOLD
#                 for p in primary
#             ):
#                 merged.append(sec)
#         return merged
#
#     # ── Preprocessing ──────────────────────────────────────────────────
#
#     def _preprocess(self, frame: np.ndarray) -> np.ndarray:
#         """
#         Full preprocessing pipeline for camera-captured text.
#
#         Step order (each step degrades gracefully on failure):
#           1. Brightness-adaptive contrast via LAB CLAHE (dark images)
#              or gamma LUT (overexposed/bright-sun images)
#           2. Conservative skew correction via Hough lines (if enabled)
#           3. Unsharp mask sharpening for character edge crispness
#           4. Bilateral denoising — preserves text edges unlike Gaussian blur
#
#         PaddleOCR handles its own grayscale conversion internally, so the
#         full-color BGR frame is passed through and color is preserved.
#         """
#         try:
#             gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             mean_brightness = float(np.mean(gray))
#
#             # ── 1. Contrast / exposure correction ─────────────────────
#             if mean_brightness < 60:
#                 # Very dark — aggressive CLAHE on L channel
#                 lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#                 clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
#                 lab[:, :, 0] = clahe.apply(lab[:, :, 0])
#                 frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#
#             elif mean_brightness < 100:
#                 # Moderately dark — lighter CLAHE
#                 lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#                 clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
#                 lab[:, :, 0] = clahe.apply(lab[:, :, 0])
#                 frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#
#             elif mean_brightness > 200:
#                 # Overexposed (bright outdoor sun) — gamma LUT to recover contrast
#                 lut   = np.array(
#                     [int(((i / 255.0) ** (1.0 / OVEREXPOSURE_GAMMA)) * 255)
#                      for i in range(256)],
#                     dtype=np.uint8,
#                 )
#                 frame = cv2.LUT(frame, lut)
#
#             # ── 2. Skew correction ─────────────────────────────────────
#             if settings.OCR_SKEW_CORRECTION:
#                 frame = self._correct_skew(frame)
#
#             # ── 3. Unsharp mask sharpening ────────────────────────────
#             blurred   = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.5)
#             sharpened = cv2.addWeighted(frame, 1.3, blurred, -0.3, 0)
#
#             # ── 4. Edge-preserving denoising ──────────────────────────
#             # bilateralFilter preserves character edges; Gaussian does not.
#             denoised = cv2.bilateralFilter(
#                 sharpened, BILATERAL_D, BILATERAL_SIGMA, BILATERAL_SIGMA
#             )
#             return denoised
#
#         except Exception as e:
#             logger.warning(f"OCR preprocessing failed — using original: {e}")
#             return frame
#
#     def _is_frame_sharp_enough(self, frame: np.ndarray) -> bool:
#         """
#         Laplacian variance blur metric.
#         Sharp images have high variance; motion-blurred images have low variance.
#         Returns False when variance < OCR_BLUR_THRESHOLD so the caller can
#         skip inference and return the last good cached result instead.
#         """
#         try:
#             gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             variance = cv2.Laplacian(gray, cv2.CV_64F).var()
#             if variance < settings.OCR_BLUR_THRESHOLD:
#                 logger.debug(
#                     f"OCR: frame too blurry "
#                     f"(var={variance:.1f} < {settings.OCR_BLUR_THRESHOLD}) — skipping."
#                 )
#                 return False
#             return True
#         except Exception:
#             return True  # on any error, don't block inference
#
#     def _correct_skew(self, frame: np.ndarray) -> np.ndarray:
#         """
#         Detect and correct image skew using Hough line transform.
#
#         Only angles in [MIN_SKEW_DEG, MAX_SKEW_DEG] are corrected:
#           - Below MIN_SKEW_DEG: correction is smaller than noise — skip.
#           - Above MAX_SKEW_DEG: PaddleOCR's angle classifier handles it.
#         Requires ≥ 5 agreeing lines for enough confidence to apply rotation.
#         Falls back to the original frame on any error or insufficient evidence.
#         """
#         try:
#             gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#             lines = cv2.HoughLines(edges, 1, np.pi / 180, HOUGH_MIN_VOTES)
#
#             if lines is None or len(lines) < 5:
#                 return frame
#
#             # theta is the angle of the perpendicular to the line.
#             # For a horizontal line theta = π/2, so line angle = theta - π/2 = 0.
#             # We collect only angles within the correctable skew range.
#             angles = []
#             for line in lines[:40]:
#                 theta = float(line[0][1])
#                 angle = np.degrees(theta) - 90.0  # deviation from horizontal
#                 if MIN_SKEW_DEG <= abs(angle) <= MAX_SKEW_DEG:
#                     angles.append(angle)
#
#             if len(angles) < 5:
#                 return frame
#
#             skew = float(np.median(angles))
#
#             # Rotate by -skew to straighten: a +5° CCW tilt needs -5° (CW) correction.
#             h, w    = frame.shape[:2]
#             M       = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -skew, 1.0)
#             rotated = cv2.warpAffine(
#                 frame, M, (w, h),
#                 flags=cv2.INTER_CUBIC,
#                 borderMode=cv2.BORDER_REPLICATE,
#             )
#             logger.debug(f"OCR skew corrected: {skew:+.1f}°")
#             return rotated
#
#         except Exception as e:
#             logger.debug(f"OCR skew correction skipped: {e}")
#             return frame
#
#     # ── Text processing ────────────────────────────────────────────────
#
#     def _stable_text(self) -> str:
#         """
#         Returns the most-common non-empty text in history when it appears in
#         at least 2 of TEXT_STABILITY_FRAMES frames.
#         More robust than requiring all frames to match identically — PaddleOCR,
#         like EasyOCR, can produce minor character-level variance across frames.
#         """
#         if len(self._text_history) < TEXT_STABILITY_FRAMES:
#             return ""
#         non_empty = [t for t in self._text_history if t]
#         if not non_empty:
#             return ""
#         most_common, count = Counter(non_empty).most_common(1)[0]
#         return most_common if count >= 2 else ""
#
#     def _clean_text(self, text_blocks: List[str]) -> str:
#         cleaned = []
#         for text in text_blocks:
#             text = text.strip()
#             if not text or len(text) < MIN_TEXT_LENGTH:
#                 continue
#             alpha = sum(1 for c in text if c.isalnum())
#             if alpha / max(len(text), 1) < 0.4:
#                 continue
#             cleaned.append(text)
#
#         if not cleaned:
#             return ""
#
#         # Arabic is RTL — reverse block order so TTS speaks in the correct sequence
#         if self._is_predominantly_arabic(" ".join(cleaned)):
#             cleaned = list(reversed(cleaned))
#
#         unique   = list(dict.fromkeys(cleaned))
#         combined = " ".join(unique)
#
#         if len(combined) > MAX_SPEAK_CHARS:
#             truncated  = combined[:MAX_SPEAK_CHARS]
#             last_space = truncated.rfind(" ")
#             combined   = (
#                 truncated[:last_space]
#                 if last_space > MAX_SPEAK_CHARS // 2
#                 else truncated
#             )
#
#         return combined.strip()
#
#     @staticmethod
#     def _is_predominantly_arabic(text: str) -> bool:
#         """True when more than 40% of characters are in the Arabic Unicode block."""
#         if not text:
#             return False
#         arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
#         return arabic / max(len(text), 1) > 0.4
#
#     def _prioritise(
#         self, detections: List[Dict], frame_width: int, frame_height: int
#     ) -> List[Dict]:
#         """Sort by proximity to frame centre — the text the user is aiming at."""
#         cx = frame_width  // 2
#         cy = frame_height // 2
#         fd = (frame_width**2 + frame_height**2) ** 0.5
#
#         def score(d: Dict) -> float:
#             x1, y1, x2, y2 = d["bbox"]
#             rx, ry = (x1 + x2) // 2, (y1 + y2) // 2
#             ndist  = ((rx - cx)**2 + (ry - cy)**2) ** 0.5 / fd
#             narea  = 1.0 - ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)
#             return ndist * 0.7 + narea * 0.3
#
#         return sorted(detections, key=score)
#
#     # ── Stats ──────────────────────────────────────────────────────────
#
#     def get_stats(self) -> Dict:
#         return {
#             "ready":         self._ready,
#             "engine":        "PaddleOCR PP-OCRv4",
#             "english_model": self._ocr_en is not None,
#             "read_count":    self._read_count,
#             "success_count": self._success_count,
#             "avg_ocr_ms":    round(self._avg_ocr_ms, 1),
#             "history_len":   len(self._text_history),
#             "last_text":     self._last_spoken_text[:50],
#             "last_dist_mm":  round(self._last_dist_mm, 0),
#         }
#
#
# # ── Module-level singleton ─────────────────────────────────────────────
#
# _ocr_reader: Optional[OCRReader] = None
#
#
# def init_ocr():
#     global _ocr_reader
#     if _ocr_reader is not None:
#         return
#     _ocr_reader = OCRReader()
#     _ocr_reader.load_model()
#     logger.info("OCR module ready.")
#
#
# def read_text(frame: np.ndarray) -> str:
#     return _ocr_reader.read_text(frame) if _ocr_reader else ""
#
#
# def get_text_distance(frame: np.ndarray, depth_map: np.ndarray) -> float:
#     return _ocr_reader.get_text_distance(frame, depth_map) if _ocr_reader else 0.0
#
#
# def reset_ocr():
#     if _ocr_reader:
#         _ocr_reader.reset()
#