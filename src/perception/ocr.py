import numpy as np
import cv2
import os
import time
import threading
from collections import deque, Counter
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple

from src.core.config import settings
from src.core.utils import logger, get_timestamp_ms, depth_in_region

MIN_WORD_CONFIDENCE   = 0.4   # raised from EasyOCR's 0.3 â€” PaddleOCR is more precise
MIN_TEXT_LENGTH       = 2
MAX_SPEAK_CHARS       = 150
TEXT_STABILITY_FRAMES = 3
OCR_SCALE_FACTOR      = 1.0   # full resolution â€” GPU handles it (was 0.75 for CPU)
BBOX_PADDING          = 8
INFERENCE_CACHE_MS    = 200
STALE_CACHE_MS        = 1000
MERGE_IOU_THRESHOLD   = 0.3   # IoU above this â†’ duplicate; Arabic result wins

# â”€â”€ Preprocessing tuning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MAX_SKEW_DEG    = 15.0   # maximum tilt to auto-correct; beyond this PaddleOCR's
                          # angle classifier takes over
MIN_SKEW_DEG    = 0.8    # below this angle correction is noise â€” skip it
HOUGH_MIN_VOTES = 80     # Hough accumulator threshold; higher = fewer but more
                          # confident line detections
OVEREXPOSURE_GAMMA = 0.7  # applied when mean brightness > 200; < 1.0 darkens
                           # highlights and recovers text contrast in bright sun
BILATERAL_D        = 5    # bilateral filter neighbourhood diameter
BILATERAL_SIGMA    = 15   # colour and spatial sigma (kept low to preserve edges)


def _add_nvidia_dll_dirs():
    """Make pip-installed CUDA/cuDNN DLLs visible to PaddleOCR on Windows."""
    if os.name != "nt":
        return
    nvidia_dir = Path(os.sys.prefix) / "Lib" / "site-packages" / "nvidia"
    dll_dirs = [
        nvidia_dir / "cudnn" / "bin",
        nvidia_dir / "cublas" / "bin",
        nvidia_dir / "cuda_nvrtc" / "bin",
    ]
    existing_path = os.environ.get("PATH", "")
    prepend = []
    for dll_dir in dll_dirs:
        if dll_dir.exists():
            prepend.append(str(dll_dir))
            try:
                os.add_dll_directory(str(dll_dir))
            except (AttributeError, OSError):
                pass
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + [existing_path])


_add_nvidia_dll_dirs()


def _use_gpu() -> bool:
    if not settings.OCR_USE_GPU:
        logger.info("OCR: GPU disabled by configuration; using CPU.")
        return False
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            logger.info("OCR: CUDA-capable Paddle detected; using GPU.")
            return True
    except Exception as e:
        logger.warning(f"OCR: Paddle GPU check failed; using CPU. {e}")
    logger.info("OCR: no CUDA-capable Paddle detected; using CPU.")
    return False


def _get_ocr_gpu() -> bool:
    return _use_gpu()


def _resolve_model_dir(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = settings.BASE_DIR / path
    return str(path)


def _valid_paddle_inference_dir(path_value: str) -> bool:
    if not path_value:
        return False
    path = Path(_resolve_model_dir(path_value))
    required = ("inference.pdmodel", "inference.pdiparams")
    return path.is_dir() and all((path / name).exists() for name in required)


def _normalise_ocr_mode(mode: str) -> str:
    normalised = (mode or "both").strip().lower()
    aliases = {
        "arabic": "ar",
        "ara": "ar",
        "english": "en",
        "eng": "en",
        "both": "both",
        "all": "both",
        "ar": "ar",
        "en": "en",
    }
    if normalised not in aliases:
        logger.warning(f"Unknown OCR_MODE '{mode}'. Falling back to both.")
    return aliases.get(normalised, "both")


def _enabled_ocr_languages(mode: str) -> Tuple[bool, bool]:
    mode = _normalise_ocr_mode(mode)
    return mode in ("both", "ar"), mode in ("both", "en")


def _bbox_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    """IoU for axis-aligned bounding boxes (x1, y1, x2, y2)."""
    ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)


class OCRReader:
    """
    Reads Arabic and English text from camera frames using PaddleOCR PP-OCRv4.

    Two-model strategy:
      - Arabic model (primary)  â€” covers Arabic and mixed Arabic/Latin content.
      - English model (secondary) â€” adds Latin-only detections not found by Arabic.
    Results are IoU-deduplicated so Arabic regions are never duplicated by the
    English pass.

    Inference lock ensures only one PaddleOCR call runs at a time across
    the probe thread and the OCR-mode text thread.
    """

    def __init__(self):
        self._ocr_ar: Optional[Any] = None
        self._ocr_en: Optional[Any] = None
        self._ready:  bool = False
        self._ocr_mode: str = _normalise_ocr_mode(settings.OCR_MODE)
        self._use_arabic, self._use_english = _enabled_ocr_languages(self._ocr_mode)

        self._text_history:     deque = deque(maxlen=TEXT_STABILITY_FRAMES)
        self._last_spoken_text: str   = ""

        self._last_boxes:        List[Dict] = []
        self._last_inference_ts: float      = 0.0
        self._inference_generation: int     = 0
        self._last_history_generation: int  = -1
        self._last_dist_mm:      float      = 0.0

        self._read_count:    int   = 0
        self._success_count: int   = 0
        self._avg_ocr_ms:    float = 0.0

        self._inference_lock = threading.Lock()

    # â”€â”€ Lifecycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def load_model(self):
        logger.info("Loading PaddleOCR PP-OCRv4...")
        logger.info("  First run downloads model weights (~200 MB) automatically.")
        t0      = time.time()
        from paddleocr import PaddleOCR
        use_gpu = _use_gpu()

        _base = dict(
            use_angle_cls=True,
            use_gpu=use_gpu,
            show_log=False,
            det_db_unclip_ratio=1.6,
            drop_score=settings.OCR_CONFIDENCE_THRESHOLD,
            use_mp=False,
        )

        ar_custom_model = False
        if self._use_arabic:
            ar_kwargs = dict(_base)
            ar_kwargs["lang"] = "ar"
            ar_custom_model = _valid_paddle_inference_dir(settings.OCR_CUSTOM_AR_MODEL)
            if ar_custom_model:
                ar_kwargs["rec_model_dir"] = _resolve_model_dir(settings.OCR_CUSTOM_AR_MODEL)
                logger.info(f"  Arabic model: custom runtime model at {ar_kwargs['rec_model_dir']}")
            else:
                if settings.OCR_CUSTOM_AR_MODEL:
                    logger.warning(
                        f"  Arabic custom model not found or incomplete: "
                        f"{_resolve_model_dir(settings.OCR_CUSTOM_AR_MODEL)}. Using PaddleOCR default."
                    )
                logger.info("  Arabic model: default PaddleOCR Arabic model")

            try:
                self._ocr_ar = self._create_paddle_ocr(PaddleOCR, ar_kwargs, "Arabic")
                logger.info(f"PaddleOCR Arabic model ready in {(time.time()-t0)*1000:.0f}ms.")
            except Exception as e:
                logger.error(f"PaddleOCR Arabic model failed to load: {e}")
                raise

        if self._use_english:
            try:
                t1 = time.time()
                en_kwargs = dict(_base)
                en_kwargs["lang"] = "en"
                if _valid_paddle_inference_dir(settings.OCR_CUSTOM_EN_MODEL):
                    en_kwargs["rec_model_dir"] = _resolve_model_dir(settings.OCR_CUSTOM_EN_MODEL)
                    logger.info(f"  English model: custom runtime model at {en_kwargs['rec_model_dir']}")
                elif settings.OCR_CUSTOM_EN_MODEL:
                    logger.warning(
                        f"  English custom model not found or incomplete: "
                        f"{_resolve_model_dir(settings.OCR_CUSTOM_EN_MODEL)}. Using PaddleOCR default."
                    )
                self._ocr_en = self._create_paddle_ocr(PaddleOCR, en_kwargs, "English")
                logger.info(f"PaddleOCR English model ready in {(time.time()-t1)*1000:.0f}ms.")
            except Exception as e:
                if not self._use_arabic:
                    logger.error(f"PaddleOCR English model failed to load: {e}")
                    raise
                logger.warning(f"PaddleOCR English model failed ({e}) - Arabic-only mode.")
                self._ocr_en = None

        self._ready = self._ocr_ar is not None or self._ocr_en is not None
        logger.info(
            f"PaddleOCR ready. Total load: {(time.time()-t0)*1000:.0f}ms  "
            f"gpu={use_gpu}  mode={self._ocr_mode}  "
            f"arabic={'custom' if ar_custom_model else ('default' if self._ocr_ar else 'off')}  "
            f"english={'yes' if self._ocr_en else 'no'}"
        )

    @staticmethod
    def _create_paddle_ocr(PaddleOCR: Any, kwargs: Dict, label: str) -> Any:
        try:
            return PaddleOCR(**kwargs)
        except Exception as e:
            if kwargs.get("use_gpu"):
                logger.warning(f"{label} OCR GPU load failed ({e}); retrying on CPU.")
                cpu_kwargs = dict(kwargs)
                cpu_kwargs["use_gpu"] = False
                return PaddleOCR(**cpu_kwargs)
            raise

    def reset(self):
        self._text_history.clear()
        self._last_spoken_text = ""
        self._last_boxes       = []
        self._last_inference_ts = 0.0
        self._inference_generation = 0
        self._last_history_generation = -1

    # â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def get_text_distance(self, frame: np.ndarray, depth_map: np.ndarray) -> float:
        """
        Lightweight probe: returns depth (mm) of the nearest text region.
        Used by the state machine to decide whether to enter OCR mode.
        """
        detections = self._run_ocr_cached(frame)
        self._last_dist_mm = 0.0

        if not detections:
            return 0.0

        min_dist = float("inf")
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            d = depth_in_region(depth_map, x1, y1, x2, y2)
            if 0 < d < min_dist:
                min_dist = d

        self._last_dist_mm = 0.0 if min_dist == float("inf") else min_dist
        return self._last_dist_mm

    def read_text(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> str:
        """
        Full OCR pass: run inference (or use cache), apply stability filter,
        return the cleaned text string ready for TTS.
        Returns "" when text is not yet stable or unchanged since last call.
        """
        if not self._ready:
            return ""

        self._read_count += 1
        t0 = get_timestamp_ms()

        detections = self._run_ocr_cached(frame)

        # Never treat the same cached inference as multiple independent OCR
        # observations. Text must be confirmed by separate OCR runs.
        if self._last_history_generation == self._inference_generation:
            return ""
        self._last_history_generation = self._inference_generation

        if not detections:
            self._text_history.append("")
            return ""

        detections = self._order_for_reading(detections)
        combined  = self._clean_text([d["text"] for d in detections])

        self._text_history.append(combined)

        stable_text = self._stable_text()
        if not stable_text:
            return ""
        if stable_text == self._last_spoken_text:
            return ""

        self._last_spoken_text = stable_text
        self._success_count   += 1
        elapsed = get_timestamp_ms() - t0
        self._avg_ocr_ms = self._avg_ocr_ms * 0.8 + elapsed * 0.2
        logger.info(f"OCR confirmed: '{stable_text[:80]}'")
        return stable_text

    # â”€â”€ Core inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _run_ocr_cached(self, frame: np.ndarray) -> List[Dict]:
        """
        Returns cached inference result if younger than INFERENCE_CACHE_MS.
        Blurry frames (motion blur while walking) skip inference entirely and
        return the previous cache unchanged â€” avoids garbage results mid-stride.
        Double-checked locking prevents redundant inference when two threads race.
        """
        age = get_timestamp_ms() - self._last_inference_ts
        if age < INFERENCE_CACHE_MS:
            return self._last_boxes

        # Blur gate: check BEFORE acquiring the lock so we don't block the
        # inference thread on a frame that won't produce useful results anyway.
        if not self._is_frame_sharp_enough(frame):
            if age < STALE_CACHE_MS:
                return self._last_boxes
            self._last_boxes = []
            return []

        with self._inference_lock:
            age = get_timestamp_ms() - self._last_inference_ts
            if age < INFERENCE_CACHE_MS:
                return self._last_boxes

            detections = self._run_ocr_on_frame(frame)
            self._last_boxes        = detections
            self._last_inference_ts = get_timestamp_ms()
            self._inference_generation += 1
            return detections

    def _run_ocr_on_frame(self, frame: np.ndarray) -> List[Dict]:
        if not self._ready:
            return []
        try:
            h, w = frame.shape[:2]
            if OCR_SCALE_FACTOR != 1.0:
                frame = cv2.resize(
                    frame,
                    (int(w * OCR_SCALE_FACTOR), int(h * OCR_SCALE_FACTOR)),
                )

            preprocessed   = self._preprocess(frame)
            orig_shape     = frame.shape[:2]

            ar_dets = (
                self._run_paddle_pass(self._ocr_ar, preprocessed, orig_shape)
                if self._ocr_ar is not None else []
            )
            en_dets = (
                self._run_paddle_pass(self._ocr_en, preprocessed, orig_shape)
                if self._ocr_en is not None else []
            )

            if ar_dets and en_dets:
                return self._merge_detections(ar_dets, en_dets)
            return ar_dets or en_dets

        except Exception as e:
            logger.error(f"OCR frame error: {e}")
            return []

    def _run_paddle_pass(
        self,
        ocr:            Any,
        frame:          np.ndarray,
        original_shape: Tuple[int, int],
    ) -> List[Dict]:
        """
        One PaddleOCR inference pass.
        Normalises bbox coordinates back to original (pre-scale) dimensions,
        filters by confidence, size, and alphanumeric density.

        PaddleOCR result layout:
          result[0]  â€” first (only) page
          result[0][i] â€” [quad_bbox_points, (text, confidence)]
          quad_bbox_points â€” [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        """
        try:
            result = ocr.ocr(frame, cls=True)
            if not result or result[0] is None:
                return []

            h_orig, w_orig = original_shape
            h_inp,  w_inp  = frame.shape[:2]
            h_scale = h_orig / max(h_inp, 1)
            w_scale = w_orig / max(w_inp, 1)

            detections = []
            for line in result[0]:
                if line is None:
                    continue

                bbox_points, (text, confidence) = line

                text = text.strip()
                if not text or len(text) < MIN_TEXT_LENGTH:
                    continue
                if float(confidence) < settings.OCR_CONFIDENCE_THRESHOLD:
                    continue

                alpha = sum(1 for c in text if c.isalnum())
                if alpha / max(len(text), 1) < 0.4:
                    continue

                xs = [int(pt[0] * w_scale) for pt in bbox_points]
                ys = [int(pt[1] * h_scale) for pt in bbox_points]
                x1 = max(0,      min(xs) - BBOX_PADDING)
                y1 = max(0,      min(ys) - BBOX_PADDING)
                x2 = min(w_orig, max(xs) + BBOX_PADDING)
                y2 = min(h_orig, max(ys) + BBOX_PADDING)

                if x2 <= x1 or y2 <= y1:
                    continue
                if (y2 - y1) < settings.OCR_MIN_TEXT_HEIGHT_PX:
                    continue

                detections.append({
                    "text":       text,
                    "confidence": float(confidence),
                    "bbox":       (x1, y1, x2, y2),
                })

            return detections

        except Exception as e:
            logger.error(f"PaddleOCR pass error: {e}")
            return []

    def _merge_detections(
        self, primary: List[Dict], secondary: List[Dict]
    ) -> List[Dict]:
        """
        Merge Arabic (primary) and English (secondary) detection lists.
        A secondary detection is included only when it does not substantially
        overlap any primary detection (IoU â‰¤ MERGE_IOU_THRESHOLD).
        """
        if not secondary:
            return primary

        merged = list(primary)
        for sec in secondary:
            if not any(
                _bbox_iou(sec["bbox"], p["bbox"]) > MERGE_IOU_THRESHOLD
                for p in primary
            ):
                merged.append(sec)
        return merged

    # â”€â”€ Preprocessing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline for camera-captured text.

        Step order (each step degrades gracefully on failure):
          1. Brightness-adaptive contrast via LAB CLAHE (dark images)
             or gamma LUT (overexposed/bright-sun images)
          2. Conservative skew correction via Hough lines (if enabled)
          3. Unsharp mask sharpening for character edge crispness
          4. Bilateral denoising â€” preserves text edges unlike Gaussian blur

        PaddleOCR handles its own grayscale conversion internally, so the
        full-color BGR frame is passed through and color is preserved.
        """
        try:
            gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray))

            # â”€â”€ 1. Contrast / exposure correction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if mean_brightness < 60:
                # Very dark â€” aggressive CLAHE on L channel
                lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            elif mean_brightness < 100:
                # Moderately dark â€” lighter CLAHE
                lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            elif mean_brightness > 200:
                # Overexposed (bright outdoor sun) â€” gamma LUT to recover contrast
                lut   = np.array(
                    [int(((i / 255.0) ** (1.0 / OVEREXPOSURE_GAMMA)) * 255)
                     for i in range(256)],
                    dtype=np.uint8,
                )
                frame = cv2.LUT(frame, lut)

            # â”€â”€ 2. Skew correction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if settings.OCR_SKEW_CORRECTION:
                frame = self._correct_skew(frame)

            # â”€â”€ 3. Unsharp mask sharpening â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            blurred   = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.5)
            sharpened = cv2.addWeighted(frame, 1.3, blurred, -0.3, 0)

            # â”€â”€ 4. Edge-preserving denoising â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # bilateralFilter preserves character edges; Gaussian does not.
            denoised = cv2.bilateralFilter(
                sharpened, BILATERAL_D, BILATERAL_SIGMA, BILATERAL_SIGMA
            )
            return denoised

        except Exception as e:
            logger.warning(f"OCR preprocessing failed â€” using original: {e}")
            return frame

    def _is_frame_sharp_enough(self, frame: np.ndarray) -> bool:
        """
        Laplacian variance blur metric.
        Sharp images have high variance; motion-blurred images have low variance.
        Returns False when variance < OCR_BLUR_THRESHOLD so the caller can
        skip inference and return the last good cached result instead.
        """
        try:
            gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if variance < settings.OCR_BLUR_THRESHOLD:
                logger.debug(
                    f"OCR: frame too blurry "
                    f"(var={variance:.1f} < {settings.OCR_BLUR_THRESHOLD}) â€” skipping."
                )
                return False
            return True
        except Exception:
            return True  # on any error, don't block inference

    def _correct_skew(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect and correct image skew using Hough line transform.

        Only angles in [MIN_SKEW_DEG, MAX_SKEW_DEG] are corrected:
          - Below MIN_SKEW_DEG: correction is smaller than noise â€” skip.
          - Above MAX_SKEW_DEG: PaddleOCR's angle classifier handles it.
        Requires â‰¥ 5 agreeing lines for enough confidence to apply rotation.
        Falls back to the original frame on any error or insufficient evidence.
        """
        try:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, HOUGH_MIN_VOTES)

            if lines is None or len(lines) < 5:
                return frame

            # theta is the angle of the perpendicular to the line.
            # For a horizontal line theta = Ï€/2, so line angle = theta - Ï€/2 = 0.
            # We collect only angles within the correctable skew range.
            angles = []
            for line in lines[:40]:
                theta = float(line[0][1])
                angle = np.degrees(theta) - 90.0  # deviation from horizontal
                if MIN_SKEW_DEG <= abs(angle) <= MAX_SKEW_DEG:
                    angles.append(angle)

            if len(angles) < 5:
                return frame

            skew = float(np.median(angles))

            # Rotate by -skew to straighten: a +5Â° CCW tilt needs -5Â° (CW) correction.
            h, w    = frame.shape[:2]
            M       = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -skew, 1.0)
            rotated = cv2.warpAffine(
                frame, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            logger.debug(f"OCR skew corrected: {skew:+.1f}Â°")
            return rotated

        except Exception as e:
            logger.debug(f"OCR skew correction skipped: {e}")
            return frame

    # â”€â”€ Text processing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _stable_text(self) -> str:
        """
        Returns the most-common non-empty text in history when it appears in
        at least 2 of TEXT_STABILITY_FRAMES frames.
        More robust than requiring all frames to match identically â€” PaddleOCR,
        like EasyOCR, can produce minor character-level variance across frames.
        """
        if len(self._text_history) < TEXT_STABILITY_FRAMES:
            return ""
        non_empty = [t for t in self._text_history if t]
        if not non_empty:
            return ""
        normalised = [self._normalise_for_stability(t) for t in non_empty]
        candidates = [t for t in normalised if t]
        if not candidates:
            return ""
        most_common, count = Counter(candidates).most_common(1)[0]
        if count < 2:
            return ""
        for original in reversed(non_empty):
            if self._normalise_for_stability(original) == most_common:
                return original
        return ""

    @staticmethod
    def _normalise_for_stability(text: str) -> str:
        text = " ".join(text.lower().split())
        return "".join(c for c in text if c.isalnum() or c.isspace())

    def _clean_text(self, text_blocks: List[str]) -> str:
        cleaned = []
        for text in text_blocks:
            text = text.strip()
            if not text or len(text) < MIN_TEXT_LENGTH:
                continue
            alpha = sum(1 for c in text if c.isalnum())
            if alpha / max(len(text), 1) < 0.4:
                continue
            cleaned.append(text)

        if not cleaned:
            return ""

        # Arabic is RTL â€” reverse block order so TTS speaks in the correct sequence
        if self._is_predominantly_arabic(" ".join(cleaned)):
            cleaned = list(reversed(cleaned))

        unique   = list(dict.fromkeys(cleaned))
        combined = " ".join(unique)

        if len(combined) > MAX_SPEAK_CHARS:
            truncated  = combined[:MAX_SPEAK_CHARS]
            last_space = truncated.rfind(" ")
            combined   = (
                truncated[:last_space]
                if last_space > MAX_SPEAK_CHARS // 2
                else truncated
            )

        return combined.strip()

    @staticmethod
    def _is_predominantly_arabic(text: str) -> bool:
        """True when more than 40% of characters are in the Arabic Unicode block."""
        if not text:
            return False
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False
        arabic = sum(1 for c in letters if OCRReader._is_arabic_char(c))
        return arabic / len(letters) > 0.4

    @staticmethod
    def _is_arabic_char(char: str) -> bool:
        code = ord(char)
        return (
            0x0600 <= code <= 0x06FF or
            0x0750 <= code <= 0x077F or
            0x08A0 <= code <= 0x08FF or
            0xFB50 <= code <= 0xFDFF or
            0xFE70 <= code <= 0xFEFF
        )

    def _prioritise(
        self,
        detections: List[Dict],
        frame_width: int,
        frame_height: int,
        depth_map: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Sort by proximity, size, and optional depth."""
        cx = frame_width  // 2
        cy = frame_height // 2
        fd = (frame_width**2 + frame_height**2) ** 0.5

        def score(d: Dict) -> float:
            x1, y1, x2, y2 = d["bbox"]
            rx, ry = (x1 + x2) // 2, (y1 + y2) // 2
            ndist  = ((rx - cx)**2 + (ry - cy)**2) ** 0.5 / fd
            narea  = 1.0 - ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)
            depth_score = 0.5
            if depth_map is not None:
                depth = depth_in_region(depth_map, x1, y1, x2, y2)
                if depth > 0:
                    depth_score = min(depth / max(settings.OCR_TRIGGER_DIST_MM, 1), 1.0)
            return ndist * 0.55 + narea * 0.25 + depth_score * 0.20

        return sorted(detections, key=score)

    @staticmethod
    def _order_for_reading(detections: List[Dict]) -> List[Dict]:
        """Order English document text by line, then left-to-right in each line."""
        if len(detections) < 2:
            return list(detections)

        def center_y(det: Dict) -> float:
            _, y1, _, y2 = det["bbox"]
            return (y1 + y2) / 2.0

        def center_x(det: Dict) -> float:
            x1, _, x2, _ = det["bbox"]
            return (x1 + x2) / 2.0

        heights = sorted(
            max(1, det["bbox"][3] - det["bbox"][1]) for det in detections
        )
        median_height = heights[len(heights) // 2]
        line_tolerance = max(10.0, median_height * 0.65)

        lines: List[List[Dict]] = []
        line_centers: List[float] = []
        for det in sorted(detections, key=lambda item: (center_y(item), center_x(item))):
            y = center_y(det)
            if lines and abs(y - line_centers[-1]) <= line_tolerance:
                lines[-1].append(det)
                line_centers[-1] = sum(center_y(item) for item in lines[-1]) / len(lines[-1])
            else:
                lines.append([det])
                line_centers.append(y)

        return [
            det
            for line in lines
            for det in sorted(line, key=center_x)
        ]

    # â”€â”€ Stats â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def get_stats(self) -> Dict:
        return {
            "ready":         self._ready,
            "engine":        "PaddleOCR PP-OCRv4",
            "english_model": self._ocr_en is not None,
            "read_count":    self._read_count,
            "success_count": self._success_count,
            "avg_ocr_ms":    round(self._avg_ocr_ms, 1),
            "history_len":   len(self._text_history),
            "last_text":     self._last_spoken_text[:50],
            "last_dist_mm":  round(self._last_dist_mm, 0),
        }


# â”€â”€ Module-level singleton â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_ocr_reader: Optional[OCRReader] = None


def init_ocr():
    global _ocr_reader
    if _ocr_reader is not None:
        return
    _ocr_reader = OCRReader()
    _ocr_reader.load_model()
    logger.info("OCR module ready.")


def read_text(frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> str:
    return _ocr_reader.read_text(frame, depth_map=depth_map) if _ocr_reader else ""


def get_text_distance(frame: np.ndarray, depth_map: np.ndarray) -> float:
    return _ocr_reader.get_text_distance(frame, depth_map) if _ocr_reader else 0.0


def reset_ocr():
    if _ocr_reader:
        _ocr_reader.reset()
