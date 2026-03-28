# =============================================================================
# ocr.py — ECHORA Text Detection and Reading
# =============================================================================
# Two-stage pipeline:
#   Stage 1: Detect WHERE text is in the frame (fast, runs every frame)
#   Stage 2: Read WHAT the text says (slower, runs when text is confirmed)
#
# Calibrated for glasses-mounted camera:
#   - Prioritises text in the center of the frame (where user is facing)
#   - Confirms text across 2 frames before reading (prevents blurry reads)
#   - Triggers only when text is within OCR_TRIGGER_DIST_MM (2 metres)
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

# EasyOCR — the OCR engine that reads text from images.
# Supports 80+ languages including English and Arabic.
# More accurate than Tesseract on real-world camera images.
import easyocr

# numpy for image array operations.
import numpy as np

# cv2 for image preprocessing — grayscale, blur, threshold.
import cv2

# time for performance timing.
import time

# collections.deque — a double-ended queue we use as a fixed-size history buffer.
# deque(maxlen=N) automatically discards old items when N items are exceeded.
# Perfect for tracking text stability across the last N frames.
from collections import deque

# Type hints.
from typing import List, Dict, Optional, Tuple

# Our config and utils.
from config import (
    OCR_CONFIDENCE_THRESHOLD,
    OCR_MIN_TEXT_HEIGHT_PX,
    OCR_MAX_CHARS,
    OCR_TRIGGER_DIST_MM,
    OCR_LANGUAGE,
    DEPTH_MIN_MM,
    DEPTH_MAX_MM,
    COLLISION_CORRIDOR_DEG,
    CAMERA_HFOV_DEG,
)
from utils import logger, get_timestamp_ms, depth_in_region, bbox_center


# =============================================================================
# OCR CONFIGURATION
# =============================================================================

# Minimum confidence score from EasyOCR to accept a text detection.
# EasyOCR returns a score 0.0-1.0 for each detected text block.
# Below this = likely a misread or noise.
MIN_WORD_CONFIDENCE = 0.4

# Minimum number of characters in a text block to bother reading it.
# Single letters or very short strings are usually noise.
MIN_TEXT_LENGTH = 2

# Maximum number of characters in a single spoken announcement.
# Prevents reading an entire page of text at once.
MAX_SPEAK_CHARS = 150

# How many consecutive frames the same text must appear before we read it.
# Prevents reading blurry text that appears for just one frame as user walks.
TEXT_STABILITY_FRAMES = 2

# Scale factor to resize frame before OCR — smaller = faster but less accurate.
# 0.75 = 75% of original size. Good balance for real-time performance.
OCR_SCALE_FACTOR = 0.75

# Languages to recognise.
# "en" = English. Add "ar" for Arabic support: ["en", "ar"]
# More languages = slower loading but same inference speed.
OCR_LANGUAGES = ["en"]


# =============================================================================
# OCR READER CLASS
# =============================================================================

class OCRReader:
    """
    Detects and reads text from camera frames for ECHORA.

    Designed for glasses-mounted camera — prioritises text the user
    is directly facing, confirms text stability before reading,
    and integrates with the depth map for distance-aware triggering.

    Usage:
        reader = OCRReader()
        reader.load_model()

        while True:
            bundle = cam.get_synced_bundle()
            dist   = reader.get_text_distance(bundle["rgb"], bundle["depth"])
            # state machine uses dist to decide when to enter OCR mode

            if in_ocr_mode:
                text = reader.read_text(bundle["rgb"])
                if text:
                    audio.announce_ocr(text)
    """

    def __init__(self):
        """
        Creates the OCR reader. Does NOT load the model yet.
        Call load_model() after creating this object.
        """

        # The EasyOCR Reader object — None until load_model() runs.
        # This is the main OCR engine.
        self._reader: Optional[easyocr.Reader] = None

        # Whether the model has been loaded and is ready to use.
        self._ready: bool = False

        # ── Text stability tracking ────────────────────────────────────────────
        # deque(maxlen=TEXT_STABILITY_FRAMES) stores the last N text readings.
        # When all N readings contain the same text → text is stable → read it.
        # maxlen=2 means we keep the last 2 frame readings.
        # Old readings are automatically discarded when new ones are added.
        self._text_history: deque = deque(maxlen=TEXT_STABILITY_FRAMES)

        # The last text we successfully spoke — used to avoid repeating.
        self._last_spoken_text: str = ""

        # ── Frame-level cache ──────────────────────────────────────────────────
        # Stores the text regions detected in the most recent frame.
        # Used by get_text_distance() to avoid running detection twice.
        self._last_regions: List[Dict] = []

        # Timestamp of when _last_regions was computed.
        # If the same frame is queried multiple times, we reuse the cached result.
        self._last_detection_ts: float = 0.0

        # ── Performance tracking ───────────────────────────────────────────────
        # Total number of read_text() calls since startup.
        self._read_count: int = 0

        # Total number of successful text readings (non-empty result).
        self._success_count: int = 0

        # Rolling average OCR time in milliseconds.
        self._avg_ocr_ms: float = 0.0

        logger.info("OCRReader created. Call load_model() to start.")


    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def load_model(self):
        """
        Loads the EasyOCR model.

        First run: downloads model files (~100MB) to ~/.EasyOCR/model/
        Subsequent runs: loads from cache instantly.

        gpu=False: run on CPU.
        gpu=True:  run on GPU if available (much faster).
        We use CPU by default for compatibility — change to True on RTX PC.
        """

        logger.info(f"Loading EasyOCR model (languages: {OCR_LANGUAGES})...")
        logger.info("First run will download model files (~100MB)...")

        start = time.time()

        try:
            # easyocr.Reader(languages, gpu=False) creates the OCR engine.
            # languages: list of language codes to support.
            # gpu=False:  use CPU inference (safe default).
            #   Change to gpu=True on the RTX PC for ~10x speedup.
            # verbose=False: suppress EasyOCR's internal download progress logs.
            #   We handle our own logging.
            self._reader = easyocr.Reader(
                OCR_LANGUAGES,
                gpu     = False,
                verbose = False
            )

            elapsed = round((time.time() - start) * 1000)
            logger.info(f"EasyOCR loaded in {elapsed}ms.")

        except Exception as e:
            logger.error(f"EasyOCR failed to load: {e}")
            raise

        self._ready = True
        logger.info("OCRReader ready.")


    # =========================================================================
    # MAIN PUBLIC FUNCTIONS
    # =========================================================================

    def get_text_distance(
        self,
        frame:     np.ndarray,
        depth_map: np.ndarray
    ) -> float:
        """
        Returns the distance in mm to the nearest text region.

        Called every frame by control_unit.py to feed the state machine.
        The state machine uses this to decide when to switch to OCR mode.

        This function runs a FAST text detection (not full OCR) to find
        where text regions are, then samples the depth map at those locations.

        Returns 0.0 if no text regions are detected.

        Arguments:
            frame:     RGB frame from camera
            depth_map: depth map from camera (values in mm)
        """

        # Run text detection to find where text regions are.
        regions = self._detect_text_regions(frame)

        # Cache the regions for read_text() to reuse if called this frame.
        self._last_regions = regions
        self._last_detection_ts = get_timestamp_ms()

        if not regions:
            return 0.0

        # Find the nearest text region using the depth map.
        min_dist = float('inf')

        for region in regions:
            x1, y1, x2, y2 = region["bbox"]

            # Sample depth at the center of this text region.
            # depth_in_region returns the median depth in mm.
            dist = depth_in_region(depth_map, x1, y1, x2, y2)

            # Skip regions with no valid depth data.
            if dist <= 0:
                continue

            # Track the minimum distance across all text regions.
            if dist < min_dist:
                min_dist = dist

        # If no region had valid depth, return 0.
        if min_dist == float('inf'):
            return 0.0

        return min_dist


    def read_text(self, frame: np.ndarray) -> str:
        """
        Reads text from the frame and returns a clean spoken string.

        This is the main function called by control_unit.py when in OCR mode.

        Steps:
          1. Detect text regions (or reuse cached result from get_text_distance)
          2. Prioritise regions — center of frame first
          3. For each region: preprocess the crop, run EasyOCR
          4. Clean and combine the text results
          5. Check stability — same text in last N frames?
          6. Return the stable text ready to speak

        Returns empty string if:
          - No text detected
          - Text not stable yet (still changing between frames)
          - Same text as last time (already spoken)

        Arguments:
            frame: RGB frame from camera

        Returns:
            Clean text string ready to speak, or "" if nothing to say.
        """

        if not self._ready:
            logger.warning("OCR not ready — call load_model() first.")
            return ""

        self._read_count += 1
        start_ms = get_timestamp_ms()

        # ── Step 1: Get text regions ────────────────────────────────────────────
        # Reuse the cached result from get_text_distance() if it was computed
        # this same frame (within 50ms). Otherwise run detection fresh.
        cache_age = get_timestamp_ms() - self._last_detection_ts
        if cache_age < 50 and self._last_regions:
            # Cache is fresh — reuse it.
            regions = self._last_regions
        else:
            # Cache is stale — run fresh detection.
            regions = self._detect_text_regions(frame)

        if not regions:
            # No text found — add empty string to history.
            self._text_history.append("")
            return ""

        # ── Step 2: Prioritise regions ─────────────────────────────────────────
        # Sort regions so we read the most important text first.
        # Center-of-frame text = user is facing it = highest priority.
        h, w = frame.shape[:2]
        regions = self._prioritise_regions(regions, w, h)

        # ── Step 3: Run OCR on each region ─────────────────────────────────────
        all_text_blocks = []

        for region in regions:
            x1, y1, x2, y2 = region["bbox"]

            # Crop the frame to just this text region.
            # numpy slicing: frame[y1:y2, x1:x2]
            crop = frame[y1:y2, x1:x2]

            # Skip crops that are too small to read reliably.
            # A 10×10 pixel crop has almost no readable detail.
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            # Preprocess the crop to improve OCR accuracy.
            processed = self._preprocess_region(crop)

            # Run EasyOCR on the preprocessed crop.
            text_blocks = self._run_easyocr(processed)

            # Add all found text blocks to our collection.
            all_text_blocks.extend(text_blocks)

        # ── Step 4: Clean and combine ──────────────────────────────────────────
        # Join all text blocks into one string and clean it up.
        combined = self._clean_text(all_text_blocks)

        # ── Step 5: Check stability ────────────────────────────────────────────
        # Add this frame's text to history.
        self._text_history.append(combined)

        # Check if text is stable — same across last N frames.
        if not self._is_text_stable():
            # Text is still changing — don't speak yet.
            # Could be motion blur, user still walking, partial text.
            logger.debug(f"OCR text not stable yet: '{combined[:40]}'")
            return ""

        # ── Step 6: Check for new content ──────────────────────────────────────
        # Don't repeat the same text we already spoke.
        if combined == self._last_spoken_text:
            return ""

        # Text is stable and new — record and return it.
        if combined:
            self._last_spoken_text = combined
            self._success_count += 1

            # Update rolling average OCR time.
            elapsed = get_timestamp_ms() - start_ms
            self._avg_ocr_ms = (self._avg_ocr_ms * 0.8) + (elapsed * 0.2)

            logger.info(f"OCR result: '{combined[:80]}'")

        return combined


    # =========================================================================
    # TEXT DETECTION
    # =========================================================================

    def _detect_text_regions(self, frame: np.ndarray) -> List[Dict]:
        """
        Finds bounding boxes around text regions in the frame.

        Uses EasyOCR's detection-only mode — faster than full OCR
        because it just finds WHERE text is without reading it.

        Returns a list of region dictionaries:
        [
          {
            "bbox":       (x1, y1, x2, y2),
            "confidence": float,
            "area":       int,   # bounding box area in pixels
          },
          ...
        ]
        """

        if not self._ready:
            return []

        try:
            h, w = frame.shape[:2]

            # Resize frame for faster detection.
            # OCR_SCALE_FACTOR = 0.75 means 75% of original size.
            # new_w = int(w × 0.75), new_h = int(h × 0.75)
            new_w = int(w * OCR_SCALE_FACTOR)
            new_h = int(h * OCR_SCALE_FACTOR)
            small_frame = cv2.resize(frame, (new_w, new_h))

            # reader.detect() finds text regions without reading them.
            # Returns: (horizontal_list, free_list)
            # horizontal_list = list of axis-aligned bounding boxes
            # free_list = list of rotated bounding boxes (we ignore these)
            # paragraph=False: don't group text into paragraphs
            # min_size=20: minimum text height in pixels to detect
            # Call detect() without 'paragraph' argument — removed in newer EasyOCR versions.
            # We try the newer API first, fall back to older API if needed.
            try:
                horizontal_list, _ = self._reader.detect(
                    small_frame,
                    min_size=20,
                )
            except TypeError:
                # Older EasyOCR versions require paragraph argument.
                horizontal_list, _ = self._reader.detect(
                    small_frame,
                    paragraph=False,
                    min_size=20,
                )

            # horizontal_list is a list of bounding boxes.
            # Each box is [x_min, x_max, y_min, y_max] in the scaled frame.
            regions = []

            # horizontal_list[0] contains the list of detected regions.
            # If nothing detected it may be empty.
            detected = horizontal_list[0] if horizontal_list else []

            for box in detected:
                # EasyOCR detect() returns [x_min, x_max, y_min, y_max].
                x_min, x_max, y_min, y_max = box

                # Scale coordinates back to original frame size.
                # We resized by OCR_SCALE_FACTOR so divide by it to get back.
                scale = 1.0 / OCR_SCALE_FACTOR
                x1 = int(x_min * scale)
                y1 = int(y_min * scale)
                x2 = int(x_max * scale)
                y2 = int(y_max * scale)

                # Clamp to frame boundaries — prevent out-of-bounds slicing.
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                # Filter out regions that are too small.
                # Height < OCR_MIN_TEXT_HEIGHT_PX = too small to read.
                if (y2 - y1) < OCR_MIN_TEXT_HEIGHT_PX:
                    continue

                # Calculate area for prioritisation.
                area = (x2 - x1) * (y2 - y1)

                regions.append({
                    "bbox": (x1, y1, x2, y2),
                    "area": area,
                })

            return regions

        except Exception as e:
            logger.error(f"Text detection error: {e}")
            return []


    def _run_easyocr(self, image: np.ndarray) -> List[str]:
        """
        Runs EasyOCR on a preprocessed image crop.

        Returns a list of text strings found in the image,
        filtered by minimum confidence and minimum length.

        Arguments:
            image: preprocessed grayscale or RGB numpy array

        Returns:
            List of clean text strings.
        """

        try:
            # reader.readtext() is the full OCR function.
            # It returns a list of (bbox, text, confidence) tuples.
            # detail=1: return confidence scores (detail=0 would just return text)
            # paragraph=False: don't try to group text into paragraphs
            # batch_size=1: process one image at a time
            try:
                results = self._reader.readtext(
                    image,
                    detail=1,
                    batch_size=1,
                )
            except TypeError:
                results = self._reader.readtext(
                    image,
                    detail=1,
                    paragraph=False,
                    batch_size=1,
                )

            texts = []

            for (bbox, text, confidence) in results:
                # Skip low-confidence readings.
                # confidence < MIN_WORD_CONFIDENCE = EasyOCR isn't sure.
                if confidence < MIN_WORD_CONFIDENCE:
                    continue

                # Strip leading/trailing whitespace.
                text = text.strip()

                # Skip empty or too-short strings.
                if len(text) < MIN_TEXT_LENGTH:
                    continue

                texts.append(text)

            return texts

        except Exception as e:
            logger.error(f"EasyOCR readtext error: {e}")
            return []


    # =========================================================================
    # IMAGE PREPROCESSING
    # =========================================================================

    def _preprocess_region(self, crop: np.ndarray) -> np.ndarray:
        """
        Improves image quality before OCR to increase accuracy.

        Camera images have noise, varying lighting, and compression
        artifacts that confuse OCR engines. Preprocessing helps a lot.

        Steps:
          1. Convert to grayscale — OCR doesn't need colour
          2. Resize to standard height — consistent scale for OCR
          3. Apply CLAHE contrast enhancement — helps in low light
          4. Denoise — removes camera sensor noise
          5. Adaptive threshold — makes text black on white background

        Arguments:
            crop: numpy array (H, W, 3) BGR — the text region crop

        Returns:
            Preprocessed numpy array ready for EasyOCR.
        """

        try:
            # ── Step 1: Convert to grayscale ────────────────────────────────────
            # OCR only needs luminance (brightness), not colour.
            # Grayscale = 1 channel instead of 3 = faster processing.
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # ── Step 2: Resize to standard height ───────────────────────────────
            # EasyOCR works best with text about 32-64 pixels tall.
            # We resize crops to have a consistent height of 64 pixels,
            # keeping the aspect ratio (width scales proportionally).
            target_height = 64
            h, w = gray.shape
            if h < target_height:
                # Text is small — scale up.
                scale = target_height / h
                new_w = int(w * scale)
                new_h = target_height
                # INTER_CUBIC is the best interpolation for upscaling.
                gray = cv2.resize(
                    gray, (new_w, new_h),
                    interpolation=cv2.INTER_CUBIC
                )

            # ── Step 3: CLAHE contrast enhancement ─────────────────────────────
            # CLAHE = Contrast Limited Adaptive Histogram Equalisation.
            # It improves local contrast — makes text stand out from background.
            # This is especially useful for low-light or shadowed text.
            #
            # clipLimit=2.0: how aggressively to enhance contrast.
            #   Higher = more enhancement but also more noise.
            # tileGridSize=(8,8): divides image into 8×8 tiles for local analysis.
            #   Smaller tiles = more local contrast but slower.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            # apply() runs CLAHE on the grayscale image.
            enhanced = clahe.apply(gray)

            # ── Step 4: Denoise ─────────────────────────────────────────────────
            # Camera images have random pixel noise — salt and pepper noise,
            # Gaussian noise, JPEG compression artifacts.
            # fastNlMeansDenoising removes noise while preserving edges.
            #
            # h=10: filter strength. Higher = more denoising but blurs text.
            # templateWindowSize=7: size of area used to compare pixels.
            # searchWindowSize=21: size of area searched for similar patches.
            denoised = cv2.fastNlMeansDenoising(
                enhanced,
                h                = 10,
                templateWindowSize = 7,
                searchWindowSize   = 21,
            )

            # ── Step 5: Adaptive threshold ──────────────────────────────────────
            # Converts the grayscale image to pure black and white.
            # "Adaptive" means the threshold varies across the image
            # based on local brightness — handles uneven lighting perfectly.
            #
            # THRESH_BINARY: pixels above threshold → 255 (white)
            #                pixels below threshold → 0 (black)
            # ADAPTIVE_THRESH_GAUSSIAN_C: use Gaussian-weighted local average
            # blockSize=11: size of local neighbourhood (must be odd)
            # C=2: constant subtracted from the local average
            #   Higher C = more pixels become black = darker threshold
            thresh = cv2.adaptiveThreshold(
                denoised,
                255,                                    # max value (white)
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,                                     # block size
                2                                       # C constant
            )

            # Return the preprocessed image.
            # EasyOCR accepts both grayscale and RGB — we return grayscale.
            return thresh

        except Exception as e:
            logger.warning(f"Preprocessing failed, using original: {e}")
            # If preprocessing fails, return the original crop as-is.
            # OCR may be less accurate but will still work.
            return crop


    # =========================================================================
    # TEXT PROCESSING
    # =========================================================================

    def _clean_text(self, text_blocks: List[str]) -> str:
        """
        Cleans and combines raw OCR text blocks into a single spoken string.

        OCR engines often produce noisy output:
          - Random symbols: "@#$%&" in the middle of words
          - Duplicate lines from multiple region detections
          - Very short fragments: "a", "I", ".", "|"
          - Garbled characters: "He||o" instead of "Hello"

        This function filters and cleans all of that.

        Arguments:
            text_blocks: list of raw text strings from EasyOCR

        Returns:
            Single clean string ready to speak.
        """

        if not text_blocks:
            return ""

        cleaned_blocks = []

        for text in text_blocks:
            # Strip whitespace from both ends.
            text = text.strip()

            # Skip empty strings.
            if not text:
                continue

            # Skip strings that are too short to be meaningful.
            # A single letter or number alone is usually noise.
            if len(text) < MIN_TEXT_LENGTH:
                continue

            # Skip strings that are mostly symbols/punctuation.
            # Count alphanumeric characters.
            # sum() counts True values, which Python treats as 1.
            # c.isalnum() returns True if c is a letter or digit.
            alpha_count = sum(1 for c in text if c.isalnum())

            # If less than 50% of characters are alphanumeric, skip.
            # "###@@@" is noise. "Hello!" is real text (83% alphanumeric).
            if len(text) > 0 and alpha_count / len(text) < 0.5:
                continue

            cleaned_blocks.append(text)

        if not cleaned_blocks:
            return ""

        # Remove duplicate lines — sometimes the same text region
        # is detected twice by overlapping bounding boxes.
        # dict.fromkeys() preserves order while removing duplicates.
        # (Regular sets don't preserve order.)
        unique_blocks = list(dict.fromkeys(cleaned_blocks))

        # Join all unique blocks with a space between them.
        combined = " ".join(unique_blocks)

        # Truncate to maximum speakable length.
        # We don't want Karen reading a 500-word paragraph.
        if len(combined) > MAX_SPEAK_CHARS:
            # Truncate at the last space before the limit.
            # This avoids cutting a word in half.
            truncated = combined[:MAX_SPEAK_CHARS]
            last_space = truncated.rfind(" ")

            if last_space > MAX_SPEAK_CHARS // 2:
                # There's a space in a reasonable position — cut there.
                combined = truncated[:last_space]
            else:
                # No good space found — just truncate.
                combined = truncated

        return combined.strip()


    def _is_text_stable(self) -> bool:
        """
        Returns True if the same text appeared in all recent frames.

        We require TEXT_STABILITY_FRAMES consecutive frames with the
        same text before speaking. This prevents:
          - Reading blurry text mid-stride
          - Reading partial text as the user walks toward a sign
          - Announcing garbled text from a single bad frame

        How it works:
          Frame 1: "Emergency Ex"  → history = ["Emergency Ex"]
          Frame 2: "Emergency Exi" → history = ["Emergency Ex", "Emergency Exi"]
          Still changing → return False

          Frame 1: "Emergency Exit" → history = ["Emergency Exit"]
          Frame 2: "Emergency Exit" → history = ["Emergency Exit", "Emergency Exit"]
          Same text in all frames → return True → speak it
        """

        # Not enough history yet — need at least TEXT_STABILITY_FRAMES frames.
        if len(self._text_history) < TEXT_STABILITY_FRAMES:
            return False

        # Get all entries in the history deque.
        history_list = list(self._text_history)

        # Check if all entries are the same non-empty string.
        # set() removes duplicates — if all values are the same,
        # the set will have exactly one element.
        unique_texts = set(history_list)

        # Stable if: exactly one unique text AND it's not empty.
        return (
            len(unique_texts) == 1
            and list(unique_texts)[0] != ""
        )


    def _prioritise_regions(
        self,
        regions:      List[Dict],
        frame_width:  int,
        frame_height: int
    ) -> List[Dict]:
        """
        Sorts text regions by priority for reading.

        Priority rules (highest to lowest):
          1. Center of frame — user is directly facing this text
          2. Larger text — more likely to be a sign or important label
          3. Higher in frame — signs are usually above eye level

        Arguments:
            regions:      list of region dicts from _detect_text_regions()
            frame_width:  width of the original frame in pixels
            frame_height: height of the original frame in pixels

        Returns:
            Sorted list — highest priority first.
        """

        # Calculate the center of the frame.
        # This is where the user is directly facing (0° angle).
        frame_cx = frame_width  // 2
        frame_cy = frame_height // 2

        def priority_score(region: Dict) -> float:
            """
            Calculates a single priority score for one region.
            Lower score = higher priority (sorted ascending).
            """

            x1, y1, x2, y2 = region["bbox"]

            # Center of this text region.
            region_cx = (x1 + x2) // 2
            region_cy = (y1 + y2) // 2

            # Distance from frame center — closer to center = higher priority.
            # Euclidean distance: sqrt((cx2-cx1)² + (cy2-cy1)²)
            # We normalise by dividing by frame diagonal so the score
            # is between 0.0 (center) and 1.0 (corner).
            frame_diag = (frame_width**2 + frame_height**2) ** 0.5
            center_dist = (
                (region_cx - frame_cx)**2 + (region_cy - frame_cy)**2
            ) ** 0.5
            normalised_dist = center_dist / frame_diag

            # Area score — larger text = more important.
            # We invert and normalise: small text → high score (low priority)
            # Frame area is the maximum possible region area.
            frame_area = frame_width * frame_height
            area_score = 1.0 - (region["area"] / frame_area)

            # Combine: 70% center distance + 30% area.
            # Center distance matters more because the user is facing it.
            return (normalised_dist * 0.7) + (area_score * 0.3)

        # sorted() with key= sorts by the priority_score of each region.
        # Smallest score = closest to center and largest text = first.
        return sorted(regions, key=priority_score)


    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def reset(self):
        """
        Resets the OCR reader state.

        Called by state machine when exiting OCR mode.
        Clears the text stability buffer so when we re-enter OCR mode,
        the first stable text will be spoken even if it is the same
        as what was spoken before entering the last OCR session.
        """

        self._text_history.clear()
        self._last_spoken_text = ""
        self._last_regions     = []
        logger.debug("OCRReader state reset.")


    def get_stats(self) -> Dict:
        """Returns diagnostic statistics about the OCR reader."""
        return {
            "ready":         self._ready,
            "read_count":    self._read_count,
            "success_count": self._success_count,
            "avg_ocr_ms":    round(self._avg_ocr_ms, 1),
            "history_len":   len(self._text_history),
            "last_text":     self._last_spoken_text[:50],
        }


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================
# Create one shared OCRReader instance at module level.
# All callers share this single instance — the model is only loaded once.
#
# This is the SINGLETON pattern:
#   - First import of ocr.py creates _ocr_reader = OCRReader()
#   - Every subsequent import reuses the same object
#   - No matter how many files import ocr, there is only one reader

# The shared instance — None until init_ocr() is called.
_ocr_reader: Optional[OCRReader] = None


def init_ocr():
    """
    Initialises the module-level OCR reader.

    Call this once at startup from control_unit.py before using
    read_text() or get_text_distance().
    """

    global _ocr_reader

    if _ocr_reader is not None:
        logger.debug("OCR already initialised.")
        return

    _ocr_reader = OCRReader()
    _ocr_reader.load_model()
    logger.info("Module-level OCR reader ready.")


def read_text(frame: np.ndarray) -> str:
    """
    Module-level function — reads text from a frame.

    Called by control_unit.py when in OCR mode.
    Delegates to the singleton OCRReader instance.

    Returns empty string if OCR not initialised or no text found.
    """

    if _ocr_reader is None:
        logger.warning("OCR not initialised. Call init_ocr() first.")
        return ""

    return _ocr_reader.read_text(frame)


def get_text_distance(
    frame:     np.ndarray,
    depth_map: np.ndarray
) -> float:
    """
    Module-level function — returns distance to nearest text in mm.

    Called every frame by control_unit.py to feed the state machine.
    Returns 0.0 if OCR not initialised or no text found.
    """

    if _ocr_reader is None:
        return 0.0

    return _ocr_reader.get_text_distance(frame, depth_map)


def reset_ocr():
    """Module-level reset — called when exiting OCR mode."""

    if _ocr_reader is not None:
        _ocr_reader.reset()


# =============================================================================
# SELF-TEST
# =============================================================================
# Tests OCR with live camera.
# Run with: python ocr.py

if __name__ == "__main__":

    print("=== ECHORA ocr.py self-test ===\n")

    from camera import EchoraCamera

    # ── Initialise ─────────────────────────────────────────────────────────────
    cam    = EchoraCamera()
    reader = OCRReader()

    try:
        print("Loading camera and OCR model...")
        cam.init_pipeline()
        reader.load_model()
        print("Ready. Point the camera at text (signs, labels, books).")
        print("Press Q to quit.\n")

        frame_count  = 0
        last_text    = ""

        while True:
            bundle = cam.get_synced_bundle()
            if bundle is None:
                continue

            frame_count += 1
            rgb   = bundle["rgb"]
            depth = bundle["depth"]

            # ── Get text distance ────────────────────────────────────────────
            text_dist = reader.get_text_distance(rgb, depth)

            # ── Try to read text ─────────────────────────────────────────────
            # Only run full OCR when text is close enough.
            text = ""
            if text_dist > 0 and text_dist < OCR_TRIGGER_DIST_MM:
                text = reader.read_text(rgb)

            # ── Print results every 10 frames ────────────────────────────────
            if frame_count % 10 == 0:
                print(
                    f"Frame {frame_count:4d} | "
                    f"Text dist: {text_dist:6.0f}mm | "
                    f"Regions: {len(reader._last_regions):2d}",
                    end=""
                )
                if text:
                    print(f" | READ: '{text}'")
                else:
                    print()

            # ── Draw debug overlay ───────────────────────────────────────────
            debug = rgb.copy()

            # Draw detected text region bounding boxes in yellow.
            for region in reader._last_regions:
                x1, y1, x2, y2 = region["bbox"]
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Show text distance in top-left.
            dist_str = f"Text: {text_dist:.0f}mm" if text_dist > 0 else "No text"
            cv2.putText(
                debug, dist_str,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2
            )

            # Show last read text in green at bottom.
            if text:
                last_text = text
            if last_text:
                cv2.putText(
                    debug,
                    f"'{last_text[:60]}'",
                    (10, debug.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

            cv2.imshow("ECHORA OCR Test", debug)
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        cv2.destroyAllWindows()
        cam.release()
        stats = reader.get_stats()
        print(f"\nFinal stats: {stats}")
        print("\n=== Self-test complete ===")