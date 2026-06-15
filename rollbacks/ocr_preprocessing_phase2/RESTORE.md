# Rollback — OCR Phase 2: Preprocessing Improvements
Saved: 2026-06-05

## Files Changed
| File | What changed |
|------|-------------|
| `src/perception/ocr.py` | New blur gate, skew correction, overexposure handling, bilateral denoising |
| `src/core/config.py` | Added OCR_BLUR_THRESHOLD, OCR_SKEW_CORRECTION |

## To Restore
```bash
cp rollbacks/ocr_preprocessing_phase2/original_ocr.py     src/perception/ocr.py
cp rollbacks/ocr_preprocessing_phase2/original_config.py  src/core/config.py
```

## What Each Change Does

### config.py
- `OCR_BLUR_THRESHOLD = 60.0` — Laplacian variance minimum. Frames below this are
  skipped for OCR inference entirely (previous cache returned unchanged). Tune higher
  if you get false passes (blurry frames that slip through), lower if sharp frames
  are falsely rejected.
- `OCR_SKEW_CORRECTION = True` — enable/disable the Hough-based skew correction step.
  Set to False if it causes regressions on your specific environment.

### ocr.py — new constants
- `MAX_SKEW_DEG = 15.0` — max tilt angle to auto-correct (beyond this, PaddleOCR's
  angle classifier handles it).
- `MIN_SKEW_DEG = 0.8` — minimum tilt to bother correcting (below this = noise).
- `HOUGH_MIN_VOTES = 80` — Hough accumulator threshold; higher = fewer, more
  confident line detections.

### ocr.py — new methods
- `_is_frame_sharp_enough(frame)` — Laplacian variance blur metric. Returns False
  when the frame is too blurry to produce reliable OCR (motion blur while walking).
- `_correct_skew(frame)` — detects dominant horizontal angle via Hough lines and
  rotates by the inverse. Only fires between MIN_SKEW_DEG and MAX_SKEW_DEG, needs
  ≥ 5 agreeing lines. Falls back to original frame on any error.

### ocr.py — updated _run_ocr_cached()
- Blur check runs BEFORE acquiring the inference lock. Blurry frames return the
  previous cache unchanged — no garbage OCR output while the user is walking.

### ocr.py — updated _preprocess()
- Added overexposure case (mean > 200): applies gamma=1/0.7≈1.43 LUT to darken
  highlight-blown frames and recover text contrast (outdoor bright sun).
- Skew correction inserted after CLAHE and before sharpening.
- Replaced final `GaussianBlur(3,3)` with `bilateralFilter(d=5, σ=15)` — preserves
  character edges while still smoothing sensor noise.
