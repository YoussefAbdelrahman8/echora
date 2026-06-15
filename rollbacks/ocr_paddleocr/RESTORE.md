# Rollback — OCR EasyOCR → PaddleOCR PP-OCRv4
Saved: 2026-06-05

## Files Changed
| File | What changed |
|------|-------------|
| `src/perception/ocr.py` | Full engine swap: EasyOCR → PaddleOCR PP-OCRv4 |
| `requirements.txt` | Removed `easyocr`, added `paddleocr>=2.7.0` |

## To Restore
```bash
cp rollbacks/ocr_paddleocr/original_ocr.py          src/perception/ocr.py
cp rollbacks/ocr_paddleocr/original_requirements.txt requirements.txt
```

## What Changed in ocr.py
- Engine: EasyOCR Reader → PaddleOCR PP-OCRv4 (two-model: Arabic primary + English secondary)
- Arabic model (lang='ar') covers Arabic text + mixed Latin/Arabic content
- English model (lang='en') adds coverage for Latin-only regions not caught by Arabic model
- Results merged with IoU deduplication (Arabic detections take priority over English)
- Preprocessing: now works on full-color BGR frame (LAB CLAHE on L-channel only)
  PaddleOCR handles its own grayscale internally
- Scale factor: 1.0 (full resolution, GPU can handle it; was 0.75 for CPU)
- MIN_WORD_CONFIDENCE raised from 0.3 to 0.4 (PaddleOCR is more accurate)
- Public API identical: init_ocr / read_text / get_text_distance / reset_ocr unchanged
- control_unit.py: NO changes needed

## What Changed in requirements.txt
- Removed: easyocr
- Added: paddleocr>=2.7.0
- NOTE: paddlepaddle-gpu must be installed separately — see note in requirements.txt
