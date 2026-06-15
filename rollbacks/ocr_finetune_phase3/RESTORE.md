# Rollback — OCR Phase 3: Fine-tuning Pipeline
Saved: 2026-06-05

## Files Modified (rollback these)
```bash
cp rollbacks/ocr_finetune_phase3/original_config.py      src/core/config.py
cp rollbacks/ocr_finetune_phase3/original_ocr.py         src/perception/ocr.py
cp rollbacks/ocr_finetune_phase3/original_requirements.txt requirements.txt
```

## New files created (delete these to fully revert)
```bash
rm -rf tools/finetune/
```

## Summary of changes
- config.py: added OCR_CUSTOM_AR_MODEL, OCR_CUSTOM_EN_MODEL paths
- ocr.py: load_model() uses custom model path when configured
- requirements.txt: added arabic-reshaper, python-bidi
- tools/finetune/: full fine-tuning pipeline (data gen, training, export)
