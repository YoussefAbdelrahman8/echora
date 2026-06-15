# Rollback — Face Registration HQ Upgrade
Saved: 2026-06-05

## Files Changed
| File | What changed |
|------|-------------|
| `src/core/config.py` | Added `FACE_REGISTRATION_MIN_SCORE`, `FACE_REGISTRATION_CAPTURE_SEC` |
| `src/perception/echora_face.py` | Added `_generate_augmentations()`, `register_face_hq()` |
| `src/storage/register_face.py` | Rewrote `register_person()`, added `_show_message()`, added `import time` |

## To Restore (copy backups)
```bash
cp rollbacks/face_registration_hq/original_config.py       src/core/config.py
cp rollbacks/face_registration_hq/original_echora_face.py  src/perception/echora_face.py
cp rollbacks/face_registration_hq/original_register_face.py src/storage/register_face.py
```

## To Restore (git — only if no other uncommitted changes you want to keep)
```bash
git checkout src/core/config.py src/perception/echora_face.py src/storage/register_face.py
```

## What Each Change Does

### config.py
- `FACE_REGISTRATION_MIN_SCORE = 0.75` — minimum InsightFace det_score required to
  accept an enrollment frame. Frames below this are rejected with audio/visual guidance.
- `FACE_REGISTRATION_CAPTURE_SEC = 2.0` — duration of the silent best-frame capture
  window after the user presses SPACE.

### echora_face.py
- `_generate_augmentations(frame)` — takes one frame and returns 5 photometric
  variants (original + brightness+25 + brightness-25 + CLAHE + unsharp mask).
  Used internally by register_face_hq().
- `register_face_hq(name, frame)` — high-quality single-shot registration:
  1. Loads buffalo_l temporarily (GPU-accelerated, higher-accuracy ArcFace backbone)
  2. Runs quality check (det_score >= FACE_REGISTRATION_MIN_SCORE)
  3. Generates 5 augmented variants of the frame
  4. Extracts buffalo_l ArcFace embeddings from each valid variant
  5. Averages and re-normalises into one final 512-d embedding
  6. Saves to database and reloads in-memory embeddings
  7. Releases buffalo_l (does not stay in memory)
  Falls back to buffalo_sc if buffalo_l unavailable.
  Returns (bool, reason_str).

### register_face.py
- `_show_message(frame, line1, line2, error, delay_ms)` — reusable overlay helper
  for success/failure messages, replaces repeated inline cv2 draw blocks.
- `register_person()` rewrite:
  - Live preview now shows color-coded quality indicator (green/orange/red border)
  - SPACE starts a 2-second silent best-frame capture window (countdown shown)
  - After window: picks the highest det_score frame automatically
  - Quality gate: if best score < FACE_REGISTRATION_MIN_SCORE → rejected with guidance
  - Calls register_face_hq() instead of register_face()
