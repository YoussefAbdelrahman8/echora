"""Run ECHORA's OCR pipeline on still images and draw the detected text boxes.

Uses the real OCRReader (PaddleOCR PP-OCRv4 + fine-tuned English rec model),
so the boxes/text match what the live system would read.

Usage:
    .venv\\Scripts\\python.exe scripts/demo_ocr_image.py path/to/photo_or_folder
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.ocr import OCRReader


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="image file or folder")
    args = ap.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"ERROR: path not found: {target}")
        return 1

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = (sorted(p for p in target.iterdir() if p.suffix.lower() in exts)
              if target.is_dir() else [target])
    if not images:
        print(f"ERROR: no images in {target}")
        return 1

    reader = OCRReader()
    reader.load_model()
    if not reader._ready:
        print("ERROR: OCR model failed to load")
        return 1

    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing {len(images)} image(s)\n" + "=" * 60)

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"\n{img_path.name}: could not read (skipped)")
            continue

        dets = reader._run_full_ocr_on_frame(frame)
        print(f"\n{img_path.name}  ->  {len(dets)} text region(s)")
        for d in sorted(dets, key=lambda x: x["bbox"][1]):
            print(f"  '{d['text']}'  ({d['confidence']:.0%})")
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"{d['text']} {d['confidence']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 255, 255), -1)
            cv2.putText(frame, label, (x1 + 2, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        out_path = out_dir / f"ocr_{img_path.stem}_detected.jpg"
        cv2.imwrite(str(out_path), frame)

    print("\n" + "=" * 60)
    print(f"Annotated images saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
