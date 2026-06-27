"""Run the banknote YOLO model on a single image and report every detection.

Usage:
    .venv\\Scripts\\python.exe scripts/test_banknote_image.py path/to/photo.jpg
    .venv\\Scripts\\python.exe scripts/test_banknote_image.py path/to/photo.jpg --conf 0.10
"""
import argparse
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "models" / "banknote.pt"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to a banknote photo OR a folder of photos")
    ap.add_argument("--conf", type=float, default=0.10,
                    help="confidence threshold (default 0.10 to show raw detections)")
    args = ap.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"ERROR: path not found: {target}")
        return 1
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found: {MODEL_PATH}")
        return 1

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if target.is_dir():
        images = sorted(p for p in target.iterdir() if p.suffix.lower() in exts)
    else:
        images = [target]
    if not images:
        print(f"ERROR: no images found in {target}")
        return 1

    device = pick_device()
    print(f"Loading model on {device}: {MODEL_PATH.name}")
    model = YOLO(str(MODEL_PATH))
    print(f"Model classes ({len(model.names)}): {list(model.names.values())}")
    print(f"\nRunning on {len(images)} image(s) at conf>={args.conf:.2f}\n" + "=" * 60)

    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"\n{img_path.name}: could not read image (skipped)")
            continue

        result = model(frame, conf=args.conf, device=device, verbose=False)[0]
        boxes = result.boxes
        print(f"\n{img_path.name}")
        if boxes is None or len(boxes) == 0:
            print(f"  no detections above conf={args.conf:.2f}")
            continue

        dets = []
        for box in boxes:
            cls = int(box.cls.item())
            label = model.names.get(cls, f"class {cls}")
            conf = float(box.conf.item())
            x1, y1, x2, y2 = (int(v) for v in box.xyxy.cpu().numpy()[0])
            dets.append((conf, label, (x1, y1, x2, y2)))

        for conf, label, (x1, y1, x2, y2) in sorted(dets, reverse=True):
            flag = "  <-- above 0.85 production threshold" if conf >= 0.85 else ""
            print(f"  {label:<20} {conf:6.1%}  bbox=({x1},{y1},{x2},{y2}){flag}")

        out_path = out_dir / f"banknote_{img_path.stem}_detected.jpg"
        cv2.imwrite(str(out_path), result.plot())

    print("\n" + "=" * 60)
    print(f"Annotated images saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
