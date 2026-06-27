"""Run ECHORA's navigation YOLO model on still images and draw detections.

Uses the accessibility model (YOLO_MODEL_PATH); falls back to COCO for any
RELEVANT_CLASSES it surfaces. A still photo has no depth, so distance zones
are not shown -- this demonstrates the detection stage of navigation.

Usage:
    .venv\\Scripts\\python.exe scripts/demo_obstacle_image.py path_or_folder
"""
import argparse
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.config import settings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    target = Path(args.path)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = (sorted(p for p in target.iterdir() if p.suffix.lower() in exts)
              if target.is_dir() else [target])
    if not images:
        print("no images")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = settings.YOLO_MODEL_PATH
    if not model_path.exists():
        model_path = settings.COCO_YOLO_MODEL_PATH
    print(f"Device {device} | model {model_path.name}")
    model = YOLO(str(model_path))

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    print("=" * 60)
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        res = model(frame, conf=args.conf, device=device, verbose=False)[0]
        print(f"\n{img_path.name}  ->  {0 if res.boxes is None else len(res.boxes)} detections")
        if res.boxes is not None:
            for b in res.boxes:
                lbl = model.names.get(int(b.cls.item()), "?")
                print(f"  {lbl:<16} {float(b.conf.item()):.0%}")
        out_path = out_dir / f"obstacle_{img_path.stem}_detected.jpg"
        cv2.imwrite(str(out_path), res.plot())
    print("\n" + "=" * 60)
    print(f"saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
