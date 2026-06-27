"""Demonstrate ECHORA face recognition on two still photos.

Registers a face from the FIRST image, then identifies the SECOND using the
same InsightFace (buffalo_sc, ArcFace) model and cosine similarity that the
live system uses. Draws the match box + name + similarity on the test image.

Usage:
    .venv\\Scripts\\python.exe scripts/demo_face_image.py <register.jpg> <test.jpg> --name "Ahmad"
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.config import settings

THRESH = settings.FACE_RECOGNITION_THRESHOLD


def providers():
    return (["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available() else ["CPUExecutionProvider"])


def best_face(app, path):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read {path}")
    faces = app.get(img)
    if not faces:
        return img, None
    return img, max(faces, key=lambda f: f.det_score)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("register")
    ap.add_argument("test")
    ap.add_argument("--name", default="Registered Person")
    args = ap.parse_args()

    app = FaceAnalysis(name="buffalo_sc", providers=providers())
    app.prepare(ctx_id=0, det_size=(640, 640))

    _, reg_face = best_face(app, args.register)
    if reg_face is None:
        print("no face in register image")
        return 1
    reg_emb = reg_face.embedding / np.linalg.norm(reg_face.embedding)
    print(f"Registered '{args.name}' from {Path(args.register).name} (det {reg_face.det_score:.2f})")

    test_img, test_face = best_face(app, args.test)
    if test_face is None:
        print("no face in test image")
        return 1
    test_emb = test_face.embedding / np.linalg.norm(test_face.embedding)
    sim = float(np.dot(reg_emb, test_emb))
    match = sim >= THRESH
    name = args.name if match else "Unknown"
    print(f"Test {Path(args.test).name}: cosine sim={sim:.3f}  threshold={THRESH}  -> {name}")

    x1, y1, x2, y2 = [int(v) for v in test_face.bbox]
    color = (0, 220, 0) if match else (0, 0, 255)
    cv2.rectangle(test_img, (x1, y1), (x2, y2), color, 3)
    label = f"{name}  ({sim:.0%})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(test_img, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), color, -1)
    cv2.putText(test_img, label, (x1 + 5, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"face_{Path(args.test).stem}_identified.jpg"
    cv2.imwrite(str(out_path), test_img)
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
