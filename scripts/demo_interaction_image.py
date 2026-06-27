"""Demonstrate INTERACTION-mode navigation on a still photo.

Detects the hand (MediaPipe) + target cup/bottle (YOLO COCO), computes the
left/right guidance vector exactly as the real pipeline does, lights the
matching electrodes on the 4x5 wristband grid using the real
ElectrodeGridBuilder, and renders a labelled demo image.

NOTE: a still photo has no depth, so the forward/back (dz) axis is set to 0.
Only the left/right (column) guidance is demonstrated -- which is what guides
the hand toward a cup sitting to its left/right.

Usage:
    .venv\\Scripts\\python.exe scripts/demo_interaction_image.py path/to/photo.jpg
    .venv\\Scripts\\python.exe scripts/demo_interaction_image.py path/to/folder
"""
import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import settings
from src.perception.interaction_detection import (
    ElectrodeGridBuilder,
    GRID_ROWS,
    GRID_COLS,
    GUIDANCE_MAX_PX,
)


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def direction_text(dx_norm: float) -> str:
    if dx_norm < -0.6:
        return "MOVE LEFT  (far)  -> cols 0,1"
    if dx_norm < -0.2:
        return "MOVE LEFT  (slight) -> col 0"
    if dx_norm > 0.6:
        return "MOVE RIGHT (far)  -> cols 3,4"
    if dx_norm > 0.2:
        return "MOVE RIGHT (slight) -> col 4"
    return "ALIGNED (center)"


def detect_hand(hands, frame):
    """Return (hand_bbox, index_tip, landmarks_px) or None."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    lms = res.multi_hand_landmarks[0]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms.landmark]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bbox = (max(0, min(xs) - 15), max(0, min(ys) - 15),
            min(w, max(xs) + 15), min(h, max(ys) + 15))
    return bbox, pts[8], pts            # landmark 8 = index fingertip


def detect_target(model, frame, device):
    """Return the largest interactable (cup/bottle/...) detection or None."""
    res = model(frame, conf=settings.DETECTION_CONFIDENCE_THRESHOLD,
                device=device, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    best = None
    best_area = 0
    for box in res.boxes:
        label = model.names.get(int(box.cls.item()), "")
        if label not in settings.INTERACTABLE_CLASSES:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy.cpu().numpy()[0])
        area = (x2 - x1) * (y2 - y1)
        if area < settings.MIN_INTERACTABLE_AREA_PX:
            continue
        if area > best_area:
            best_area = area
            best = {"label": label, "bbox": (x1, y1, x2, y2),
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "conf": float(box.conf.item())}
    return best


def draw_grid_panel(grid, direction):
    """Render the 4x5 electrode grid as a standalone labelled panel."""
    cell = 70
    pad = 8
    margin = 20
    header = 70
    pw = GRID_COLS * cell + (GRID_COLS + 1) * pad + 2 * margin
    ph = header + GRID_ROWS * cell + (GRID_ROWS + 1) * pad + 2 * margin + 30
    panel = np.full((ph, pw, 3), 30, dtype=np.uint8)

    cv2.putText(panel, "WRISTBAND 4x5 GRID", (margin, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, direction, (margin, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

    lit = []
    y0 = header + margin
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            val = float(grid[r, c])
            pin = r * GRID_COLS + c
            x = margin + pad + c * (cell + pad)
            y = y0 + pad + r * (cell + pad)
            if val > 0.5:
                color = (0, 220, 60)
                lit.append(pin)
            elif val > 0:
                color = (0, 110, 110)
            else:
                color = (55, 55, 55)
            cv2.rectangle(panel, (x, y), (x + cell, y + cell), color, -1)
            cv2.rectangle(panel, (x, y), (x + cell, y + cell), (90, 90, 90), 1)
            txt_color = (0, 0, 0) if val > 0.5 else (160, 160, 160)
            cv2.putText(panel, str(pin), (x + 22, y + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2, cv2.LINE_AA)

    bits = "".join("1" if float(grid.flatten()[i]) > 0.5 else "0" for i in range(20))
    cv2.putText(panel, f"pins on: {lit}", (margin, ph - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, f"bits: {bits}", (margin, ph - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    return panel, lit, bits


def process(img_path, hands, model, builder, device, out_dir):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"\n{img_path.name}: could not read (skipped)")
        return

    print(f"\n{img_path.name}")
    hand = detect_hand(hands, frame)
    target = detect_target(model, frame, device)

    if hand is None:
        print("  no hand detected")
    if target is None:
        print("  no cup/bottle (interactable) detected")
    if hand is None or target is None:
        cv2.imwrite(str(out_dir / f"interact_{img_path.stem}.jpg"), frame)
        return

    hand_bbox, index_tip, landmarks = hand
    fx, fy = index_tip
    ox, oy = target["center"]
    dx_px = ox - fx
    dy_px = oy - fy
    dx_norm = float(np.clip(dx_px / GUIDANCE_MAX_PX, -1.0, 1.0))

    # No depth in a still photo -> dz_norm = 0. Demonstrate left/right guidance.
    grid = builder.build_guidance(dx_norm=dx_norm, dy_norm=0.0, dz_norm=0.0,
                                  intensity=1.0)
    direction = direction_text(dx_norm)

    print(f"  hand index tip: ({fx},{fy})   {target['label']} center: ({ox},{oy})")
    print(f"  dx={dx_px:+d}px  dx_norm={dx_norm:+.2f}  ->  {direction}")

    # ── Draw on the photo ──
    for px, py in landmarks:
        cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
    hx1, hy1, hx2, hy2 = hand_bbox
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
    cv2.putText(frame, "HAND", (hx1, hy1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(frame, (fx, fy), 9, (0, 255, 255), 2)

    tx1, ty1, tx2, ty2 = target["bbox"]
    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 255, 0), 2)
    cv2.putText(frame, f"{target['label'].upper()}  {target['conf']:.0%}",
                (tx1, ty1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                cv2.LINE_AA)

    cv2.arrowedLine(frame, (fx, fy), (ox, oy), (0, 165, 255), 3, tipLength=0.12)
    cv2.putText(frame, direction, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3, cv2.LINE_AA)

    # ── Compose photo + grid panel side by side ──
    panel, lit, bits = draw_grid_panel(grid, direction)
    fh = frame.shape[0]
    scale = fh / panel.shape[0]
    panel = cv2.resize(panel, (int(panel.shape[1] * scale), fh))
    combo = np.hstack([frame, panel])

    out_path = out_dir / f"interact_{img_path.stem}.jpg"
    cv2.imwrite(str(out_path), combo)
    print(f"  pins ON: {lit}   bits: {bits}")
    print(f"  saved: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="photo or folder of photos")
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

    device = pick_device()
    print(f"Device: {device}")
    print(f"Loading YOLO COCO: {settings.COCO_YOLO_MODEL_PATH.name}")
    model = YOLO(str(settings.COCO_YOLO_MODEL_PATH))
    print("Loading MediaPipe Hands (static image mode)...")
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                     min_detection_confidence=0.4,
                                     model_complexity=1)
    builder = ElectrodeGridBuilder()

    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing {len(images)} image(s)\n" + "=" * 60)
    for img_path in images:
        process(img_path, hands, model, builder, device, out_dir)
    hands.close()
    print("\n" + "=" * 60)
    print(f"Demo images saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
