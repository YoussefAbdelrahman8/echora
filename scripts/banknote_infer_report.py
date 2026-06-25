"""Run the banknote detector over a folder of (unlabeled) test images and
report detection behaviour. This is an INFERENCE-ONLY report: without
ground-truth labels it cannot compute accuracy or a confusion matrix.

Run: .venv/Scripts/python.exe scripts/banknote_infer_report.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets" / "models" / "banknote.pt"
IMG_DIR = ROOT / "datasets" / "banknote_test" / "test" / "test" / "images"
OUT = ROOT / "documentation" / "Copy of New Project_v2" / "assets" / "implementation"
DEPLOY_CONF = 0.85  # config BANKNOTE_CONFIDENCE_THRESHOLD

NAVY, AMBER = "#1F3A5F", "#C8742B"


def main() -> None:
    model = YOLO(str(MODEL))
    names = model.names
    images = sorted(p for p in IMG_DIR.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    print(f"Images found: {len(images)}")

    detected_any = 0          # >=1 box at low conf (0.25)
    detected_deploy = 0       # >=1 box at deploy conf (0.85)
    top_conf_sum = 0.0
    pred_counts: Counter[str] = Counter()

    n = 0
    for p in images:
        n += 1
        try:
            r = model.predict(source=str(p), conf=0.25, imgsz=640,
                              device=0, verbose=False)[0]
        except Exception as exc:  # skip unreadable images (e.g. heic)
            print(f"  [skip] {p.name}: {exc}")
            n -= 1
            continue
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        confs = boxes.conf.tolist()
        clss = boxes.cls.tolist()
        detected_any += 1
        best = max(range(len(confs)), key=lambda i: confs[i])
        top_conf_sum += confs[best]
        if confs[best] >= DEPLOY_CONF:
            detected_deploy += 1
            pred_counts[names[int(clss[best])]] += 1

    total = n
    print(f"Processed: {total}")
    print(f"Detection rate (conf>=0.25): {detected_any}/{total} = {detected_any/total:.1%}")
    print(f"Detection rate (conf>=0.85): {detected_deploy}/{total} = {detected_deploy/total:.1%}")
    if detected_any:
        print(f"Mean top-box confidence (when detected): {top_conf_sum/detected_any:.3f}")
    print("Predicted denomination distribution (conf>=0.85):")
    for k, v in pred_counts.most_common():
        print(f"  {k:14s} {v}")

    # chart of predicted distribution
    if pred_counts:
        labels = [k for k, _ in pred_counts.most_common()]
        vals = [pred_counts[k] for k in labels]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(len(labels)), vals, color=AMBER, edgecolor="white")
        ax.bar_label(bars, padding=3, color=NAVY, fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Number of images")
        ax.set_title(f"Banknote detector: predicted denomination distribution "
                     f"({total} test images, conf>={DEPLOY_CONF})",
                     color=NAVY, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / "banknote_pred_distribution.png", dpi=200, bbox_inches="tight")
        print(f"\nChart -> {OUT / 'banknote_pred_distribution.png'}")


if __name__ == "__main__":
    main()
