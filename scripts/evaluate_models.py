"""Evaluate Echora's detection models on their local test splits.

Produces, per model: mAP@50, mAP@50-95, precision, recall, a per-class table,
and a confusion-matrix PNG (saved by Ultralytics). Results are written under
assets/eval/<name>/ and a JSON summary is printed.

Only models whose test datasets are present locally are evaluated. The banknote
test set lives in a private Roboflow workspace; export it to
datasets/banknote/ (YOLO format with data.yaml) to include it here.

Run: .venv/Scripts/python.exe scripts/evaluate_models.py
"""
from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "assets" / "eval"
DOC_ASSETS = ROOT / "documentation" / "Copy of New Project_v2" / "assets" / "implementation"

# (display name, model path, data.yaml, run name)
TARGETS = [
    ("Accessibility (doors/handles/elevator)",
     ROOT / "assets" / "models" / "yolov8s_accessibility.pt",
     ROOT / "datasets" / "accessibility_combined" / "data.yaml",
     "accessibility"),
    ("Banknote (EGP)",
     ROOT / "assets" / "models" / "banknote.pt",
     ROOT / "datasets" / "banknote_test" / "data_eval.yaml",
     "banknote"),
]


def evaluate(name: str, model_path: Path, data_yaml: Path, run: str) -> dict | None:
    if not model_path.exists():
        print(f"[skip] {name}: model not found ({model_path})")
        return None
    if not data_yaml.exists():
        print(f"[skip] {name}: dataset not found ({data_yaml}) -- export it to include this model")
        return None

    print(f"\n=== Evaluating: {name} ===")
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        project=str(EVAL_DIR),
        name=run,
        plots=True,
        device=0,
        exist_ok=True,
        verbose=False,
    )

    names = model.names
    per_class = {}
    try:
        for i, c in enumerate(metrics.box.ap_class_index):
            per_class[names[int(c)]] = {
                "precision": round(float(metrics.box.p[i]), 4),
                "recall": round(float(metrics.box.r[i]), 4),
                "mAP50": round(float(metrics.box.ap50[i]), 4),
                "mAP50_95": round(float(metrics.box.ap[i]), 4),
            }
    except Exception as exc:  # pragma: no cover
        print(f"  (per-class extraction failed: {exc})")

    summary = {
        "model": name,
        "num_classes": len(names),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "save_dir": str(metrics.save_dir),
        "per_class": per_class,
    }

    # Copy the confusion matrix into the thesis assets for direct inclusion.
    cm = Path(metrics.save_dir) / "confusion_matrix_normalized.png"
    if cm.exists():
        DOC_ASSETS.mkdir(parents=True, exist_ok=True)
        dest = DOC_ASSETS / f"cm_{run}.png"
        dest.write_bytes(cm.read_bytes())
        summary["confusion_matrix"] = str(dest)
        print(f"  confusion matrix -> {dest}")

    print(f"  mAP50={summary['mAP50']}  mAP50-95={summary['mAP50_95']}  "
          f"P={summary['precision']}  R={summary['recall']}")
    return summary


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    results = [r for r in (evaluate(*t) for t in TARGETS) if r]
    out = EVAL_DIR / "summary.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSummary written to: {out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
