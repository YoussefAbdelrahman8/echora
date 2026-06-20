"""Fine-tune the ECHORA obstacle YOLO model on the Kaggle indoor dataset.

Example:
    python scripts/train_indoor_yolo.py --epochs 80 --batch 8 --device 0
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable

import yaml
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import settings


DATASET_SLUG = "thepbordin/indoor-object-detection"
DEFAULT_PROJECT = settings.BASE_DIR / "assets" / "training_runs"
DEFAULT_OUTPUT_MODEL = settings.ASSETS_DIR / "models" / "yolov8s_indoor.pt"


def _download_dataset() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required to download the indoor dataset. "
            "Install project requirements first."
        ) from exc

    return Path(kagglehub.dataset_download(DATASET_SLUG))


def _find_data_yaml(dataset_root: Path) -> Path | None:
    candidates = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.name.lower() in {"data.yaml", "data.yml", "dataset.yaml"}
    ]
    if not candidates:
        return None

    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _infer_split(root: Path, split: str) -> tuple[Path, Path] | None:
    image_dir = _first_existing(
        [
            root / "images" / split,
            root / split / "images",
            root / split,
        ]
    )
    label_dir = _first_existing(
        [
            root / "labels" / split,
            root / split / "labels",
        ]
    )
    if image_dir and label_dir:
        return image_dir, label_dir
    return None


def _read_class_names(root: Path) -> list[str]:
    for names_file in root.rglob("classes.txt"):
        names = [
            line.strip()
            for line in names_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if names:
            return names
    raise RuntimeError(
        "Could not infer class names. Expected a data.yaml/dataset.yaml or classes.txt "
        "inside the downloaded dataset."
    )


def _normalise_names(names: object) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    raise RuntimeError("Dataset YAML has an unsupported names format.")


def _write_data_yaml(dataset_root: Path, names: dict[int, str]) -> Path:
    train = _infer_split(dataset_root, "train")
    val = _infer_split(dataset_root, "valid") or _infer_split(dataset_root, "val")
    test = _infer_split(dataset_root, "test")

    if not train or not val:
        raise RuntimeError(
            "Could not infer YOLO train/validation folders. Expected a standard layout "
            "such as images/train, labels/train, images/valid, labels/valid."
        )

    generated_yaml = dataset_root / "echora_data.yaml"
    data = {
        "path": dataset_root.as_posix(),
        "train": train[0].relative_to(dataset_root).as_posix(),
        "val": val[0].relative_to(dataset_root).as_posix(),
        "names": names,
    }
    if test:
        data["test"] = test[0].relative_to(dataset_root).as_posix()

    generated_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return generated_yaml


def _make_data_yaml(dataset_root: Path) -> Path:
    names = {idx: name for idx, name in enumerate(_read_class_names(dataset_root))}
    return _write_data_yaml(dataset_root, names)


def _normalise_data_yaml(data_yaml: Path, dataset_root: Path) -> Path:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if names is None:
        return _make_data_yaml(dataset_root)

    train_path = Path(str(data.get("train", "")))
    val_path = Path(str(data.get("val", "")))
    yaml_base = Path(data.get("path", data_yaml.parent))

    def exists(path: Path) -> bool:
        if path.is_absolute():
            return path.exists()
        return (yaml_base / path).exists() or (data_yaml.parent / path).exists()

    if exists(train_path) and exists(val_path):
        return data_yaml

    return _write_data_yaml(dataset_root, _normalise_names(names))


def prepare_dataset(dataset_root: Path | None) -> Path:
    root = dataset_root or _download_dataset()
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    data_yaml = _find_data_yaml(root)
    if data_yaml:
        return _normalise_data_yaml(data_yaml, root)

    return _make_data_yaml(root)


def train(args: argparse.Namespace) -> Path:
    data_yaml = prepare_dataset(args.dataset)
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Base YOLO model not found: {model_path}")

    model = YOLO(str(model_path))
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        pretrained=True,
        patience=args.patience,
        workers=args.workers,
        exist_ok=args.exist_ok,
    )

    save_dir = Path(results.save_dir)
    best_model = save_dir / "weights" / "best.pt"
    if not best_model.exists():
        raise FileNotFoundError(f"Training finished but best checkpoint was not found: {best_model}")

    output_model = args.output.expanduser().resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, output_model)
    return output_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the current ECHORA YOLO obstacle model on indoor objects."
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Existing dataset path. Downloads from Kaggle when omitted.")
    parser.add_argument("--model", type=Path, default=settings.YOLO_MODEL_PATH, help="Base .pt model to fine-tune.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_MODEL, help="Where to copy the trained best.pt.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Ultralytics run output directory.")
    parser.add_argument("--name", default="indoor_yolov8s", help="Ultralytics run name.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Example: 0 for CUDA GPU, cpu for CPU.")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--exist-ok", action="store_true", help="Allow Ultralytics to reuse the run directory.")
    parser.add_argument("--prepare-only", action="store_true", help="Download/find the dataset YAML and exit without training.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.prepare_only:
        prepared_yaml = prepare_dataset(cli_args.dataset)
        print(f"Dataset YAML ready: {prepared_yaml}")
    else:
        trained_model = train(cli_args)
        print(f"Fine-tuned model saved to: {trained_model}")
        print("To use it at runtime, set YOLO_MODEL_PATH to this path in your environment or .env file.")
