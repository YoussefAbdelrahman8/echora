"""Build a focused YOLO dataset for doors and selected elevator controls.

The source datasets are left unchanged.  Elevator annotations are restricted to
``up``, ``updown``, ``close``, ``down``, and ``open`` and are remapped to the
class IDs in ``CLASS_NAMES``.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
DEFAULT_OUTPUT = DATASETS / "accessibility_combined"
CLASS_NAMES = ["door", "door handle", "up", "updown", "close", "down", "open"]
ELEVATOR_CLASSES = ["up", "updown", "close", "down", "open"]


def read_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return [names[index] for index in range(len(names))]
    return list(names)


def clean_output(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "valid", "test"):
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)


def copy_image(source: Path, destination: Path) -> None:
    """Use a hard link when possible, otherwise copy the image."""
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def parse_label(label_path: Path) -> list[list[str]]:
    annotations: list[list[str]] = []
    for number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            if fields:
                print(f"Warning: skipping malformed annotation {label_path}:{number}")
            continue
        try:
            int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            print(f"Warning: skipping malformed annotation {label_path}:{number}")
            continue
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            print(f"Warning: skipping out-of-range annotation {label_path}:{number}")
            continue
        annotations.append(fields)
    return annotations


def add_door_split(source_root: Path, output: Path, split: str, counts: Counter[str]) -> int:
    source_names = read_names(source_root / "data.yaml")
    mapping = {source_names.index("door"): 0, source_names.index("door handle"): 1}
    written = 0
    for image_path in sorted((source_root / split / "images").glob("*")):
        label_path = source_root / split / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path}")
        lines = []
        for fields in parse_label(label_path):
            source_id = int(fields[0])
            if source_id not in mapping:
                continue
            target_id = mapping[source_id]
            lines.append(" ".join([str(target_id), *fields[1:]]))
            counts[CLASS_NAMES[target_id]] += 1
        target_stem = f"door_{image_path.stem}"
        copy_image(image_path, output / split / "images" / f"{target_stem}{image_path.suffix.lower()}")
        (output / split / "labels" / f"{target_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
    return written


def add_elevator_split(source_root: Path, output: Path, split: str, counts: Counter[str]) -> int:
    source_names = read_names(source_root / "data.yaml")
    mapping = {source_names.index(name): CLASS_NAMES.index(name) for name in ELEVATOR_CLASSES}
    written = 0
    for image_path in sorted((source_root / split / "images").glob("*")):
        label_path = source_root / split / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path}")
        lines = []
        for fields in parse_label(label_path):
            source_id = int(fields[0])
            if source_id not in mapping:
                continue
            target_id = mapping[source_id]
            lines.append(" ".join([str(target_id), *fields[1:]]))
            counts[CLASS_NAMES[target_id]] += 1
        # Retain only elevator frames containing at least one requested control.
        if not lines:
            continue
        target_stem = f"elevator_{image_path.stem}"
        copy_image(image_path, output / split / "images" / f"{target_stem}{image_path.suffix.lower()}")
        (output / split / "labels" / f"{target_stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1
    return written


def build(output: Path) -> None:
    clean_output(output)
    door_root = DATASETS / "door"
    elevator_root = DATASETS / "elevator"
    all_counts: Counter[str] = Counter()
    for split in ("train", "valid", "test"):
        counts: Counter[str] = Counter()
        door_images = add_door_split(door_root, output, split, counts)
        elevator_images = add_elevator_split(elevator_root, output, split, counts)
        all_counts.update(counts)
        print(f"{split}: {door_images} door images + {elevator_images} elevator images; {dict(counts)}")

    data = {
        "path": output.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"Dataset written to: {output}")
    print(f"Total boxes: {dict(all_counts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args().output.resolve())
