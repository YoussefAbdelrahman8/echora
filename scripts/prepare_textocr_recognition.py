"""Prepare English TextOCR word crops for PaddleOCR recognition fine-tuning.

The Kaggle TextOCR export provides one row per word with an axis-aligned
bounding box and transcript. This script samples rows uniformly by annotation
ID, writes cropped word images, and produces PaddleOCR ``image<TAB>text``
label files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image


def sampled(annotation_id: str, sample_percent: int) -> bool:
    value = int(hashlib.sha1(annotation_id.encode("utf-8")).hexdigest()[:8], 16)
    return value % 100 < sample_percent


def validation_split(annotation_id: str) -> bool:
    value = int(hashlib.sha1((annotation_id + "-split").encode("utf-8")).hexdigest()[:8], 16)
    return value % 20 == 0


def english_text(text: str) -> bool:
    if not text or len(text) < 2 or len(text) > 40:
        return False
    return text.isascii() and text.isprintable() and any(char.isalnum() for char in text)


def source_image(images_dir: Path, image_id: str) -> Path:
    return images_dir / f"{image_id}.jpg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-percent", type=int, default=12)
    parser.add_argument("--padding", type=int, default=2)
    args = parser.parse_args()

    if not 1 <= args.sample_percent <= 100:
        parser.error("--sample-percent must be between 1 and 100")

    annotations = args.dataset_root / "annot.csv"
    images_dir = args.dataset_root / "train_val_images" / "train_images"
    if not annotations.is_file() or not images_dir.is_dir():
        parser.error("dataset root must contain annot.csv and train_val_images/train_images")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")

    train_dir = args.output_dir / "train"
    val_dir = args.output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "skipped": 0}
    current_image_id = ""
    current_image: Image.Image | None = None

    with annotations.open("r", encoding="utf-8", newline="") as source, \
            (args.output_dir / "train_list.txt").open("w", encoding="utf-8", newline="") as train_labels, \
            (args.output_dir / "val_list.txt").open("w", encoding="utf-8", newline="") as val_labels:
        reader = csv.DictReader(source)
        for row in reader:
            annotation_id = row["id"]
            text = row["utf8_string"].strip()
            if not sampled(annotation_id, args.sample_percent) or not english_text(text):
                counts["skipped"] += 1
                continue

            image_id = row["image_id"]
            if image_id != current_image_id:
                if current_image is not None:
                    current_image.close()
                image_path = source_image(images_dir, image_id)
                if not image_path.is_file():
                    current_image = None
                    current_image_id = image_id
                    counts["skipped"] += 1
                    continue
                current_image = Image.open(image_path).convert("RGB")
                current_image_id = image_id

            if current_image is None:
                counts["skipped"] += 1
                continue

            x, y, width, height = json.loads(row["bbox"])
            x1 = max(0, int(x) - args.padding)
            y1 = max(0, int(y) - args.padding)
            x2 = min(current_image.width, int(x + width) + args.padding)
            y2 = min(current_image.height, int(y + height) + args.padding)
            if x2 - x1 < 8 or y2 - y1 < 12:
                counts["skipped"] += 1
                continue

            split = "val" if validation_split(annotation_id) else "train"
            filename = f"{annotation_id}.jpg"
            relative_path = f"{split}/{filename}"
            crop = current_image.crop((x1, y1, x2, y2))
            crop.save(args.output_dir / relative_path, quality=95)
            crop.close()

            label_file = val_labels if split == "val" else train_labels
            label_file.write(f"{relative_path}\t{text}\n")
            counts[split] += 1

    if current_image is not None:
        current_image.close()
    print(
        f"Prepared {counts['train']} training and {counts['val']} validation crops "
        f"({counts['skipped']} rows skipped) in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
