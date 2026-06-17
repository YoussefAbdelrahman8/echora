"""
Export fine-tuned model to PaddleOCR inference format and test it.

After training completes, run this script to:
  1. Convert the .pdparams checkpoint → inference model (.pdmodel / .pdiparams)
  2. Smoke-test the exported model with PaddleOCR
  3. Print the path to paste into your .env file

Usage (run from repo root):
    python tools/finetune/export_model.py

Optional — test against a specific image:
    python tools/finetune/export_model.py --test-image path/to/image.jpg
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _add_nvidia_dll_dirs():
    """Make pip-installed CUDA/cuDNN DLLs visible to Paddle on Windows."""
    if sys.platform != "win32":
        return
    nvidia_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    dll_dirs = [
        nvidia_dir / "cudnn" / "bin",
        nvidia_dir / "cublas" / "bin",
        nvidia_dir / "cuda_nvrtc" / "bin",
    ]
    existing_path = os.environ.get("PATH", "")
    prepend = []
    for dll_dir in dll_dirs:
        if dll_dir.exists():
            prepend.append(str(dll_dir))
            try:
                os.add_dll_directory(str(dll_dir))
            except (AttributeError, OSError):
                pass
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + [existing_path])


_add_nvidia_dll_dirs()

ROOT         = Path(__file__).parent.parent.parent
FINETUNE_DIR = ROOT / "tools" / "finetune"
PADDLE_SRC   = FINETUNE_DIR / "PaddleOCR_source"
CONFIG       = FINETUNE_DIR / "configs" / "rec_arabic_finetune.yml"
TRAINED_DIR  = FINETUNE_DIR / "output" / "rec_arabic_ft"
INFERENCE_DIR = FINETUNE_DIR / "output" / "rec_arabic_ft" / "inference"
CHAR_DICT    = PADDLE_SRC   / "ppocr" / "utils" / "dict" / "arabic_dict.txt"


def _run(cmd: list, cwd: Path = ROOT):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"ERROR: command failed (exit {result.returncode}).")
        sys.exit(result.returncode)


def export():
    print("\n" + "="*55)
    print("  Echora OCR — Export Fine-tuned Model")
    print("="*55)

    best = TRAINED_DIR / "best_accuracy.pdparams"
    if not best.exists():
        print(f"\nERROR: trained model not found at {best}")
        print("  Run python tools/finetune/train.py first.")
        sys.exit(1)

    export_script = PADDLE_SRC / "tools" / "export_model.py"
    if not export_script.exists():
        print(f"\nERROR: PaddleOCR source not found at {PADDLE_SRC}")
        print("  Run python tools/finetune/train.py to set it up.")
        sys.exit(1)

    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting checkpoint → inference model...")
    _run(
        [
            sys.executable, str(export_script),
            "-c", str(CONFIG),
            "-o",
            f"Global.pretrained_model={TRAINED_DIR / 'best_accuracy'}",
            f"Global.save_inference_dir={INFERENCE_DIR}",
            "Global.export_with_pir=False",
        ],
        cwd=ROOT,
    )

    print(f"\n✓ Inference model saved to: {INFERENCE_DIR}")
    _print_next_steps()


def smoke_test(image_path: str):
    """Load the exported model and run it on one image."""
    print(f"\nSmoke-testing exported model on: {image_path}")
    try:
        from paddleocr import PaddleOCR  # noqa: F401
        import torch
        use_gpu = torch.cuda.is_available()

        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='ar',
            rec_model_dir=str(INFERENCE_DIR),
            use_gpu=use_gpu,
            show_log=False,
        )

        img = cv2.imread(image_path)
        if img is None:
            print(f"  Could not read image: {image_path}")
            return

        result = ocr.ocr(img, cls=True)
        if result and result[0]:
            print(f"\n  Detected text:")
            for line in result[0]:
                if line:
                    _, (text, conf) = line
                    print(f"    [{conf:.2f}]  {text}")
        else:
            print("  No text detected.")

    except ImportError as e:
        print(f"  Import error: {e}")
    except Exception as e:
        print(f"  Error: {e}")


def _print_next_steps():
    rel = INFERENCE_DIR.relative_to(ROOT)
    print(f"""
{'='*55}
  DONE — integrate the fine-tuned model
{'='*55}

Add this to your .env file (or set directly in config):

  OCR_CUSTOM_AR_MODEL={rel}

Then restart Echora. The OCR mode will automatically load
your fine-tuned Arabic model instead of the default one.

To re-run training from scratch (more data / more epochs):
  python tools/finetune/generate_data.py --samples 10000
  python tools/finetune/train.py

To resume interrupted training:
  python tools/finetune/train.py --resume
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-image", type=str, default=None,
                        help="Path to an image for smoke-testing the exported model")
    args = parser.parse_args()

    export()

    if args.test_image:
        smoke_test(args.test_image)
    else:
        _print_next_steps()
