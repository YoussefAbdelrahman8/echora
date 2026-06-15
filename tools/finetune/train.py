"""
Fine-tuning runner for Echora OCR — Arabic PP-OCRv4 recognition model.

This script:
  1. Checks all prerequisites (data, PaddleOCR source, pretrained model).
  2. Downloads anything that's missing (PaddleOCR source via git, pretrained weights).
  3. Runs the PaddleOCR training pipeline with our fine-tuning config.

Usage (run from repo root):
    python tools/finetune/train.py

Prerequisites:
    pip install paddlepaddle-gpu==2.6.1   # GPU — match your CUDA version
    pip install arabic-reshaper python-bidi
    python tools/finetune/generate_data.py   # generate training data first
"""

import os
import platform
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


# ── Paddle check (must happen before anything else) ───────────────────────────

def _check_paddle():
    """
    Verify PaddlePaddle (the training framework) is installed.
    paddleocr ≠ paddlepaddle — the inference package does not pull in the
    training framework on all platforms.
    """
    try:
        import paddle  # noqa: F401
        return
    except ImportError:
        pass

    is_mac    = platform.system() == "Darwin"
    is_linux  = platform.system() == "Linux"
    is_arm    = platform.machine() in ("arm64", "aarch64")

    print("\n" + "="*60)
    print("  ERROR: 'paddle' (PaddlePaddle) is not installed.")
    print("  This is the training framework, separate from paddleocr.")
    print("="*60)

    if is_mac and is_arm:
        print("""
  You are on Apple Silicon (M1/M2/M3).
  PaddlePaddle GPU training requires Linux + NVIDIA CUDA.

  OPTIONS:
  ─────────────────────────────────────────────────────
  A) Run training on your GPU PC (recommended):
       • Copy the dataset there:
           scp -r tools/finetune/data/ user@gpu-pc:~/echora/tools/finetune/
       • On the GPU PC, install PaddlePaddle GPU and run train.py.

  B) Run CPU training on this Mac (slow, for testing only):
       pip install paddlepaddle==2.6.1
     Then re-run:  python tools/finetune/train.py
""")
    elif is_mac:
        print("""
  You are on macOS (Intel).
  PaddlePaddle GPU is not available on macOS — CPU training only.

  Install CPU version:
    pip install paddlepaddle==2.6.1

  Then re-run:  python tools/finetune/train.py

  For GPU training, run on a Linux machine with NVIDIA GPU.
""")
    elif is_linux:
        print("""
  Install the version matching your CUDA:

    CUDA 11.x:
      pip install paddlepaddle-gpu==2.6.1 \\
        -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

    CUDA 12.x:
      pip install paddlepaddle-gpu==2.6.2.post120 \\
        -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

    CPU only:
      pip install paddlepaddle==2.6.1

  Then re-run:  python tools/finetune/train.py
""")
    else:
        print("""
  Install PaddlePaddle:
    pip install paddlepaddle==2.6.1        # CPU
    pip install paddlepaddle-gpu==2.6.1    # GPU (Linux + CUDA 11.x)

  See: https://www.paddlepaddle.org.cn/install/quick
""")
    sys.exit(1)


_check_paddle()

# ── Paths (all relative to repo root) ─────────────────────────────────────────
ROOT         = Path(__file__).parent.parent.parent   # repo root
FINETUNE_DIR = ROOT / "tools" / "finetune"
PADDLE_SRC   = FINETUNE_DIR / "PaddleOCR_source"
PRETRAIN_DIR = FINETUNE_DIR / "pretrain_models"
DATA_DIR     = FINETUNE_DIR / "data" / "finetune"
CONFIG       = FINETUNE_DIR / "configs" / "rec_arabic_finetune.yml"
OUTPUT_DIR   = FINETUNE_DIR / "output" / "rec_arabic_ft"

PADDLEOCR_REPO = "https://github.com/PaddlePaddle/PaddleOCR.git"
PRETRAIN_URL   = (
    "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/"
    "arabic_PP-OCRv4_rec_train.tar"
)
PRETRAIN_TAR   = PRETRAIN_DIR / "arabic_PP-OCRv4_rec_train.tar"
PRETRAIN_UNTAR = PRETRAIN_DIR / "arabic_PP-OCRv4_rec_train"


def _header(msg: str):
    print(f"\n{'─'*55}")
    print(f"  {msg}")
    print(f"{'─'*55}")


def _run(cmd: list, cwd: Path = ROOT):
    """Run a subprocess command, streaming output. Exits on failure."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"\nERROR: command failed (exit {result.returncode}).")
        sys.exit(result.returncode)


# ── Step 1 — Data check ────────────────────────────────────────────────────────

def check_data():
    _header("Step 1 — Checking training data")
    train_labels = DATA_DIR / "train" / "labels.txt"
    val_labels   = DATA_DIR / "val"   / "labels.txt"

    if not train_labels.exists() or not val_labels.exists():
        print("  Training data not found. Generating now...")
        print("  (Re-run with --samples N for a different dataset size)")
        _run([sys.executable, str(FINETUNE_DIR / "generate_data.py")])
    else:
        n_train = len(train_labels.read_text().strip().splitlines())
        n_val   = len(val_labels.read_text().strip().splitlines())
        print(f"  ✓ train: {n_train} samples  |  val: {n_val} samples")


# ── Step 2 — PaddleOCR source ─────────────────────────────────────────────────

def check_paddle_src():
    _header("Step 2 — Checking PaddleOCR source")

    if PADDLE_SRC.exists() and (PADDLE_SRC / "tools" / "train.py").exists():
        print(f"  ✓ PaddleOCR source already at {PADDLE_SRC}")
        return

    print(f"  Cloning PaddleOCR into {PADDLE_SRC} ...")
    print("  (This is ~200 MB and may take a few minutes.)")
    _run(
        ["git", "clone", "--depth", "1", PADDLEOCR_REPO, str(PADDLE_SRC)],
        cwd=ROOT,
    )

    # Install training-specific requirements (different from inference)
    req_file = PADDLE_SRC / "requirements.txt"
    if req_file.exists():
        print("  Installing PaddleOCR training requirements...")
        _run([sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"])

    print(f"  ✓ PaddleOCR source ready.")


# ── Step 3 — Pretrained model ─────────────────────────────────────────────────

def check_pretrained():
    _header("Step 3 — Checking pretrained model")

    best_acc = PRETRAIN_UNTAR / "best_accuracy.pdparams"
    if best_acc.exists():
        print(f"  ✓ Pretrained model at {PRETRAIN_UNTAR}")
        return

    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)

    if not PRETRAIN_TAR.exists():
        print(f"  Downloading pretrained Arabic PP-OCRv4 weights (~50 MB)...")
        print(f"  URL: {PRETRAIN_URL}")
        try:
            urllib.request.urlretrieve(
                PRETRAIN_URL, str(PRETRAIN_TAR),
                reporthook=_download_progress,
            )
            print()
        except Exception as e:
            print(f"\n  ERROR: download failed: {e}")
            print("  Please download manually and place at:")
            print(f"    {PRETRAIN_TAR}")
            sys.exit(1)

    print(f"  Extracting {PRETRAIN_TAR.name} ...")
    with tarfile.open(str(PRETRAIN_TAR)) as t:
        t.extractall(str(PRETRAIN_DIR))
    print(f"  ✓ Pretrained model ready.")


def _download_progress(block, block_size, total):
    downloaded = block * block_size
    if total > 0:
        pct = min(100, downloaded * 100 // total)
        mb  = downloaded / 1_048_576
        print(f"\r  {pct:3d}%  {mb:.1f} MB", end="", flush=True)


# ── Step 4 — Training ─────────────────────────────────────────────────────────

def run_training(resume: bool = False):
    _header("Step 4 — Fine-tuning")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_script = PADDLE_SRC / "tools" / "train.py"
    cmd = [
        sys.executable, str(train_script),
        "-c", str(CONFIG),
    ]
    if resume:
        latest = OUTPUT_DIR / "latest"
        if latest.exists():
            cmd += ["-o", f"Global.checkpoints={latest}"]
            print(f"  Resuming from checkpoint: {latest}")

    print(f"\n  Training config : {CONFIG}")
    print(f"  Output dir      : {OUTPUT_DIR}")
    print(f"  Model will save every 5 epochs — best_accuracy is kept.\n")

    _run(cmd, cwd=ROOT)

    best = OUTPUT_DIR / "best_accuracy.pdparams"
    if best.exists():
        print(f"\n  ✓ Training complete. Best model: {best}")
        print(f"\n  Next step: python tools/finetune/export_model.py")
    else:
        print(f"\n  WARNING: best_accuracy.pdparams not found in {OUTPUT_DIR}")
        print("  Check training logs for errors.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune Echora OCR model.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--skip-data",    action="store_true")
    parser.add_argument("--skip-source",  action="store_true")
    parser.add_argument("--skip-pretrain",action="store_true")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  Echora OCR — Fine-tuning Pipeline")
    print("="*55)

    if not args.skip_data:
        check_data()
    if not args.skip_source:
        check_paddle_src()
    if not args.skip_pretrain:
        check_pretrained()

    run_training(resume=args.resume)
