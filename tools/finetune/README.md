# Echora OCR — Fine-tuning Guide

Fine-tunes PaddleOCR PP-OCRv4 Arabic recognition on Egyptian-specific text
(street signs, product labels, mixed Arabic/English content).

## What you get after fine-tuning

A custom recognition model trained on realistic Egyptian text that:
- Reads Arabic signage vocabulary more accurately
- Handles mixed Arabic/English text common in Egypt
- Recognises digit-mixed labels (e.g., "الطابق ٣", "محل No.5")

The detection model (DB++) is NOT changed — it is already robust for
natural scenes. Only the recognition head is fine-tuned.

---

## Prerequisites

```bash
# 1 — GPU PaddlePaddle (match your CUDA version)
pip install paddlepaddle-gpu==2.6.1 \
  -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 2 — Arabic rendering libraries
pip install arabic-reshaper python-bidi

# 3 — Everything else (if not already installed)
pip install -r requirements.txt
```

---

## Steps

### Step 1 — Generate synthetic training data

```bash
python tools/finetune/generate_data.py
```

Options:
| Flag | Default | Notes |
|------|---------|-------|
| `--samples` | 6000 | Total images. 6000 is a good starting point. |
| `--output` | `tools/finetune/data/finetune` | Output directory. |
| `--val-split` | 0.1 | 10% validation. |

On first run, Cairo, Amiri, and Noto Arabic fonts are auto-downloaded (~3 MB).
The script prints a live progress counter. Expect ~2–5 minutes for 6000 samples.

Output structure:
```
tools/finetune/data/finetune/
├── train/
│   ├── images/img_000001.jpg ... img_005400.jpg
│   └── labels.txt
└── val/
    ├── images/img_000001.jpg ... img_000600.jpg
    └── labels.txt
```

**Tip:** Add your own real-world images by appending lines to `labels.txt`:
```
images/my_real_sign.jpg	نص الصورة هنا
```

---

### Step 2 — Run fine-tuning

```bash
python tools/finetune/train.py
```

This script automatically:
1. Detects missing data and generates it if needed
2. Clones PaddleOCR source into `tools/finetune/PaddleOCR_source/` (~200 MB, once)
3. Downloads pretrained Arabic PP-OCRv4 weights (~50 MB, once)
4. Starts training

Expect ~30–60 minutes on a modern GPU (RTX 3070+) for 50 epochs.
Checkpoints are saved every 5 epochs. Best accuracy model is kept at:
```
tools/finetune/output/rec_arabic_ft/best_accuracy.pdparams
```

To resume an interrupted run:
```bash
python tools/finetune/train.py --resume
```

---

### Step 3 — Export and deploy

```bash
python tools/finetune/export_model.py
```

Converts the trained checkpoint to PaddleOCR inference format and prints
the path to add to your `.env` file.

To smoke-test on a real image before deploying:
```bash
python tools/finetune/export_model.py --test-image path/to/sign.jpg
```

---

### Step 4 — Activate in Echora

Add one line to your `.env` file (repo root):

```
OCR_CUSTOM_AR_MODEL=tools/finetune/output/rec_arabic_ft/inference
```

Then restart Echora. The OCR module will automatically use your fine-tuned
model instead of the default PP-OCRv4 Arabic model.

To revert to the default model: remove the line from `.env`.

---

## Improving accuracy further

| Action | Effect |
|--------|--------|
| Double sample count (`--samples 12000`) | +5–10% accuracy |
| Add real photos from your environment to `labels.txt` | Biggest accuracy gain |
| Train more epochs (edit `epoch_num` in config) | Diminishing returns after 50 |
| Lower `learning_rate` to 0.00005 for second fine-tune pass | Stabilises tricky words |

---

## Troubleshooting

**`arabic_reshaper` not installed warning**
Arabic letters will not connect properly. Fix: `pip install arabic-reshaper python-bidi`

**CUDA out of memory during training**
Reduce `batch_size_per_card` in `configs/rec_arabic_finetune.yml` from 64 to 32.

**Training loss doesn't decrease**
Check that `pretrained_model` in the config points to the correct `.pdparams` path
(without the `.pdparams` extension — PaddlePaddle adds it automatically).

**PaddleOCR source clone fails (no internet / proxy)**
Manually clone: `git clone https://github.com/PaddlePaddle/PaddleOCR.git tools/finetune/PaddleOCR_source`

**Exported model gives worse results than default**
The fine-tuned model specialises on your training vocabulary. If the real
environment has very different text, add real labeled images to `labels.txt`
and re-train.
