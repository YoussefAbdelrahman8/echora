"""
Synthetic OCR training data generator — Egyptian Arabic + English.

Renders text on realistic backgrounds with photometric augmentations to
produce cropped text-region images in PaddleOCR SimpleDataSet format:

    data/finetune/train/images/img_00001.jpg   <-- cropped text image
    data/finetune/train/labels.txt             <-- "images/img_00001.jpg\ttext"
    data/finetune/val/images/...
    data/finetune/val/labels.txt

Usage:
    python tools/finetune/generate_data.py
    python tools/finetune/generate_data.py --samples 8000 --output data/finetune
    python tools/finetune/generate_data.py --samples 2000 --val-split 0.15
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ── Dependency checks ──────────────────────────────────────────────────────────

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _ARABIC_SUPPORT = True
except ImportError:
    print(
        "WARNING: arabic_reshaper / python-bidi not installed.\n"
        "  Arabic text will render incorrectly (letters won't connect).\n"
        "  Fix: pip install arabic-reshaper python-bidi\n"
    )
    _ARABIC_SUPPORT = False


# ── Vocabulary ─────────────────────────────────────────────────────────────────

ARABIC_WORDS = [
    # Navigation / streets
    "شارع", "طريق", "ميدان", "حارة", "ممر", "مدخل", "مخرج", "دور", "طابق",
    "يمين", "يسار", "أمام", "خلف", "شمال", "جنوب", "شرق", "غرب",
    "تقاطع", "دوار", "كوبري", "نفق", "محور",

    # Places
    "مطعم", "مقهى", "محل", "صيدلية", "مستشفى", "مدرسة", "جامعة",
    "بنك", "فندق", "مسجد", "كنيسة", "سوبرماركت", "بقالة", "مخبز",
    "عيادة", "مكتب", "شركة", "مصنع", "مخزن", "مغسلة", "حلاق",
    "صالون", "جراج", "محطة", "موقف",

    # Signs
    "مفتوح", "مغلق", "للبيع", "للإيجار", "ممنوع", "خطر", "احذر",
    "خروج", "طوارئ", "ممنوع الدخول", "ممنوع الوقوف",
    "يمنع التدخين", "محجوز", "مجاني", "مدفوع", "كاش", "فيزا",

    # Common phrases
    "أهلاً وسهلاً", "مرحباً", "شكراً جزيلاً", "من فضلك", "لو سمحت",
    "ساعات العمل", "يومياً", "جميع الأيام",

    # Floor / room numbers
    "الطابق الأول", "الطابق الثاني", "الطابق الثالث", "الدور الأرضي",
    "الباب الأول", "شقة رقم", "مكتب رقم", "غرفة رقم",

    # Products / food
    "كوكاكولا", "بيبسي", "عصير برتقال", "مياه معدنية", "قهوة عربية",
    "شاي أخضر", "خبز بلدي", "جبنة بيضاء", "لبن طازج",

    # Numbers
    "١٢٣", "٤٥٦", "٧٨٩٠", "٢٠٢٥", "٠١٠٠", "١١٠",

    # Mixed-language (common in Egypt)
    "واي فاي مجاني", "اتصل بنا", "تواصل معنا", "زورونا على",
]

ARABIC_PHRASES = [
    "مدخل العمارة", "موقف السيارات", "حارة مسدودة",
    "طريق سريع", "منطقة عمل", "تحت الإنشاء",
    "ممنوع التصوير", "ممنوع الانتظار", "خطر الموت",
    "يُرجى المحافظة على النظافة", "للتواصل اتصل",
    "ساعات العمل من ٩ص حتى ٩م", "مفتوح طوال اليوم",
    "عروض خاصة اليوم فقط", "تخفيضات حتى ٥٠٪",
]

ENGLISH_WORDS = [
    "OPEN", "CLOSED", "ENTRANCE", "EXIT", "PUSH", "PULL",
    "DANGER", "WARNING", "CAUTION", "NO ENTRY", "NO PARKING",
    "WELCOME", "THANK YOU", "PLEASE", "SORRY",
    "FLOOR 1", "FLOOR 2", "FLOOR 3", "GROUND FLOOR",
    "OFFICE", "SHOP", "PHARMACY", "HOSPITAL", "SCHOOL",
    "ATM", "WiFi", "FREE WiFi", "SALE", "DISCOUNT",
    "OPEN 24H", "DELIVERY", "TAKEAWAY",
    "Tel:", "Phone:", "Email:", "Website:",
    "Monday", "Tuesday", "Sunday", "Daily",
    "January", "February", "March",
    "2024", "2025", "No.", "Apt.",
]

MIXED_PHRASES = [
    "WhatsApp واتساب", "Online أونلاين", "ATM ماكينة",
    "WiFi واي فاي", "Facebook فيسبوك", "Instagram انستغرام",
    "Coca Cola كوكاكولا", "Pepsi بيبسي",
    "محل No.5", "مكتب No.12", "غرفة No.3",
    "رقم 1", "رقم 15", "رقم 200",
    "Open مفتوح", "Closed مغلق",
]

NUMBER_STRINGS = [
    str(random.randint(1, 9999)) for _ in range(50)
] + [
    f"+20{random.randint(1000000000, 9999999999)}" for _ in range(20)
] + [f"{random.randint(1,99)}%" for _ in range(15)]


# ── Font management ────────────────────────────────────────────────────────────

FONT_URLS = {
    "Cairo-Regular": (
        "https://github.com/google/fonts/raw/main/ofl/cairo/Cairo%5Bslnt%2Cwght%5D.ttf",
        "Cairo-Regular.ttf",
    ),
    "Amiri-Regular": (
        "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf",
        "Amiri-Regular.ttf",
    ),
    "NotoSansArabic": (
        "https://github.com/google/fonts/raw/main/ofl/notosansarabic/"
        "NotoSansArabic%5Bwdth%2Cwght%5D.ttf",
        "NotoSansArabic.ttf",
    ),
}


def _download_fonts(font_dir: Path) -> List[Path]:
    """Download Arabic fonts if not already present. Returns list of font paths."""
    font_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (url, filename) in FONT_URLS.items():
        dest = font_dir / filename
        if dest.exists():
            paths.append(dest)
            continue
        print(f"  Downloading font: {filename} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            paths.append(dest)
            print(f"  ✓ {filename}")
        except Exception as e:
            print(f"  ✗ {filename} failed: {e}")
    return paths


def _load_fonts(font_dir: Path, sizes: List[int]) -> dict:
    """
    Load PIL fonts at each size. Returns dict[size] = list[ImageFont].
    Falls back to PIL default font if no TTF fonts are available.
    """
    ttf_paths = list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf"))
    fonts: dict = {}
    for size in sizes:
        loaded = []
        for path in ttf_paths:
            try:
                loaded.append(ImageFont.truetype(str(path), size))
            except Exception:
                pass
        if not loaded:
            loaded = [ImageFont.load_default()]
        fonts[size] = loaded
    return fonts


# ── Background generators ──────────────────────────────────────────────────────

def _bg_solid(w: int, h: int) -> np.ndarray:
    """Random solid color background, biased towards light colors."""
    if random.random() < 0.7:
        c = random.randint(200, 255)
        bg = np.full((h, w, 3), (c, c, c), dtype=np.uint8)
    else:
        bg = np.random.randint(30, 220, (h, w, 3), dtype=np.uint8)
    return bg


def _bg_gradient(w: int, h: int) -> np.ndarray:
    """Horizontal or vertical two-color gradient."""
    c1 = np.array([random.randint(180, 255)] * 3)
    c2 = np.array([random.randint(180, 255)] * 3)
    bg = np.zeros((h, w, 3), dtype=np.float32)
    if random.random() < 0.5:
        for x in range(w):
            bg[:, x] = c1 + (c2 - c1) * (x / max(w - 1, 1))
    else:
        for y in range(h):
            bg[y, :] = c1 + (c2 - c1) * (y / max(h - 1, 1))
    return bg.astype(np.uint8)


def _bg_noisy(w: int, h: int) -> np.ndarray:
    """Paper-like texture: light base + gaussian noise."""
    base = random.randint(210, 250)
    bg   = np.full((h, w, 3), base, dtype=np.float32)
    noise = np.random.normal(0, random.uniform(5, 20), (h, w, 3))
    return np.clip(bg + noise, 0, 255).astype(np.uint8)


_BACKGROUNDS = [_bg_solid, _bg_solid, _bg_gradient, _bg_noisy]


# ── Text color selection ───────────────────────────────────────────────────────

def _text_color(bg: np.ndarray) -> Tuple[int, int, int]:
    """Pick a text color with sufficient contrast to the background."""
    mean_lum = float(np.mean(bg))
    if mean_lum > 160:
        return (
            random.randint(0, 60),
            random.randint(0, 60),
            random.randint(0, 60),
        )
    return (
        random.randint(200, 255),
        random.randint(200, 255),
        random.randint(200, 255),
    )


# ── Augmentation pipeline ──────────────────────────────────────────────────────

def _augment(img: np.ndarray) -> np.ndarray:
    """
    Apply random photometric + mild geometric augmentation.
    All operations are probabilistic so some images get little or no augmentation.
    """
    # Brightness jitter
    if random.random() < 0.5:
        delta = random.randint(-40, 40)
        img   = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    # Contrast jitter
    if random.random() < 0.4:
        alpha = random.uniform(0.7, 1.4)
        img   = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    # Gaussian noise
    if random.random() < 0.4:
        sigma = random.uniform(2, 12)
        noise = np.random.normal(0, sigma, img.shape)
        img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Mild blur (camera defocus / motion)
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # JPEG compression artifact
    if random.random() < 0.3:
        q = random.randint(50, 90)
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return img


# ── Single image generator ─────────────────────────────────────────────────────

def _render_text_image(
    text:  str,
    font:  ImageFont.FreeTypeFont,
    pad:   int = 6,
) -> Optional[np.ndarray]:
    """
    Render one text string into a cropped BGR image on a random background.
    Returns None if the text cannot be rendered (font missing, zero-size, etc.)
    """
    # Prepare display text (Arabic RTL + reshaping)
    display = text
    if _ARABIC_SUPPORT and any("؀" <= c <= "ۿ" for c in text):
        try:
            display = get_display(arabic_reshaper.reshape(text))
        except Exception:
            pass

    # Measure text size
    try:
        dummy = Image.new("RGB", (1, 1))
        draw  = ImageDraw.Draw(dummy)
        bbox  = draw.textbbox((0, 0), display, font=font)
        tw    = bbox[2] - bbox[0]
        th    = bbox[3] - bbox[1]
    except Exception:
        return None

    if tw <= 0 or th <= 0:
        return None

    w = tw + pad * 2
    h = th + pad * 2

    # Background
    bg_fn = random.choice(_BACKGROUNDS)
    bg    = bg_fn(w, h)
    pil_bg = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))

    # Draw text
    draw  = ImageDraw.Draw(pil_bg)
    color = _text_color(bg)
    draw.text((pad - bbox[0], pad - bbox[1]), display, font=font, fill=color)

    # Convert back to BGR numpy
    img = cv2.cvtColor(np.array(pil_bg), cv2.COLOR_RGB2BGR)

    # Augment
    img = _augment(img)

    return img


# ── Dataset generation ─────────────────────────────────────────────────────────

def _build_vocabulary() -> List[str]:
    """Combine all vocabulary lists into a single pool."""
    pool = (
        ARABIC_WORDS * 4
        + ARABIC_PHRASES * 3
        + ENGLISH_WORDS * 2
        + MIXED_PHRASES * 2
        + NUMBER_STRINGS
    )
    # Also add numbers appended to common words
    extras = []
    for word in random.sample(ARABIC_WORDS, min(30, len(ARABIC_WORDS))):
        extras.append(f"{word} {random.randint(1, 99)}")
    for word in random.sample(ENGLISH_WORDS, min(20, len(ENGLISH_WORDS))):
        extras.append(f"{word} {random.randint(1, 99)}")
    return pool + extras


def generate(
    n_samples:  int,
    output_dir: Path,
    val_split:  float = 0.1,
    font_dir:   Optional[Path] = None,
    seed:       int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    font_dir = font_dir or (output_dir.parent / "fonts")
    print(f"\n{'='*55}")
    print(f"  Echora OCR — Synthetic Data Generator")
    print(f"{'='*55}")
    print(f"  Samples  : {n_samples}")
    print(f"  Val split: {val_split:.0%}")
    print(f"  Output   : {output_dir}")
    print(f"  Font dir : {font_dir}\n")

    # Download fonts
    print("Downloading fonts (skip if already cached)...")
    _download_fonts(font_dir)
    font_sizes = [18, 22, 26, 32, 38, 44]
    all_fonts  = _load_fonts(font_dir, font_sizes)

    n_val   = max(1, int(n_samples * val_split))
    n_train = n_samples - n_val
    splits  = [("train", n_train), ("val", n_val)]

    vocab = _build_vocabulary()

    for split_name, n in splits:
        img_dir   = output_dir / split_name / "images"
        label_path = output_dir / split_name / "labels.txt"
        img_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating {n} {split_name} samples...")
        lines   = []
        written = 0
        tried   = 0

        while written < n:
            tried += 1
            if tried > n * 5:
                print(f"  WARNING: gave up after {tried} attempts ({written}/{n} written).")
                break

            text = random.choice(vocab)
            if not text.strip():
                continue

            size  = random.choice(font_sizes)
            fonts = all_fonts[size]
            font  = random.choice(fonts)

            img = _render_text_image(text, font)
            if img is None:
                continue

            fname = f"img_{written:06d}.jpg"
            fpath = img_dir / fname
            cv2.imwrite(str(fpath), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lines.append(f"images/{fname}\t{text}")
            written += 1

            if written % 500 == 0:
                print(f"  {written}/{n} ...")

        label_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  ✓ {written} images  |  labels → {label_path}")

    # Write metadata
    meta = {
        "n_train": n_train,
        "n_val":   n_val,
        "vocab_size": len(set(vocab)),
        "font_sizes": font_sizes,
        "arabic_support": _ARABIC_SUPPORT,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n✓ Dataset ready at: {output_dir}")
    print("  Next step: python tools/finetune/train.py\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic OCR training data.")
    parser.add_argument("--samples",   type=int,   default=6000,
                        help="Total number of images (default: 6000)")
    parser.add_argument("--output",    type=str,   default="tools/finetune/data/finetune",
                        help="Output dataset directory")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of samples for validation (default: 0.1)")
    parser.add_argument("--font-dir",  type=str,   default=None,
                        help="Directory with .ttf/.otf fonts (auto-downloaded if absent)")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    generate(
        n_samples  = args.samples,
        output_dir = Path(args.output),
        val_split  = args.val_split,
        font_dir   = Path(args.font_dir) if args.font_dir else None,
        seed       = args.seed,
    )
