"""Generate implementation-chapter diagrams for the Echora thesis.

Outputs PNGs into documentation/Copy of New Project_v2/assets/implementation/.
Run: .venv/Scripts/python.exe scripts/make_implementation_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "documentation" / "Copy of New Project_v2" / "assets" / "implementation"
OUT.mkdir(parents=True, exist_ok=True)

# Palette
NAVY = "#1F3A5F"
BLUE = "#2C5F8A"
TEAL = "#2E8B8B"
AMBER = "#C8742B"
GRAY = "#5A5A5A"
LIGHT = "#EAF0F6"
LIGHT2 = "#E3F0EF"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def rounded(ax, xy, w, h, text, fc, ec, fs=11, tc="white", bold=True):
    box = FancyBboxPatch(
        (xy[0], xy[1]), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=3)
    return box


def arrow(ax, p1, p2, color=GRAY, style="-|>", lw=1.8, rad=0.0, ls="-"):
    a = FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=16,
        connectionstyle=f"arc3,rad={rad}", color=color, lw=lw,
        linestyle=ls, zorder=1,
    )
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Diagram 1: Real-time processing pipeline
# ---------------------------------------------------------------------------
def pipeline():
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Stage headers
    rounded(ax, (0.3, 6.9), 2.6, 0.8, "Sensing", NAVY, NAVY, fs=12)
    rounded(ax, (3.4, 6.9), 5.0, 0.8, "Perception", NAVY, NAVY, fs=12)
    rounded(ax, (8.9, 6.9), 2.8, 0.8, "Decision & Feedback", NAVY, NAVY, fs=12)

    # Sensing
    rounded(ax, (0.3, 3.2), 2.6, 2.9,
            "OAK-D Camera\n\nRGB 1280x800\n+ Stereo Depth\n+ IMU\n\n(synced bundle)",
            BLUE, NAVY, fs=10.5, tc="white")

    # Perception engines
    eng = [
        ("Obstacle Detection\n+ ByteTrack lifecycle", "every frame", TEAL),
        ("OCR  (PaddleOCR)", "sub-rate", TEAL),
        ("Face ID  (InsightFace)", "every 5 frames", TEAL),
        ("Banknote  (YOLOv8)", "every 5 frames", TEAL),
        ("Interaction  (MediaPipe)", "every 5 frames", TEAL),
    ]
    y = 5.55
    for name, rate, c in eng:
        rounded(ax, (3.4, y), 5.0, 0.78, name, c, "#1F6F6F", fs=10, tc="white")
        ax.text(8.3, y + 0.39, rate, ha="right", va="center", fontsize=8.2,
                color="white", style="italic", zorder=4)
        y -= 0.92

    # Decision + feedback
    rounded(ax, (8.9, 4.4), 2.8, 1.7,
            "State Machine\n5 modes\n(danger override)",
            AMBER, "#8A4E1B", fs=10.5)
    rounded(ax, (8.9, 2.0), 2.8, 1.9,
            "Feedback\n\nAudio: TTS +\nspatial alerts\n\nBLE wrist haptics",
            BLUE, NAVY, fs=10)

    # Arrows
    arrow(ax, (2.9, 4.6), (3.4, 4.6), color=NAVY)
    arrow(ax, (8.4, 5.25), (8.9, 5.25), color=NAVY)
    arrow(ax, (10.3, 4.4), (10.3, 3.9), color=NAVY)
    # loop-back
    arrow(ax, (8.9, 2.6), (2.0, 3.2), color=GRAY, rad=-0.25, ls="--", lw=1.4)
    ax.text(5.0, 1.7, "control loop  (~30 FPS, per-frame iteration)",
            ha="center", va="center", fontsize=9.5, color=GRAY, style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagram 2: State machine
# ---------------------------------------------------------------------------
def state_machine():
    import numpy as np
    fig, ax = plt.subplots(figsize=(11.5, 8.8))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 9)
    ax.axis("off")

    cx, cy = 5.75, 4.6
    cw, chh = 1.55, 0.62  # center half-width / half-height
    sw, shh = 1.25, 0.55  # satellite half-width / half-height

    # four-corner layout: (label, trigger, dx, dy, label_pos)
    sats = [
        ("INTERACTION", "interactable\nobject < 0.8 m", -3.7, 2.5),
        ("OCR", "stable text < 2 m\n& user still", 3.7, 2.5),
        ("FACE_ID", "face detected\n(confident)", -3.7, -2.5),
        ("BANKNOTE", "banknote near\n& user still", 3.7, -2.5),
    ]

    # center node
    rounded(ax, (cx - cw, cy - chh), 2 * cw, 2 * chh, "NAVIGATION\n(default)", AMBER, "#8A4E1B", fs=13)

    for name, trig, dx, dy in sats:
        sx, sy = cx + dx, cy + dy
        rounded(ax, (sx - sw, sy - shh), 2 * sw, 2 * shh, name, BLUE, NAVY, fs=11.5)

        d = np.array([dx, dy], dtype=float)
        d /= np.linalg.norm(d)
        perp = np.array([-d[1], d[0]])
        # start just outside center box, end just outside satellite box
        start = np.array([cx + d[0] * (cw + 0.15), cy + d[1] * (chh + 0.15)])
        end = np.array([sx - d[0] * (sw + 0.15), sy - d[1] * (shh + 0.15)])
        off = perp * 0.16
        # enter (center -> satellite), solid teal
        arrow(ax, tuple(start + off), tuple(end + off), color=TEAL, rad=0.0, lw=1.9)
        # exit (satellite -> center), dashed gray
        arrow(ax, tuple(end - off), tuple(start - off), color=GRAY, rad=0.0, lw=1.4, ls="--")
        # trigger label at midpoint, nudged to the outer side
        mid = (start + end) / 2 + perp * 0.0
        ax.text(mid[0], mid[1], trig, ha="center", va="center", fontsize=8.6,
                color=TEAL, style="italic", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=TEAL, lw=0.6, alpha=0.95))

    ax.text(5.75, 8.55, "Echora Mode State Machine", ha="center", fontsize=15,
            fontweight="bold", color=NAVY)
    ax.text(5.75, 0.45,
            "Solid arrow = enter (trigger persists N frames);   Dashed arrow = exit (trigger gone / dwell timeout)\n"
            "A DANGER-zone obstacle overrides all modes and forces an immediate return to NAVIGATION.",
            ha="center", va="center", fontsize=9.5, color=GRAY)

    fig.savefig(OUT / "state_machine.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagram 3: Accessibility dataset composition
# ---------------------------------------------------------------------------
def dataset():
    splits = ["Train", "Validation", "Test"]
    door = [1067, 134, 133]
    elev = [1089, 155, 141]
    totals = [d + e for d, e in zip(door, elev)]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = range(len(splits))
    b1 = ax.bar(x, door, label="Door / Handle", color=BLUE, edgecolor="white")
    b2 = ax.bar(x, elev, bottom=door, label="Elevator controls (filtered)", color=TEAL, edgecolor="white")

    for i, t in enumerate(totals):
        ax.text(i, t + 40, str(t), ha="center", va="bottom", fontweight="bold", color=NAVY)
    ax.bar_label(b1, label_type="center", color="white", fontsize=9)
    ax.bar_label(b2, label_type="center", color="white", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(splits)
    ax.set_ylabel("Number of images")
    ax.set_title("Combined Accessibility Dataset (2,719 images, 7 classes)",
                 color=NAVY, fontweight="bold")
    ax.set_ylim(0, max(totals) * 1.18)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "accessibility_dataset.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def banknote_dataset():
    # 7,427 images, manually labeled, 80/10/10 split, 11 EGP denomination classes.
    splits = ["Train (80%)", "Validation (10%)", "Test (10%)"]
    counts = [5942, 743, 742]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = range(len(splits))
    bars = ax.bar(x, counts, color=AMBER, edgecolor="white", width=0.6)
    ax.bar_label(bars, padding=4, fontweight="bold", color=NAVY)

    ax.set_xticks(list(x))
    ax.set_xticklabels(splits)
    ax.set_ylabel("Number of images")
    ax.set_title("Custom Egyptian Banknote Dataset (7,400+ images, 11 classes)",
                 color=NAVY, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.5, -0.22,
            "Classes: 1, 5, 10, 20, 50, 100, 200 EGP  -  50 piastres  -  1 EGP coin  -  new 10 & 20 EGP",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color=GRAY)

    fig.tight_layout()
    fig.savefig(OUT / "banknote_dataset.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    state_machine()
    dataset()
    banknote_dataset()
    print("Diagrams written to:", OUT)
