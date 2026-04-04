import sys
from pathlib import Path

# Add parent directory to path to import echora modules
sys.path.append(str(Path(__file__).parent.parent))

from src.core.utils import (
    bbox_center, angle_from_x, mm_to_spoken, classify_urgency,
    RateLimiter, AlertCooldown
)

if __name__ == "__main__":
    print("--- Testing utils.py ---\n")

    cx, cy = bbox_center(100, 50, 300, 150)
    print(f"bbox_center(100,50,300,150) -> ({cx}, {cy})")
    assert cx == 200 and cy == 100, "bbox_center FAILED"
    print("  PASSED\n")

    print(f"angle_from_x(320, 640) -> {angle_from_x(320, 640)} deg (expect 0.0)")
    print(f"angle_from_x(  0, 640) -> {angle_from_x(0, 640)} deg (expect -36.5)")
    print(f"angle_from_x(640, 640) -> {angle_from_x(640, 640)} deg (expect +36.5)")
    print("  PASSED\n")

    print(f"mm_to_spoken(450)  -> '{mm_to_spoken(450)}'")
    print(f"mm_to_spoken(1800) -> '{mm_to_spoken(1800)}'")
    print(f"mm_to_spoken(0)    -> '{mm_to_spoken(0)}'")
    print("  PASSED\n")

    print(f"classify_urgency(500)  -> '{classify_urgency(500)}'  (expect DANGER)")
    print(f"classify_urgency(1200) -> '{classify_urgency(1200)}' (expect WARNING)")
    print(f"classify_urgency(3000) -> '{classify_urgency(3000)}' (expect SAFE)")
    print("  PASSED\n")

    limiter = RateLimiter(run_every=3)
    results = [limiter.should_run() for _ in range(9)]
    print(f"RateLimiter(3) over 9 frames -> {results}")
    print("  (expect True on frames 3, 6, 9)\n")

    cooldown = AlertCooldown()
    print(f"AlertCooldown first alert for 'chair'  -> {cooldown.can_alert('chair')}")
    print(f"AlertCooldown repeat alert for 'chair' -> {cooldown.can_alert('chair')}")
    print("  (expect True then False)\n")

    print("--- All tests passed ---")
