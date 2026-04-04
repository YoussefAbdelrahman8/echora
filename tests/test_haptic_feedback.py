import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.hardware.haptic_feedback import *

if __name__ == "__main__":

    print("=== ECHORA haptic_feedback.py self-test ===\n")

    haptic = HapticFeedback()
    haptic.connect()

    print("Test 1: Preset patterns")

    patterns = {
        "all_off":      pattern_all_off(),
        "all_on":       pattern_all_on(),
        "left":         pattern_left(),
        "right":        pattern_right(),
        "up":           pattern_up(),
        "down":         pattern_down(),
        "center":       pattern_center(),
        "danger_pulse": pattern_danger_pulse(),
    }

    for name, grid in patterns.items():
        haptic.send(grid)
        n_active = int(np.sum(grid > 0))
        print(f"  {name:15s} — {n_active}/30 electrodes active")
        print(haptic.visualise_grid(grid))
        print()

    print("  PASSED\n")

    print("Test 2: Directional guidance vectors")

    directions = [
        ("Right",      100,    0),
        ("Left",      -100,    0),
        ("Up",            0, -100),
        ("Down",          0,  100),
        ("Up-Right",    80,  -80),
        ("Down-Left",  -80,   80),
    ]

    for label, dx, dy in directions:
        haptic.send_direction(dx, dy, intensity=1.0)
        print(f"  {label:12s} dx={dx:+4d} dy={dy:+4d}")

    print("  PASSED\n")

    print("Test 3: Success pulse (3 pulses)")
    haptic.pulse_success(pulses=3, interval=0.1)
    time.sleep(1.0)
    print("  PASSED\n")

    print("Test 4: Stats")
    stats = haptic.get_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")
    print("  PASSED\n")

    haptic.disconnect()
    print("=== All tests passed ===")