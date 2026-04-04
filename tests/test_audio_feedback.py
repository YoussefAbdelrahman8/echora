import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.hardware.audio_feedback import *

if __name__ == "__main__":

    print("=== ECHORA audio_feedback.py self-test ===")
    print("You should hear audio through your speakers/earphones.\n")

    audio = AudioFeedback()
    audio.init_audio()
    time.sleep(0.5)

    print("Test 1: Basic speech")
    audio.speak("ECHORA audio system online.", priority=SpeechPriority.NORMAL)
    time.sleep(2.5)
    print("  Done\n")

    print("Test 2: Priority — LOW first, URGENT should play first")
    audio.speak("This is a low priority message.", priority=SpeechPriority.LOW)
    audio.speak("Danger. Person. 40 centimetres.",  priority=SpeechPriority.URGENT)
    time.sleep(5.0)
    print("  Done\n")

    print("Test 3: Spatial alerts")
    print("  Far left...")
    audio.play_spatial_alert(-35.0, "DANGER")
    time.sleep(1.0)
    print("  Center...")
    audio.play_spatial_alert(0.0, "WARNING")
    time.sleep(1.0)
    print("  Far right...")
    audio.play_spatial_alert(35.0, "DANGER")
    time.sleep(1.0)
    print("  Done\n")

    print("Test 4: Obstacle announcements")
    audio.announce_obstacle({
        "label": "person", "distance_mm": 500,
        "angle_deg": -20.0, "urgency": "DANGER"
    })
    time.sleep(0.1)
    audio.announce_obstacle({
        "label": "chair", "distance_mm": 1500,
        "angle_deg": 15.0, "urgency": "WARNING"
    })
    time.sleep(4.0)
    print("  Done\n")

    print("Test 5: Mode changes")
    audio.announce_mode_change("OCR")
    time.sleep(2.0)
    audio.announce_mode_change("NAVIGATION")
    time.sleep(2.0)
    print("  Done\n")

    print("Test 6: OCR and banknote")
    audio.announce_ocr("Emergency Exit")
    time.sleep(3.0)
    audio.announce_banknote("50 Egyptian pounds")
    time.sleep(3.0)
    print("  Done\n")

    print("Shutting down...")
    audio.release()
    print("\n=== Self-test complete ===")