import cv2
import sys
from pathlib import Path

# Add parent directory to path to import echora modules
sys.path.append(str(Path(__file__).parent.parent))

from src.hardware.camera import EchoraCamera
from src.perception.ocr import OCRReader, OCR_LANGUAGE, _get_ocr_gpu
from src.core.config import OCR_TRIGGER_DIST_MM

if __name__ == "__main__":
    print("=== ECHORA ocr.py self-test (Arabic + English) ===\n")
    print(f"Languages configured: {OCR_LANGUAGE}")
    print(f"GPU: {_get_ocr_gpu()}\n")

    cam = EchoraCamera()
    reader = OCRReader()

    try:
        print("Loading camera and OCR model...")
        cam.init_pipeline()
        reader.load_model()
        print("Ready. Point the camera at Arabic or English text.")
        print("Press Q to quit.\n")

        frame_count = 0
        last_text = ""

        while True:
            bundle = cam.get_synced_bundle()
            if bundle is None:
                continue

            frame_count += 1
            rgb = bundle["rgb"]
            depth = bundle["depth"]

            text_dist = 0.0
            if frame_count % 5 == 0:
                text_dist = reader.get_text_distance(rgb, depth)

            text = ""
            if 0 < text_dist < OCR_TRIGGER_DIST_MM:
                text = reader.read_text(rgb)

            if frame_count % 10 == 0:
                print(
                    f"Frame {frame_count:4d} | "
                    f"Dist: {text_dist:6.0f}mm | "
                    f"Detections: {len(reader._last_boxes):2d}",
                    end=""
                )
                if text:
                    print(f" | READ: '{text}'")
                else:
                    print()

            debug = rgb.copy()
            for det in reader._last_boxes:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    debug,
                    f"{det['text'][:20]} ({det['confidence']:.2f})",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1
                )

            dist_str = f"Text: {text_dist:.0f}mm" if text_dist > 0 else "No text"
            cv2.putText(debug, dist_str, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if text:
                last_text = text
            if last_text:
                cv2.putText(
                    debug, f"'{last_text[:60]}'",
                    (10, debug.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            cv2.imshow("ECHORA OCR Test", debug)
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        cv2.destroyAllWindows()
        cam.release()
        print(f"\nStats: {reader.get_stats()}")
        print("=== Done ===")
