import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import echora modules
sys.path.append(str(Path(__file__).parent.parent))

from src.hardware.camera import EchoraCamera

if __name__ == "__main__":
    print("=== ECHORA camera.py self-test (depthai v3) ===\n")

    cam = EchoraCamera()
    cam.init_pipeline()

    frame_count = 0

    try:
        while cam.pipeline.isRunning():
            bundle = cam.get_synced_bundle()

            if bundle is None:
                continue

            frame_count += 1
            rgb   = bundle["rgb"]
            depth = bundle["depth"]
            imu   = bundle["imu"]

            if frame_count % 10 == 0:
                h, w      = depth.shape
                center_d  = depth[h // 2, w // 2]
                accel     = imu["accel"]
                gyro      = imu["gyro"]

                print(
                    f"Frame {frame_count:4d} | "
                    f"Depth centre: {center_d:5d}mm | "
                    f"Accel x:{accel['x']:+.2f} y:{accel['y']:+.2f} z:{accel['z']:+.2f} | "
                    f"Gyro y:{gyro['y']:+.3f} rad/s"
                )

                sx, sy = cam.kalman_update(float(w // 2), float(h // 2))
                print(f"             | Kalman: ({sx:.1f}, {sy:.1f})")

            cv2.imshow("ECHORA RGB", rgb)

            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_col = cv2.applyColorMap(np.uint8(depth_vis), cv2.COLORMAP_JET)
            cv2.imshow("ECHORA Depth", depth_col)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        cam.release()
        print(f"\nTest complete. Processed {frame_count} frames.")
