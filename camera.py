# =============================================================================
# camera.py — ECHORA Sensor Layer (depthai v3 API)
# =============================================================================
# Owns all communication with the OAK-D hardware.
# Provides one clean synchronised Kalman-filtered data bundle per frame.
#
# Performance fixes applied:
#   - setSyncThreshold(200) — tolerant sync on fast hardware
#   - sync queue maxSize=2, blocking=False — always latest frame, no buildup
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import depthai as dai
import numpy as np
import cv2
import time
import threading
from typing import Optional, Dict, Any

from config import (
    CAMERA_RGB_WIDTH,
    CAMERA_RGB_HEIGHT,
    CAMERA_DEPTH_WIDTH,
    CAMERA_DEPTH_HEIGHT,
    CAMERA_FPS,
    DEPTH_MAX_MM,
    DEPTH_MIN_MM,
    KALMAN_PROCESS_NOISE,
    KALMAN_MEASUREMENT_NOISE,
    KALMAN_MAX_MISSED_FRAMES,
)
from utils import logger, get_timestamp_ms


# =============================================================================
# ECHORA CAMERA CLASS
# =============================================================================

class EchoraCamera:
    """
    Manages the OAK-D hardware for ECHORA using the depthai v3 API.

    Key v3 API differences from v2:
      - No XLinkOut nodes needed. Queues created with .createOutputQueue()
        directly on any node output.
      - Sync node outputs a dai.MessageGroup — access streams by name:
        group["rgb"], group["depth"].
      - Camera nodes use .build(socket) and .requestOutput().
      - pipeline.start() starts the device.
    """

    def __init__(self):
        """
        Prepares empty variables. Does NOT start the camera yet.
        Call init_pipeline() to start the hardware.
        """

        # The depthai Pipeline object — None until init_pipeline() runs.
        self.pipeline: Optional[dai.Pipeline] = None

        # Output queues — created directly on node outputs in v3.
        self.sync_queue = None   # synchronised RGB + depth bundles
        self.imu_queue  = None   # IMU packets

        # Kalman filter — None until init_kalman() runs.
        self.kalman: Optional[cv2.KalmanFilter] = None

        # Tracks consecutive missed frames per object ID.
        self.missed_frames: Dict[str, int] = {}

        # Latest IMU reading — updated by background thread.
        self._latest_imu: Dict[str, Any] = {
            "accel": {"x": 0.0, "y": 0.0, "z": 0.0},
            "gyro":  {"x": 0.0, "y": 0.0, "z": 0.0},
            "timestamp_ms": 0.0
        }

        # Lock prevents two threads from reading/writing _latest_imu at once.
        self._imu_lock = threading.Lock()

        # Flag to control the IMU background thread.
        self._running = False

        logger.info("EchoraCamera created. Call init_pipeline() to start.")


    # =========================================================================
    # PIPELINE INITIALISATION
    # =========================================================================

    def init_pipeline(self):
        """
        Builds the depthai v3 pipeline and starts the camera hardware.

        Performance notes:
          - Sync threshold set to 200ms — tolerant on fast hardware like RTX PC.
            Prevents excessive sync warnings when processing is fast.
          - Sync queue maxSize=2, blocking=False — if we are processing faster
            than the camera produces frames, we always get the LATEST frame
            and old queued frames are dropped automatically.
          - IMU queue maxSize=50 — large buffer for 400-480Hz IMU stream.
        """

        logger.info("Initialising depthai v3 pipeline...")

        # ── Create the Pipeline ───────────────────────────────────────────────
        self.pipeline = dai.Pipeline()

        # ── Create Camera Nodes ───────────────────────────────────────────────
        # .build(socket) sets which physical camera to use.
        # CAM_A = center RGB camera
        # CAM_B = left mono camera
        # CAM_C = right mono camera
        cam_rgb   = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A
        )
        cam_left  = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B
        )
        cam_right = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C
        )

        # ── Request Output Streams ────────────────────────────────────────────
        # BGR888p = Blue Green Red, 8 bits per channel — OpenCV native format.
        rgb_out = cam_rgb.requestOutput(
            (CAMERA_RGB_WIDTH, CAMERA_RGB_HEIGHT),
            dai.ImgFrame.Type.BGR888p
        )

        # Fixed size divisible by 16 — required by StereoDepth node.
        left_out = cam_left.requestOutput(
            (CAMERA_DEPTH_WIDTH, CAMERA_DEPTH_HEIGHT),
            dai.ImgFrame.Type.GRAY8
        )
        right_out = cam_right.requestOutput(
            (CAMERA_DEPTH_WIDTH, CAMERA_DEPTH_HEIGHT),
            dai.ImgFrame.Type.GRAY8
        )

        # ── Create and Configure StereoDepth Node ─────────────────────────────
        stereo = self.pipeline.create(dai.node.StereoDepth)

        # Lens distortion correction — required for accurate depth.
        stereo.setRectification(True)

        # Left-right consistency check — removes noisy readings at edges.
        stereo.setLeftRightCheck(True)

        # Extended disparity — better accuracy for objects closer than ~70cm.
        # Critical for ECHORA — blind users get close to obstacles.
        stereo.setExtendedDisparity(True)

        # Align depth map to RGB camera viewpoint.
        # After this, depth[y][x] corresponds to rgb[y][x] in world space.
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(CAMERA_DEPTH_WIDTH, CAMERA_DEPTH_HEIGHT)

        # Connect mono cameras to stereo depth inputs.
        left_out.link(stereo.left)
        right_out.link(stereo.right)

        # ── Create Sync Node ──────────────────────────────────────────────────
        # Holds frames until it has a matching RGB + depth pair at the same
        # timestamp, then releases them together as a MessageGroup.
        sync = self.pipeline.create(dai.node.Sync)

        # Run sync logic on the host CPU — recommended for depthai v3.
        sync.setRunOnHost(True)

        # ── KEY PERFORMANCE FIX: setSyncThreshold ─────────────────────────────
        # Allows RGB and depth frames to be up to 200ms apart in timestamp
        # and still be considered "synchronised".
        #
        # Why 200ms on RTX PC?
        #   On fast hardware (RTX GPU), our main loop processes frames in ~5ms.
        #   The camera produces frames at ~30 FPS (33ms per frame).
        #   If our loop is too fast, the Sync node can't keep up matching
        #   timestamps and floods the log with sync warnings.
        #   200ms threshold eliminates these warnings without affecting quality.
        #
        # On slower hardware (MacBook CPU) use 100ms.
        # On very fast hardware (RTX GPU) use 200ms.
        import datetime
        sync.setSyncThreshold(datetime.timedelta(milliseconds=200))

        # Connect RGB and depth outputs to sync's named input slots.
        # Names "rgb" and "depth" are how we retrieve them from MessageGroup.
        rgb_out.link(sync.inputs["rgb"])
        stereo.depth.link(sync.inputs["depth"])

        # ── KEY PERFORMANCE FIX: Queue size and blocking mode ─────────────────
        # maxSize=2: keep only 2 bundles buffered.
        # blocking=False: if queue is full, DROP the oldest frame automatically.
        #
        # Why this matters:
        #   With blocking=True (default), if our processing is slow,
        #   frames pile up in the queue. When we speed up, we process
        #   stale frames from seconds ago — this is the lag you see.
        #
        #   With blocking=False + maxSize=2, the queue always contains
        #   the LATEST 2 frames. Old frames are discarded. We always
        #   process the most current view of the world.
        #
        #   This is critical for a real-time wearable — the blind user
        #   needs to know what is in front of them RIGHT NOW, not 2
        #   seconds ago.
        self.sync_queue = sync.out.createOutputQueue(
            maxSize=2, blocking=False
        )

        # ── Create IMU Node ───────────────────────────────────────────────────
        imu_node = self.pipeline.create(dai.node.IMU)

        # Accelerometer at 480Hz — raw values include gravity (m/s²).
        imu_node.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 480)

        # Gyroscope at 400Hz — rotation rate in rad/s.
        imu_node.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)

        # Send packets immediately — don't batch them (minimises latency).
        imu_node.setBatchReportThreshold(1)

        # Buffer up to 10 packets if we fall behind — prevents USB overload.
        imu_node.setMaxBatchReports(10)

        # Large IMU queue — IMU runs at 400-480Hz, much faster than main loop.
        self.imu_queue = imu_node.out.createOutputQueue(
            maxSize=50, blocking=False
        )

        # ── Start the Pipeline ────────────────────────────────────────────────
        logger.info("Starting OAK-D device...")
        self.pipeline.start()
        logger.info("Device started successfully.")

        # ── Initialise Kalman Filter ──────────────────────────────────────────
        self.init_kalman()

        # ── Start IMU Background Thread ───────────────────────────────────────
        # IMU produces data at 400-480Hz but main loop runs at ~30Hz.
        # Background thread reads IMU continuously and stores latest value.
        # Main loop reads stored value — always fresh, never blocking.
        self._running = True
        self._imu_thread = threading.Thread(
            target=self._imu_reader_thread,
            daemon=True
        )
        self._imu_thread.start()

        logger.info("Pipeline fully initialised. Ready to stream.")


    # =========================================================================
    # KALMAN FILTER SETUP
    # =========================================================================

    def init_kalman(self):
        """
        Creates and configures the OpenCV Kalman filter.

        Tracks: [x, y, velocity_x, velocity_y] — 4 state variables.
        Observes: [x, y] — 2 measurement variables (position only).
        """

        # KalmanFilter(dynamParams=4, measureParams=2)
        # 4 state variables: x, y, vx, vy
        # 2 measurement variables: x, y
        self.kalman = cv2.KalmanFilter(4, 2)

        # Transition matrix — encodes "position += velocity × 1 frame"
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix — maps state [x,y,vx,vy] to observation [x,y]
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise — how much we trust our physics prediction.
        self.kalman.processNoiseCov = (
            np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        )

        # Measurement noise — how much we trust raw camera readings.
        self.kalman.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
        )

        # Initial error covariance — starting uncertainty about state.
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        logger.info("Kalman filter initialised.")


    # =========================================================================
    # KALMAN FILTER OPERATIONS
    # =========================================================================

    def kalman_update(self, x: float, y: float) -> tuple:
        """
        Feeds a new measured position into the Kalman filter.
        Returns the corrected, smoothed position estimate.

        Call every frame when YOLO gives a fresh detection position.
        """

        self.kalman.predict()
        measurement = np.array([[x], [y]], dtype=np.float32)
        corrected   = self.kalman.correct(measurement)
        return float(corrected[0][0]), float(corrected[1][0])


    def kalman_predict(self) -> tuple:
        """
        Predicts next position with NO new measurement.
        Call when a tracked object temporarily disappears.
        """

        predicted = self.kalman.predict()
        return float(predicted[0][0]), float(predicted[1][0])


    # =========================================================================
    # IMU BACKGROUND THREAD
    # =========================================================================

    def _imu_reader_thread(self):
        """
        Runs continuously in background, reading IMU packets at 400-480Hz.
        Stores the latest reading in self._latest_imu.
        Thread-safe via self._imu_lock.
        """

        logger.info("IMU reader thread started.")

        while self._running:
            try:
                # tryGet() is non-blocking — returns None if nothing ready.
                imu_data = self.imu_queue.tryGet()

                if imu_data is None:
                    # Nothing available — sleep briefly to avoid 100% CPU.
                    time.sleep(0.002)
                    continue

                for packet in imu_data.packets:
                    accel = packet.acceleroMeter
                    gyro  = packet.gyroscope
                    ts_ms = accel.getTimestamp().total_seconds() * 1000

                    new_imu = {
                        "accel": {
                            "x": float(accel.x),
                            "y": float(accel.y),
                            "z": float(accel.z),
                        },
                        "gyro": {
                            "x": float(gyro.x),
                            "y": float(gyro.y),
                            "z": float(gyro.z),
                        },
                        "timestamp_ms": ts_ms
                    }

                    with self._imu_lock:
                        self._latest_imu = new_imu

            except Exception as e:
                logger.warning(f"IMU read error: {e}")
                time.sleep(0.01)

        logger.info("IMU reader thread stopped.")


    # =========================================================================
    # MAIN DATA ACCESS FUNCTION
    # =========================================================================

    def get_synced_bundle(self) -> Optional[Dict]:
        """
        Reads one synchronised frame bundle using the v3 Sync node.

        Returns the LATEST available bundle or None if nothing is ready.
        None is normal — just skip this iteration of the main loop.

        With blocking=False and maxSize=2 on the queue:
          - If the camera is faster than processing: we get the latest frame,
            old frames are silently dropped.
          - If processing is faster than camera: we get None and skip.
          - Either way: zero lag, always current data.
        """

        if not self.pipeline.isRunning():
            logger.error("Pipeline is not running.")
            return None

        try:
            # tryGet() is non-blocking — returns None if nothing ready.
            msg_group = self.sync_queue.tryGet()

            if msg_group is None:
                return None

            # Access each stream by the name used when linking to sync.inputs[].
            rgb_msg   = msg_group["rgb"]
            depth_msg = msg_group["depth"]

            # getCvFrame() → numpy array (H, W, 3) uint8 BGR.
            rgb = rgb_msg.getCvFrame()

            # getFrame() → numpy array (H, W) uint16 depth values in mm.
            depth = depth_msg.getFrame()

            # Clamp depth to configured range.
            # 0 = no valid depth data.
            # Values above DEPTH_MAX_MM = too far, set to max.
            depth = np.clip(depth, 0, DEPTH_MAX_MM).astype(np.uint16)

            # Thread-safe IMU read.
            with self._imu_lock:
                imu = self._latest_imu.copy()

            return {
                "rgb":          rgb,
                "depth":        depth,
                "imu":          imu,
                "timestamp_ms": get_timestamp_ms()
            }

        except Exception as e:
            logger.error(f"Error reading bundle: {e}")
            return None


    # =========================================================================
    # CONVENIENCE HELPERS
    # =========================================================================

    def get_imu_data(self) -> Dict:
        """Returns the latest IMU reading. Thread-safe."""
        with self._imu_lock:
            return self._latest_imu.copy()


    def update_missed_frames(self, active_ids: list) -> list:
        """
        Tracks consecutive missed frames per tracked object ID.
        Returns list of IDs missing too long — should be dropped.
        """

        for obj_id in list(self.missed_frames.keys()):
            self.missed_frames[obj_id] += 1

        for obj_id in active_ids:
            self.missed_frames[obj_id] = 0

        to_drop = [
            obj_id for obj_id, count in self.missed_frames.items()
            if count > KALMAN_MAX_MISSED_FRAMES
        ]

        for obj_id in to_drop:
            del self.missed_frames[obj_id]
            logger.debug(f"Dropped {obj_id} — missed {KALMAN_MAX_MISSED_FRAMES} frames.")

        return to_drop


    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def release(self):
        """
        Cleanly stops the pipeline and background thread.
        Always call when ECHORA exits.
        """

        logger.info("Releasing camera resources...")

        self._running = False

        if hasattr(self, '_imu_thread') and self._imu_thread.is_alive():
            self._imu_thread.join(timeout=2.0)

        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as e:
                logger.warning(f"Pipeline stop warning: {e}")
            self.pipeline = None

        logger.info("Camera released cleanly.")


# =============================================================================
# SELF-TEST
# =============================================================================

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
                    f"Accel x:{accel['x']:+.2f} y:{accel['y']:+.2f} "
                    f"z:{accel['z']:+.2f} | "
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