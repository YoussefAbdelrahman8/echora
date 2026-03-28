# =============================================================================
# camera.py — ECHORA Sensor Layer (depthai v3 API)
# =============================================================================
# Owns all communication with the OAK-D Lite hardware.
# Provides one clean synchronised Kalman-filtered data bundle per frame.
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
    Manages the OAK-D Lite hardware for ECHORA using the depthai v3 API.

    Key v3 API differences from v2:
      - No XLinkOut nodes needed. Queues are created with .createOutputQueue()
        directly on any node output.
      - Pipeline runs with 'with dai.Pipeline() as pipeline' context manager
        plus pipeline.start() and pipeline.isRunning().
      - Sync node outputs a dai.MessageGroup — access streams by name like
        a dictionary: group["rgb"], group["depth"].
      - Camera nodes use .build(socket) and .requestOutput() or
        .requestFullResolutionOutput().
    """

    def __init__(self):
        """
        Prepares empty variables. Does NOT start the camera yet.
        Call init_pipeline() to start the hardware.
        """

        # The depthai Pipeline object — None until init_pipeline() runs.
        self.pipeline: Optional[dai.Pipeline] = None

        # Output queues — created directly on node outputs in v3.
        # No device.getOutputQueue() needed anymore.
        self.sync_queue  = None   # receives synchronised RGB + depth bundles
        self.imu_queue   = None   # receives IMU packets

        # Kalman filter — None until init_kalman() runs.
        self.kalman: Optional[cv2.KalmanFilter] = None

        # Tracks consecutive missed frames per object ID.
        # Key = object ID string, Value = number of missed frames.
        self.missed_frames: Dict[str, int] = {}

        # Latest IMU reading — updated by background thread.
        # Initialised with zeros so it's safe to read before IMU starts.
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

        v3 pattern:
          - pipeline.create(dai.node.Camera).build(socket) for cameras
          - .requestOutput() for RGB, .requestFullResolutionOutput() for mono
          - stereo depth via setRectification + setLeftRightCheck
          - Sync node receives named inputs, outputs a MessageGroup
          - All queues created with .createOutputQueue() on the node output
          - pipeline.start() starts the device
        """

        logger.info("Initialising depthai v3 pipeline...")

        # ── Create the Pipeline ───────────────────────────────────────────────
        # In v3 the Pipeline object also manages the device context.
        self.pipeline = dai.Pipeline()


        # ── Create Camera Nodes ───────────────────────────────────────────────
        # .build(socket) is the v3 way to set which physical camera to use.
        # CAM_A = center RGB colour camera
        # CAM_B = left mono (grayscale) camera
        # CAM_C = right mono (grayscale) camera
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
        # BGR888p = Blue Green Red, 8 bits per channel — OpenCV's native format.
        rgb_out = cam_rgb.requestOutput(
            (CAMERA_RGB_WIDTH, CAMERA_RGB_HEIGHT),
            dai.ImgFrame.Type.BGR888p
        )

        # For the OAK-D (not Lite), requestFullResolutionOutput() gives 1280×800
        # on mono cameras, but after setDepthAlign the StereoDepth node tries to
        # match RGB native resolution (1352×1008) which is not divisible by 16.
        # Fix: request a fixed size that IS divisible by 16 on both mono cameras.
        # 1280×800 works perfectly — divisible by 16, good stereo quality.
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

        # setRectification(True) corrects lens distortion before depth
        # computation. Required for accurate depth on the OAK-D Lite.
        stereo.setRectification(True)

        # Left-right consistency check — only keeps depth pixels where both
        # cameras agree. Removes noisy ghost readings at object edges.
        stereo.setLeftRightCheck(True)

        # Extended disparity improves accuracy for objects closer than ~70cm.
        # Important for ECHORA — blind users may get very close to obstacles.
        stereo.setExtendedDisparity(True)

        # Align the depth map to the RGB camera's viewpoint.
        # After this, depth[y][x] corresponds to the same world point as
        # rgb[y][x]. Essential for combining YOLO detections with depth.
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(CAMERA_DEPTH_WIDTH, CAMERA_DEPTH_HEIGHT)

        # Connect mono camera outputs to stereo depth inputs.
        # This is like plugging cables: left camera output → stereo.left input.
        left_out.link(stereo.left)
        right_out.link(stereo.right)


        # ── Create Sync Node ──────────────────────────────────────────────────
        # The Sync node holds frames until it has a matching pair from all
        # connected inputs at the same timestamp, then releases them together
        # as a single MessageGroup.
        #
        # This ensures our RGB frame and depth frame are always from exactly
        # the same moment in time — no mismatches.
        sync = self.pipeline.create(dai.node.Sync)

        # setRunOnHost(True) means the sync logic runs on the laptop CPU,
        # not on the camera chip. For v3 this is the recommended setting.
        sync.setRunOnHost(True)

        # Connect RGB output to sync's "rgb" input slot.
        # The name "rgb" is how we'll retrieve it from the MessageGroup later.
        rgb_out.link(sync.inputs["rgb"])

        # Connect depth output to sync's "depth" input slot.
        stereo.depth.link(sync.inputs["depth"])

        # Create the output queue for the sync node.
        # v3 API: call .createOutputQueue() directly on the node output.
        # maxSize=4 — keep 4 bundles buffered max.
        # blocking=False — drop old bundles if we fall behind (stay real-time).
        self.sync_queue = sync.out.createOutputQueue(
            maxSize=4, blocking=False
        )


        # ── Create IMU Node ───────────────────────────────────────────────────
        imu_node = self.pipeline.create(dai.node.IMU)

        # enableIMUSensor(sensor_type, rate_hz) — enable a sensor at a rate.
        # ACCELEROMETER_RAW: raw accelerometer, includes gravity, in m/s².
        # We use RAW here because LINEAR_ACCELERATION (gravity-removed) is
        # less stable on some firmware versions.
        # 480 Hz is the closest supported rate to 500Hz on the BNO085.
        imu_node.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 480)

        # GYROSCOPE_RAW: raw rotation rate in rad/s.
        # 400 Hz is the max supported rate for the gyroscope.
        imu_node.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)

        # setBatchReportThreshold(1): send IMU packets immediately,
        # don't wait to fill a batch. Minimises latency.
        imu_node.setBatchReportThreshold(1)

        # setMaxBatchReports(10): if our code falls behind, buffer up to
        # 10 packets before dropping. Prevents overloading the USB bus.
        imu_node.setMaxBatchReports(10)

        # Create IMU output queue — directly on imu_node.out in v3.
        # maxSize=50 because IMU runs at 400-480Hz, much faster than 30Hz loop.
        # A large buffer prevents IMU readings from being dropped.
        self.imu_queue = imu_node.out.createOutputQueue(
            maxSize=50, blocking=False
        )


        # ── Start the Pipeline ────────────────────────────────────────────────
        # pipeline.start() deploys the pipeline to the OAK-D Lite and
        # starts all nodes running. From this moment, data is flowing.
        logger.info("Starting OAK-D Lite device...")
        self.pipeline.start()
        logger.info("Device started successfully.")


        # ── Initialise Kalman Filter ──────────────────────────────────────────
        self.init_kalman()


        # ── Start IMU Background Thread ───────────────────────────────────────
        # The IMU produces data at 400-480Hz but our main loop runs at ~30Hz.
        # A background thread continuously reads IMU packets and stores the
        # latest one. The main loop reads that stored value — always fresh,
        # never blocking.
        self._running = True
        self._imu_thread = threading.Thread(
            target=self._imu_reader_thread,
            daemon=True   # auto-killed when main program exits
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
        Observes: [x, y] — 2 measurement variables (only position is visible).

        The filter uses velocity to predict where an object will be next frame,
        even when the camera temporarily loses it.
        """

        # KalmanFilter(dynamParams=4, measureParams=2)
        # 4 state variables: x, y, vx, vy
        # 2 measurement variables: x, y
        self.kalman = cv2.KalmanFilter(4, 2)

        # Transition matrix — encodes "position += velocity × 1 frame"
        # Row 0: x_new  = x + vx
        # Row 1: y_new  = y + vy
        # Row 2: vx_new = vx (constant velocity assumption)
        # Row 3: vy_new = vy
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix — maps state [x,y,vx,vy] to observation [x,y]
        # We can only observe position, not velocity.
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise — how much we trust our own physics prediction.
        # np.eye(4) = 4×4 identity matrix (1s on diagonal, 0s elsewhere).
        self.kalman.processNoiseCov = (
            np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        )

        # Measurement noise — how much we trust the raw camera readings.
        self.kalman.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
        )

        # Initial error covariance — our starting uncertainty about the state.
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        logger.info("Kalman filter initialised.")


    # =========================================================================
    # KALMAN FILTER OPERATIONS
    # =========================================================================

    def kalman_update(self, x: float, y: float) -> tuple:
        """
        Feeds a new measured position into the Kalman filter.
        Returns the corrected, smoothed position estimate.

        Call this every frame when YOLO gives us a fresh detection position.

        Arguments:
            x: measured pixel x-coordinate
            y: measured pixel y-coordinate

        Returns:
            (smoothed_x, smoothed_y)
        """

        # Step 1: predict where the object should be this frame
        # based on its last known position and velocity.
        self.kalman.predict()

        # Step 2: create the measurement vector — a 2×1 column matrix.
        # [[x],  ← pixel x position
        #  [y]]  ← pixel y position
        measurement = np.array([[x], [y]], dtype=np.float32)

        # Step 3: correct() blends the prediction with the measurement.
        # Returns the best estimate of true position: [x, y, vx, vy].
        corrected = self.kalman.correct(measurement)

        # Extract just x and y from the 4-element state vector.
        return float(corrected[0][0]), float(corrected[1][0])


    def kalman_predict(self) -> tuple:
        """
        Predicts the next position of a tracked object with NO new measurement.

        Call this when a tracked object temporarily disappears —
        instead of immediately dropping it, the filter predicts where it went.

        Returns:
            (predicted_x, predicted_y)
        """

        # predict() uses the transition matrix to advance the state:
        # x_new = x + vx,  y_new = y + vy
        predicted = self.kalman.predict()
        return float(predicted[0][0]), float(predicted[1][0])


    # =========================================================================
    # IMU BACKGROUND THREAD
    # =========================================================================

    def _imu_reader_thread(self):
        """
        Runs in background, continuously reading IMU packets at 400-480Hz.
        Stores the latest reading in self._latest_imu.

        The underscore prefix means "private — only call from inside this class."
        """

        logger.info("IMU reader thread started.")

        while self._running:
            try:
                # tryGet() returns the next IMU message or None immediately.
                # We use tryGet (non-blocking) so the thread never freezes.
                imu_data = self.imu_queue.tryGet()

                if imu_data is None:
                    # Nothing available yet — sleep briefly to avoid
                    # burning 100% CPU in a tight empty loop.
                    time.sleep(0.002)
                    continue

                # imu_data is a dai.IMUData object.
                # .packets is a list of IMUPacket objects in this batch.
                # We configured batchReportThreshold=1 so usually just one.
                for packet in imu_data.packets:

                    # packet.acceleroMeter is a IMUReportAccelerometer object.
                    # It has .x, .y, .z attributes in m/s².
                    accel = packet.acceleroMeter

                    # packet.gyroscope is a IMUReportGyroscope object.
                    # It has .x, .y, .z attributes in rad/s.
                    gyro  = packet.gyroscope

                    # getTimestamp() returns a timedelta object.
                    # .total_seconds() × 1000 converts to milliseconds.
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

                    # Acquire lock before writing — prevents the main thread
                    # from reading a half-written value simultaneously.
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

        In v3, the Sync node outputs a dai.MessageGroup.
        We access individual streams by the names we set when linking:
            group["rgb"]   → the RGB ImgFrame
            group["depth"] → the depth ImgFrame

        Returns a dictionary or None if no new bundle is ready yet.
        None is normal — just skip this iteration of the main loop.
        """

        if not self.pipeline.isRunning():
            logger.error("Pipeline is not running.")
            return None

        try:
            # tryGet() is non-blocking — returns None if nothing is ready.
            # This keeps the main loop running at full speed.
            msg_group = self.sync_queue.tryGet()

            if msg_group is None:
                return None

            # msg_group is a dai.MessageGroup.
            # Access each stream by the name we used when linking to sync.inputs[].
            rgb_msg   = msg_group["rgb"]
            depth_msg = msg_group["depth"]

            # getCvFrame() converts the depthai ImgFrame to a numpy array
            # in OpenCV format: shape (H, W, 3) uint8 for RGB.
            rgb = rgb_msg.getCvFrame()

            # getFrame() returns a numpy array of uint16 depth values in mm.
            # Shape is (H, W) — one depth value per pixel.
            depth = depth_msg.getFrame()

            # np.clip clamps all depth values to our configured range.
            # Pixels with 0 = no valid depth data.
            # Pixels above DEPTH_MAX_MM = too far to be relevant, set to max.
            depth = np.clip(depth, 0, DEPTH_MAX_MM).astype(np.uint16)

            # Safely read the latest IMU value from the background thread.
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
        """
        Returns the latest IMU reading safely from outside the class.
        Thread-safe — handles the lock internally.
        """
        with self._imu_lock:
            return self._latest_imu.copy()


    def update_missed_frames(self, active_ids: list) -> list:
        """
        Tracks consecutive missed frames per tracked object ID.

        Call every frame with the list of object IDs currently detected.
        Returns a list of IDs that have been missing too long and should
        be dropped from tracking entirely.
        """

        # Increment counter for all tracked objects.
        for obj_id in list(self.missed_frames.keys()):
            self.missed_frames[obj_id] += 1

        # Reset to 0 for objects still visible this frame.
        for obj_id in active_ids:
            self.missed_frames[obj_id] = 0

        # Find objects missing for too many consecutive frames.
        to_drop = [
            obj_id for obj_id, count in self.missed_frames.items()
            if count > KALMAN_MAX_MISSED_FRAMES
        ]

        # Remove them from tracking.
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
        Always call this when ECHORA exits.
        """

        logger.info("Releasing camera resources...")

        # Signal the IMU thread to stop.
        self._running = False

        # Wait up to 2 seconds for the thread to finish cleanly.
        if hasattr(self, '_imu_thread') and self._imu_thread.is_alive():
            self._imu_thread.join(timeout=2.0)

        # Stop the pipeline — this releases the USB connection to the camera.
        # In v3, pipeline.stop() is the clean shutdown method.
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
# Run directly to verify the camera works:
#   python camera.py

if __name__ == "__main__":

    print("=== ECHORA camera.py self-test (depthai v3) ===\n")

    cam = EchoraCamera()
    cam.init_pipeline()

    frame_count = 0

    try:
        # pipeline.isRunning() is the v3 way to check if pipeline is active.
        while cam.pipeline.isRunning():
            bundle = cam.get_synced_bundle()

            if bundle is None:
                continue

            frame_count += 1
            rgb   = bundle["rgb"]
            depth = bundle["depth"]
            imu   = bundle["imu"]

            if frame_count % 10 == 0:
                h, w     = depth.shape
                center_d = depth[h // 2, w // 2]
                accel    = imu["accel"]
                gyro     = imu["gyro"]

                print(f"Frame {frame_count:4d} | "
                      f"Depth centre: {center_d:5d}mm | "
                      f"Accel x:{accel['x']:+.2f} y:{accel['y']:+.2f} z:{accel['z']:+.2f} | "
                      f"Gyro  y:{gyro['y']:+.3f} rad/s")

                sx, sy = cam.kalman_update(float(w // 2), float(h // 2))
                print(f"             | Kalman smoothed: ({sx:.1f}, {sy:.1f})")

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